import copy
import csv
import json
from pathlib import Path

import torch
import torch.nn as nn

from ultralytics.models.yolo.segment import SegmentationValidator
from ultralytics.nn.distill_model import DistillationModel
from ultralytics.nn.modules import Segment
from ultralytics.nn.tasks import SegmentationModel


# ============================================================
# Fixed checkpoints and independent test-set paths
# ============================================================

TEACHER = Path(
    "D:/CrackSegRuns/"
    "yolov8n_seg_spdconv_sppfmsa_100e/"
    "weights/best.pt"
)

RAW_PRUNED = Path(
    "D:/CrackSegRuns/"
    "pruning_dryrun_10_v2/"
    "teacher_pruned_10_v2_raw.pth"
)

PRUNED_FT = Path(
    "D:/CrackSegRuns/"
    "pruned_student_10_finetune_30e/"
    "weights/best.pt"
)

PRUNED_KD = Path(
    "D:/CrackSegRuns/"
    "pruned_student_10_kd_30e/"
    "weights/best.pt"
)

DATA = "D:/crack-seg/crack-seg-local.yaml"
SPLIT = "test"
OUTPUT_ROOT = Path("D:/CrackSegRuns/final_compression_test")


# ============================================================
# Unified independent-test settings
# ============================================================

IMAGE_SIZE = 640
BATCH_SIZE = 16
DEVICE = 0
WORKERS = 0
PLOTS = False

EXPECTED_PRUNED_PARAMS = 3_742_841


# ============================================================
# Fixed model list
# ============================================================

MODELS = [
    {
        "name": "Teacher",
        "slug": "teacher",
        "path": TEACHER,
        "is_pruned": False,
    },
    {
        "name": "Raw Pruned",
        "slug": "raw_pruned",
        "path": RAW_PRUNED,
        "is_pruned": True,
    },
    {
        "name": "Pruned + FT",
        "slug": "pruned_ft",
        "path": PRUNED_FT,
        "is_pruned": True,
    },
    {
        "name": "Pruned + KD",
        "slug": "pruned_kd",
        "path": PRUNED_KD,
        "is_pruned": True,
    },
]


# ============================================================
# Utility functions
# ============================================================

def count_parameters(model):
    return sum(parameter.numel() for parameter in model.parameters())


def conv_channel_signature(model):
    """Record every Conv2d channel/group structure for pruned-model checks."""
    return {
        name: (
            module.in_channels,
            module.out_channels,
            module.kernel_size,
            module.groups,
        )
        for name, module in model.named_modules()
        if isinstance(module, nn.Conv2d)
    }


def to_float(value):
    """Safely convert a torch/numpy/Python scalar to float."""
    if value is None:
        return None

    if torch.is_tensor(value):
        if value.numel() != 1:
            return None
        return float(value.detach().cpu().item())

    try:
        return float(value)
    except (TypeError, ValueError):
        return None


# ============================================================
# Load model object without YAML reconstruction
# ============================================================

def load_model_object(path):
    """
    Load one of the fixed deployment checkpoints.

    Supported formats:
      1. Normal Ultralytics .pt checkpoint.
      2. Directly saved full SegmentationModel .pth object.
      3. A checkpoint whose model/ema is a DistillationModel wrapper.

    Raw Pruned is loaded directly from its serialized Python model object;
    it is never reconstructed from YAML.
    """
    if not path.is_file():
        raise FileNotFoundError(f"Model not found:\n{path}")

    obj = torch.load(path, map_location="cpu", weights_only=False)

    if isinstance(obj, nn.Module):
        model = obj
    elif isinstance(obj, dict):
        model = obj.get("ema", None)
        if model is None:
            model = obj.get("model", None)
        if model is None:
            raise RuntimeError(f"No 'ema' or 'model' object found in:\n{path}")
    else:
        raise TypeError(f"Unsupported checkpoint object: {type(obj)}")

    if isinstance(model, DistillationModel):
        print(
            "NOTE: DistillationModel detected. "
            "Extracting student_model for evaluation."
        )
        model = model.student_model

    # Python class and final head are authoritative. Older/directly saved
    # segmentation models may legitimately have no `task` metadata.
    if not isinstance(model, SegmentationModel):
        raise TypeError(
            "Expected SegmentationModel, got "
            f"{type(model).__module__}.{type(model).__name__}"
        )

    model_sequence = getattr(model, "model", None)
    try:
        head = model_sequence[-1]
    except (TypeError, IndexError, KeyError):
        head = None

    if not isinstance(head, Segment):
        raise TypeError(
            "SegmentationModel does not end with a Segment head. "
            f"Final head is {type(head).__module__}.{type(head).__name__}"
        )

    task = getattr(model, "task", None)
    if task is None:
        print(
            "NOTE: SegmentationModel with Segment head has no task "
            "metadata; setting model.task='segment'."
        )
        model.task = "segment"
    elif task != "segment":
        raise RuntimeError(
            "Structurally valid segmentation model has conflicting "
            f"task metadata: task={task!r}"
        )

    model = model.float()
    model.cpu()
    model.eval()
    return model


# ============================================================
# Model size and operation-count helpers
# ============================================================

def get_fused_parameter_count(model):
    """Fuse a deep copy, preserving the original loaded checkpoint object."""
    try:
        fused_model = copy.deepcopy(model)
        fused_model.cpu()
        fused_model.eval()

        if hasattr(fused_model, "fuse"):
            result = fused_model.fuse(verbose=False)
            if result is not None:
                fused_model = result

        return count_parameters(fused_model)
    except Exception as exc:
        print(f"WARNING: fused parameter count failed: {exc}")
        return None


def get_macs(model):
    """Return Torch-Pruning operation count as MACs (not GFLOPs)."""
    try:
        import torch_pruning as tp

        count_model = copy.deepcopy(model)
        count_model.cpu()
        count_model.float()
        count_model.eval()

        example = torch.zeros(1, 3, IMAGE_SIZE, IMAGE_SIZE)
        with torch.no_grad():
            macs, params = tp.utils.count_ops_and_params(
                count_model,
                example_inputs=example,
            )

        return {
            "macs": float(macs),
            "macs_g": float(macs) / 1e9,
            "tp_params": int(params),
        }
    except Exception as exc:
        print(f"WARNING: MAC counting failed: {exc}")
        return {"macs": None, "macs_g": None, "tp_params": None}


def get_ultralytics_gflops(model):
    """Return Ultralytics' GFLOPs estimate; keep it separate from MACs."""
    try:
        from ultralytics.utils.torch_utils import get_flops

        count_model = copy.deepcopy(model)
        count_model.cpu()
        count_model.float()
        count_model.eval()

        return to_float(get_flops(count_model, imgsz=IMAGE_SIZE))
    except Exception as exc:
        print(f"WARNING: Ultralytics GFLOPs calculation failed: {exc}")
        return None


# ============================================================
# Metric and speed extraction
# ============================================================

def extract_metrics(validator, validation_result):
    """Prefer validator.metrics.results_dict, with result-object fallbacks."""
    metrics_dict = None
    metrics_obj = getattr(validator, "metrics", None)

    if metrics_obj is not None:
        results_dict = getattr(metrics_obj, "results_dict", None)
        if results_dict is not None:
            metrics_dict = results_dict

    if metrics_dict is None:
        if isinstance(validation_result, dict):
            metrics_dict = validation_result
        else:
            results_dict = getattr(validation_result, "results_dict", None)
            if results_dict is not None:
                metrics_dict = results_dict

    if metrics_dict is None:
        raise RuntimeError("Could not extract test metrics.")

    def metric(key):
        return to_float(metrics_dict.get(key, None))

    return {
        "box_precision": metric("metrics/precision(B)"),
        "box_recall": metric("metrics/recall(B)"),
        "box_map50": metric("metrics/mAP50(B)"),
        "box_map50_95": metric("metrics/mAP50-95(B)"),
        "mask_precision": metric("metrics/precision(M)"),
        "mask_recall": metric("metrics/recall(M)"),
        "mask_map50": metric("metrics/mAP50(M)"),
        "mask_map50_95": metric("metrics/mAP50-95(M)"),
    }


def extract_speed(validator, validation_result):
    """Extract speed values in milliseconds per image."""
    speed = getattr(validator, "speed", None)

    if not speed:
        metrics_obj = getattr(validator, "metrics", None)
        if metrics_obj is not None:
            speed = getattr(metrics_obj, "speed", None)

    if not speed:
        speed = getattr(validation_result, "speed", None)

    if not isinstance(speed, dict):
        return {
            "preprocess_ms": None,
            "inference_ms": None,
            "loss_ms": None,
            "postprocess_ms": None,
        }

    return {
        "preprocess_ms": to_float(speed.get("preprocess", None)),
        "inference_ms": to_float(speed.get("inference", None)),
        "loss_ms": to_float(speed.get("loss", None)),
        "postprocess_ms": to_float(speed.get("postprocess", None)),
    }


# ============================================================
# Unified independent test
# ============================================================

def validate_model(name, slug, model):
    """Evaluate one fixed checkpoint on the independent test split."""
    eval_model = copy.deepcopy(model)
    eval_model.float()
    eval_model.eval()

    model_output_dir = OUTPUT_ROOT / slug
    model_output_dir.mkdir(parents=True, exist_ok=True)

    args = {
        "task": "segment",
        "mode": "val",
        "data": DATA,
        "split": SPLIT,
        "imgsz": IMAGE_SIZE,
        "batch": BATCH_SIZE,
        "device": DEVICE,
        "workers": WORKERS,
        "plots": PLOTS,
        "verbose": True,
        "save_json": False,
        "save_txt": False,
        "save_conf": False,
        "project": str(OUTPUT_ROOT),
        "name": slug,
        "exist_ok": True,
    }

    validator = SegmentationValidator(args=args)

    print("\n" + "=" * 90)
    print(f"Independent test evaluation: {name}")
    print("=" * 90)

    with torch.no_grad():
        validation_result = validator(model=eval_model)

    metrics = extract_metrics(validator, validation_result)
    speed = extract_speed(validator, validation_result)
    return metrics, speed


# ============================================================
# Pretty-print helpers
# ============================================================

def format_value(value, digits=5):
    if value is None:
        return "N/A"
    return f"{value:.{digits}f}"


# ============================================================
# Main
# ============================================================

def main():
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    print("=" * 90)
    print("FINAL COMPRESSION PIPELINE - INDEPENDENT TEST SET")
    print("=" * 90)
    print(f"Dataset: {DATA}")
    print(f"Split: {SPLIT}")
    print(f"Image size: {IMAGE_SIZE}")
    print(f"Batch: {BATCH_SIZE}")
    print(f"Device: {DEVICE}")
    print(f"Workers: {WORKERS}")
    print(f"Plots: {PLOTS}")
    print()

    # Load the directly serialized raw-pruned model once as the structural
    # reference. No YAML reconstruction is permitted here.
    print("Loading raw-pruned reference for structural comparison...")
    raw_reference = load_model_object(RAW_PRUNED)
    raw_reference_signature = conv_channel_signature(raw_reference)
    raw_reference_params = count_parameters(raw_reference)

    if raw_reference_params != EXPECTED_PRUNED_PARAMS:
        raise RuntimeError(
            "Raw-pruned reference parameter count mismatch.\n"
            f"Expected: {EXPECTED_PRUNED_PARAMS:,}\n"
            f"Actual: {raw_reference_params:,}"
        )

    del raw_reference
    summaries = []

    for spec in MODELS:
        print("\n" + "#" * 90)
        print(f"# {spec['name']}")
        print(f"# {spec['path']}")
        print("#" * 90)

        model = load_model_object(spec["path"])
        params = count_parameters(model)
        fused_params = get_fused_parameter_count(model)
        signature = conv_channel_signature(model)

        if spec["is_pruned"]:
            same_pruned_structure = signature == raw_reference_signature

            if params != EXPECTED_PRUNED_PARAMS:
                raise RuntimeError(
                    f"{spec['name']} parameter count mismatch.\n"
                    f"Expected: {EXPECTED_PRUNED_PARAMS:,}\n"
                    f"Actual: {params:,}"
                )

            if not same_pruned_structure:
                raise RuntimeError(
                    f"{spec['name']} Conv2d structure is different from "
                    "Raw Pruned Student."
                )
        else:
            same_pruned_structure = None

        print(f"Params:       {params:,}")
        print(
            "Fused Params: "
            + (f"{fused_params:,}" if fused_params is not None else "N/A")
        )
        if spec["is_pruned"]:
            print(f"Same pruned Conv structure: {same_pruned_structure}")

        mac_info = get_macs(model)
        gflops = get_ultralytics_gflops(model)

        print(
            "MACs: "
            + (
                f"{mac_info['macs_g']:.4f} G"
                if mac_info["macs_g"] is not None
                else "N/A"
            )
        )
        print(
            "Ultralytics GFLOPs: "
            + (f"{gflops:.4f}" if gflops is not None else "N/A")
        )

        metrics, speed = validate_model(
            spec["name"],
            spec["slug"],
            model,
        )

        summary = {
            "model": spec["name"],
            "checkpoint": str(spec["path"]),
            "params": params,
            "params_m": params / 1e6,
            "fused_params": fused_params,
            "fused_params_m": (
                fused_params / 1e6 if fused_params is not None else None
            ),
            "same_pruned_structure_as_raw": same_pruned_structure,
            "macs": mac_info["macs"],
            "macs_g": mac_info["macs_g"],
            "torch_pruning_param_count": mac_info["tp_params"],
            "ultralytics_gflops": gflops,
            **metrics,
            **speed,
        }
        summaries.append(summary)

        print("\nFixed-checkpoint independent-test results:")
        print(f"  Box P:         {format_value(metrics['box_precision'])}")
        print(f"  Box R:         {format_value(metrics['box_recall'])}")
        print(f"  Box mAP50:     {format_value(metrics['box_map50'])}")
        print(f"  Box mAP50-95:  {format_value(metrics['box_map50_95'])}")
        print(f"  Mask P:        {format_value(metrics['mask_precision'])}")
        print(f"  Mask R:        {format_value(metrics['mask_recall'])}")
        print(f"  Mask mAP50:    {format_value(metrics['mask_map50'])}")
        print(f"  Mask mAP50-95: {format_value(metrics['mask_map50_95'])}")
        print(
            "  Inference:     "
            + (
                f"{speed['inference_ms']:.3f} ms/image"
                if speed["inference_ms"] is not None
                else "N/A"
            )
        )

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Save a test-only JSON summary.
    json_path = OUTPUT_ROOT / "final_compression_test_summary.json"
    json_payload = {
        "evaluation_protocol": {
            "evaluation_type": "independent_test_set",
            "data": DATA,
            "split": SPLIT,
            "imgsz": IMAGE_SIZE,
            "batch": BATCH_SIZE,
            "device": DEVICE,
            "workers": WORKERS,
            "plots": PLOTS,
            "note": (
                "All metrics are independent test-set results from the four "
                "fixed checkpoints evaluated under one unified configuration. "
                "Validation-set results and per-epoch training peaks are "
                "intentionally excluded."
            ),
        },
        "models": summaries,
    }
    json_path.write_text(
        json.dumps(json_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    # Save a test-only CSV summary.
    csv_path = OUTPUT_ROOT / "final_compression_test_summary.csv"
    fieldnames = [
        "model",
        "checkpoint",
        "params",
        "params_m",
        "fused_params",
        "fused_params_m",
        "same_pruned_structure_as_raw",
        "macs",
        "macs_g",
        "torch_pruning_param_count",
        "ultralytics_gflops",
        "box_precision",
        "box_recall",
        "box_map50",
        "box_map50_95",
        "mask_precision",
        "mask_recall",
        "mask_map50",
        "mask_map50_95",
        "preprocess_ms",
        "inference_ms",
        "loss_ms",
        "postprocess_ms",
    ]

    with csv_path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
            extrasaction="ignore",
        )
        writer.writeheader()
        for summary in summaries:
            writer.writerow(summary)

    # Final fixed-checkpoint comparison table.
    print("\n" + "=" * 120)
    print("INDEPENDENT TEST SET - FINAL FIXED-CHECKPOINT COMPARISON")
    print("=" * 120)

    header = (
        f"{'Model':<18}"
        f"{'Params(M)':>12}"
        f"{'MACs(G)':>11}"
        f"{'Box50-95':>12}"
        f"{'Mask50':>11}"
        f"{'Mask50-95':>13}"
        f"{'Infer(ms)':>12}"
    )
    print(header)
    print("-" * 120)

    for result in summaries:
        params_text = f"{result['params_m']:.3f}"
        macs_text = (
            f"{result['macs_g']:.3f}"
            if result["macs_g"] is not None
            else "N/A"
        )
        box_text = format_value(result["box_map50_95"])
        mask50_text = format_value(result["mask_map50"])
        mask5095_text = format_value(result["mask_map50_95"])
        inference_text = (
            f"{result['inference_ms']:.2f}"
            if result["inference_ms"] is not None
            else "N/A"
        )

        print(
            f"{result['model']:<18}"
            f"{params_text:>12}"
            f"{macs_text:>11}"
            f"{box_text:>12}"
            f"{mask50_text:>11}"
            f"{mask5095_text:>13}"
            f"{inference_text:>12}"
        )

    print("=" * 120)

    lookup = {result["model"]: result for result in summaries}
    teacher = lookup["Teacher"]
    raw = lookup["Raw Pruned"]
    ft = lookup["Pruned + FT"]
    kd = lookup["Pruned + KD"]

    print("\nIndependent-test Mask mAP50-95 comparisons:")
    print(f"Teacher:      {format_value(teacher['mask_map50_95'])}")
    print(f"Raw Pruned:   {format_value(raw['mask_map50_95'])}")
    print(f"Pruned + FT:  {format_value(ft['mask_map50_95'])}")
    print(f"Pruned + KD:  {format_value(kd['mask_map50_95'])}")

    if kd["mask_map50_95"] is not None and ft["mask_map50_95"] is not None:
        delta = kd["mask_map50_95"] - ft["mask_map50_95"]
        print(f"\nKD - FT:      {delta:+.5f}")

    if (
        kd["mask_map50_95"] is not None
        and teacher["mask_map50_95"] is not None
    ):
        delta = kd["mask_map50_95"] - teacher["mask_map50_95"]
        print(f"KD - Teacher: {delta:+.5f}")

    print("\nIMPORTANT:")
    print(
        "The table above contains independent test-set results for the four "
        "fixed checkpoints under one unified protocol."
    )
    print(
        "Do not mix these values with validation results or independent "
        "per-epoch peak metrics from training logs."
    )
    print(f"\nJSON summary:\n{json_path}")
    print(f"\nCSV summary:\n{csv_path}")
    print("\nIndependent test evaluation completed.")


if __name__ == "__main__":
    main()
