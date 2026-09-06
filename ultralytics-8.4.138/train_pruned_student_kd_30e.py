import csv
import json
import math
import time
from pathlib import Path

import torch
import torch.nn as nn

from ultralytics.models.yolo.segment import SegmentationTrainer
from ultralytics.nn.distill_model import DistillationModel
from ultralytics.nn.tasks import SegmentationModel
from ultralytics.utils.torch_utils import unwrap_model


# Paths
TEACHER = Path("D:/CrackSegRuns/yolov8n_seg_spdconv_sppfmsa_100e/weights/best.pt")
PRUNED_STUDENT = Path("D:/CrackSegRuns/pruning_dryrun_10_v2/teacher_pruned_10_v2_raw.pth")
DATA = "D:/crack-seg/crack-seg-local.yaml"
PROJECT = "D:/CrackSegRuns"
NAME = "pruned_student_10_kd_30e"


# Formal experiment settings
EPOCHS = 30
IMAGE_SIZE = 640
BATCH_SIZE = 16
DEVICE = 0
WORKERS = 0
OPTIMIZER = "AdamW"
LR0 = 0.001
WEIGHT_DECAY = 0.0005
DISTILL_WEIGHT = 6.0
FRACTION = 1.0
PLOTS = True


# Structural guardrails
EXPECTED_STUDENT_PARAMS = 3_742_841
TEACHER_REFERENCE_PARAMS = 4_494_915


def count_parameters(model):
    """Return the total number of parameters in a model."""
    return sum(parameter.numel() for parameter in model.parameters())


def count_trainable_tensors(model):
    """Return the number of parameter tensors that require gradients."""
    return sum(parameter.requires_grad for parameter in model.parameters())


def conv_channel_signature(model):
    """Capture every Conv2d channel shape so pruning can be verified after training."""
    return {
        name: (module.in_channels, module.out_channels, module.kernel_size, module.groups)
        for name, module in model.named_modules()
        if isinstance(module, nn.Conv2d)
    }


def load_pruned_student():
    """Load and validate the complete raw pruned Student object without YAML reconstruction."""
    if not PRUNED_STUDENT.is_file():
        raise FileNotFoundError(f"Pruned Student not found:\n{PRUNED_STUDENT}")

    student = torch.load(PRUNED_STUDENT, map_location="cpu", weights_only=False)

    if not isinstance(student, SegmentationModel):
        raise TypeError(
            "Expected a complete SegmentationModel, got "
            f"{type(student).__module__}.{type(student).__name__}"
        )
    if getattr(student, "task", None) != "segment":
        raise RuntimeError(f"Expected task='segment', got {getattr(student, 'task', None)!r}")

    params = count_parameters(student)

    print("=" * 80)
    print("Loading raw structurally-pruned Student")
    print("=" * 80)
    print(f"Source: {PRUNED_STUDENT}")
    print(f"Type:   {type(student).__module__}.{type(student).__name__}")
    print(f"Params: {params:,} ({params / 1e6:.6f} M)")

    if params != EXPECTED_STUDENT_PARAMS:
        raise RuntimeError(
            "\nUnexpected Student parameter count.\n"
            f"Expected: {EXPECTED_STUDENT_PARAMS:,}\n"
            f"Actual:   {params:,}\n\n"
            "Stop training to avoid accidentally using a rebuilt or different network."
        )
    if params >= TEACHER_REFERENCE_PARAMS:
        raise RuntimeError("Loaded Student is not smaller than Teacher. The pruned structure may have been lost.")

    batchnorm_count = sum(isinstance(module, nn.BatchNorm2d) for module in student.modules())
    print(f"BatchNorm2d layers: {batchnorm_count}")
    if batchnorm_count == 0:
        print("NOTE: Student is already fused. This formal KD run trains the fused, structurally-pruned Student.")

    head = student.model[-1]
    proto_channels = head.proto.cv3.conv.out_channels
    coefficient_channels = [branch[-1].out_channels for branch in head.cv4]
    print(f"Mask prototype channels:   {proto_channels}")
    print(f"Mask coefficient channels: {coefficient_channels}")

    if proto_channels != int(head.nm):
        raise RuntimeError("Mask prototype output dimension does not match head.nm.")
    if any(channels != int(head.nm) for channels in coefficient_channels):
        raise RuntimeError("Mask coefficient output dimension does not match head.nm.")

    before_enable = count_trainable_tensors(student)
    print(f"Trainable Student parameter tensors before enable: {before_enable}")
    student.requires_grad_(True)
    after_enable = count_trainable_tensors(student)
    print(f"Trainable Student parameter tensors after enable:  {after_enable}")
    if after_enable == 0:
        raise RuntimeError("No trainable Student parameters after requires_grad_(True).")

    # Recreate the segmentation loss with this run's training arguments.
    student.criterion = None
    student.train()
    return student


class PrunedStudentKDTrainer(SegmentationTrainer):
    """Train the in-memory pruned Student with native Ultralytics knowledge distillation."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.distillation_wrapper_verified = False
        self.teacher_frozen_verified = False
        self.student_backward_verified = False
        self.student_optimizer_update_verified = False
        self.projector_backward_verified = False
        self.first_updated_student_parameter = None
        self.first_student_update_max_abs_delta = 0.0

    def setup_model(self):
        """Reject any attempt to replace the injected pruned Student with a YAML-built model."""
        if not isinstance(self.model, SegmentationModel):
            raise TypeError(
                "Trainer did not receive the in-memory pruned SegmentationModel before setup. "
                "Refusing to rebuild Student from YAML."
            )
        if count_parameters(self.model) != EXPECTED_STUDENT_PARAMS:
            raise RuntimeError("Student parameter count changed before DistillationModel wrapping.")

        print("\nTrainer setup:")
        print("  Using in-memory raw pruned Student.")
        print("  Native Ultralytics distillation will be enabled later inside _setup_train().")

        # For an in-memory nn.Module, the parent keeps the supplied object rather than rebuilding from YAML.
        return super().setup_model()

    def _setup_train(self):
        """Run native setup first, then validate the DistillationModel that it creates."""
        super()._setup_train()

        # This check belongs after the parent call: BaseTrainer creates the native KD wrapper there.
        kd_model = unwrap_model(self.model)
        if not isinstance(kd_model, DistillationModel):
            raise RuntimeError(
                "\nNative DistillationModel was NOT created after SegmentationTrainer._setup_train().\n"
                f"distill_model={self.args.distill_model!r}"
            )
        self.distillation_wrapper_verified = True
        print("\nDistillationModel successfully created.")

        wrapped_student_params = count_parameters(kd_model.student_model)
        print(f"Wrapped Student params: {wrapped_student_params:,}")
        if wrapped_student_params != EXPECTED_STUDENT_PARAMS:
            raise RuntimeError("The wrapped Student is not the expected structurally-pruned Student.")

        teacher = kd_model.teacher_model
        if teacher is None:
            raise RuntimeError("DistillationModel contains no Teacher.")

        teacher_params = count_parameters(teacher)
        teacher_trainable = count_trainable_tensors(teacher)
        teacher_is_eval = not teacher.training
        print(f"Teacher params: {teacher_params:,}")
        print(f"Teacher trainable parameter tensors: {teacher_trainable}")
        print(f"Teacher eval mode: {teacher_is_eval}")

        self.teacher_frozen_verified = teacher_trainable == 0 and teacher_is_eval
        if not self.teacher_frozen_verified:
            raise RuntimeError("Teacher is not fully frozen in eval mode.")

        print(f"Distillation feature layer indices: {kd_model.feats_idx}")
        print("KD projectors:")
        for index, projector in enumerate(kd_model.projector):
            first_conv = projector[0]
            second_conv = projector[2]
            print(
                f"  Level {index}: {first_conv.in_channels} -> "
                f"{first_conv.out_channels} -> {second_conv.out_channels}"
            )
        print(f"Distillation weight dis = {kd_model.dis}")
        if not math.isclose(float(kd_model.dis), DISTILL_WEIGHT, rel_tol=0.0, abs_tol=1e-12):
            raise RuntimeError(f"Unexpected distillation weight: {kd_model.dis}")

    def optimizer_step(self):
        """Verify gradients and one real Student parameter update around the normal optimizer step."""
        tracked_name = None
        tracked_parameter = None
        kd_model = unwrap_model(self.model)

        if isinstance(kd_model, DistillationModel):
            for name, parameter in kd_model.student_model.named_parameters():
                gradient = parameter.grad
                if (
                    parameter.requires_grad
                    and gradient is not None
                    and torch.isfinite(gradient).all()
                    and torch.count_nonzero(gradient).item() > 0
                ):
                    self.student_backward_verified = True
                    tracked_name = name
                    tracked_parameter = parameter
                    break

            for parameter in kd_model.projector.parameters():
                gradient = parameter.grad
                if (
                    parameter.requires_grad
                    and gradient is not None
                    and torch.isfinite(gradient).all()
                    and torch.count_nonzero(gradient).item() > 0
                ):
                    self.projector_backward_verified = True
                    break

        before = None
        if tracked_parameter is not None and not self.student_optimizer_update_verified:
            before = tracked_parameter.detach().clone()

        super().optimizer_step()

        if before is not None:
            delta = (tracked_parameter.detach() - before).abs().max().item()
            self.first_updated_student_parameter = tracked_name
            self.first_student_update_max_abs_delta = delta
            if delta > 0.0:
                self.student_optimizer_update_verified = True


def find_column(columns, exact=None, contains=None, starts_with=None):
    """Find a stripped results.csv column by exact name or contained tokens."""
    if exact is not None:
        for column in columns:
            if column == exact:
                return column
    for column in columns:
        if starts_with is not None and not column.startswith(starts_with):
            continue
        if contains is not None and all(token in column for token in contains):
            return column
    return None


def read_results_csv(results_csv):
    """Parse formal KD metrics and select the best epoch by Mask mAP50-95."""
    with results_csv.open("r", encoding="utf-8-sig", newline="") as file:
        rows = [
            {key.strip(): value.strip() for key, value in row.items()}
            for row in csv.DictReader(file)
        ]

    if not rows:
        raise RuntimeError("results.csv contains no epochs.")

    columns = list(rows[0].keys())
    dis_loss_key = find_column(columns, contains=["dis_loss"])
    box_map50_key = find_column(columns, contains=["mAP50(B)"], starts_with="metrics/")
    box_map5095_key = find_column(columns, contains=["mAP50-95(B)"], starts_with="metrics/")
    mask_map50_key = find_column(columns, contains=["mAP50(M)"], starts_with="metrics/")
    mask_map5095_key = find_column(columns, contains=["mAP50-95(M)"], starts_with="metrics/")

    required = {
        "dis_loss": dis_loss_key,
        "Box mAP50": box_map50_key,
        "Box mAP50-95": box_map5095_key,
        "Mask mAP50": mask_map50_key,
        "Mask mAP50-95": mask_map5095_key,
    }
    missing = [name for name, key in required.items() if key is None]
    if missing:
        raise RuntimeError("Could not locate results.csv columns: " + ", ".join(missing))

    loss_keys = [key for key in columns if key.startswith("train/") and key.endswith("_loss")]
    if not loss_keys:
        raise RuntimeError("No training loss columns found in results.csv.")

    parsed_rows = []
    for row_number, row in enumerate(rows, start=1):
        epoch = int(float(row.get("epoch", row_number)))
        parsed = {
            "epoch": epoch,
            "box_map50": float(row[box_map50_key]),
            "box_map50_95": float(row[box_map5095_key]),
            "mask_map50": float(row[mask_map50_key]),
            "mask_map50_95": float(row[mask_map5095_key]),
            "dis_loss": float(row[dis_loss_key]),
        }
        all_losses = {key: float(row[key]) for key in loss_keys}
        numeric_values = [*parsed.values(), *all_losses.values()]
        if not all(math.isfinite(value) for value in numeric_values):
            raise RuntimeError(f"Non-finite metric or training loss at epoch {epoch}.")
        parsed_rows.append(parsed)

    if len(parsed_rows) != EPOCHS:
        raise RuntimeError(f"Expected {EPOCHS} epochs in results.csv, found {len(parsed_rows)}.")

    kd_losses = [row["dis_loss"] for row in parsed_rows]
    if not any(abs(value) > 0.0 for value in kd_losses):
        raise RuntimeError("Every recorded dis_loss is zero; knowledge distillation may not be active.")

    metric_names = ("box_map50", "box_map50_95", "mask_map50", "mask_map50_95")
    best_row = max(parsed_rows, key=lambda item: item["mask_map50_95"])
    independent_peaks = {
        metric: {
            "epoch": peak["epoch"],
            "value": peak[metric],
        }
        for metric in metric_names
        for peak in [max(parsed_rows, key=lambda item, metric=metric: item[metric])]
    }

    # Endpoint direction is informational only. It is deliberately not a pass/fail condition.
    return {
        "epochs_recorded": len(parsed_rows),
        "distillation_loss_column": dis_loss_key,
        "distillation_loss_first": kd_losses[0],
        "distillation_loss_last": kd_losses[-1],
        "distillation_loss_endpoint_decreased": kd_losses[-1] < kd_losses[0],
        "distillation_loss_finite": True,
        "distillation_loss_nonzero": True,
        "best_epoch_by_mask_map50_95": {
            key: best_row[key] for key in ("epoch", *metric_names)
        },
        "last_epoch": {
            key: parsed_rows[-1][key] for key in ("epoch", *metric_names)
        },
        "independent_peaks": independent_peaks,
    }


def load_student_from_checkpoint(checkpoint_path):
    """Load the trained Student, unwrapping a saved DistillationModel when necessary."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = checkpoint.get("ema") or checkpoint.get("model")
    if model is None:
        raise RuntimeError(f"No EMA/model object found in {checkpoint_path}")

    model = model.float()
    if isinstance(model, DistillationModel):
        model = model.student_model
    if not isinstance(model, SegmentationModel):
        raise TypeError("Saved checkpoint does not contain a SegmentationModel Student.")
    return model


def main():
    if not TEACHER.is_file():
        raise FileNotFoundError(f"Teacher not found:\n{TEACHER}")

    print("=" * 80)
    print("Teacher")
    print("=" * 80)
    print(f"Teacher checkpoint: {TEACHER}")

    student = load_pruned_student()
    params_before = count_parameters(student)
    structure_before = conv_channel_signature(student)

    overrides = {
        # Metadata only: trainer.model is replaced below with the actual in-memory module.
        "model": str(PRUNED_STUDENT),
        "data": DATA,
        "task": "segment",
        "epochs": EPOCHS,
        "imgsz": IMAGE_SIZE,
        "batch": BATCH_SIZE,
        "device": DEVICE,
        "workers": WORKERS,
        "optimizer": OPTIMIZER,
        "lr0": LR0,
        "weight_decay": WEIGHT_DECAY,
        "distill_model": str(TEACHER),
        "dis": DISTILL_WEIGHT,
        "pretrained": False,
        "resume": False,
        "freeze": None,
        "project": PROJECT,
        "name": NAME,
        "exist_ok": True,
        "val": True,
        "save": True,
        "plots": PLOTS,
        "amp": True,
        "cache": False,
        "fraction": FRACTION,
        "seed": 0,
        "deterministic": True,
        "compile": False,
    }

    trainer = PrunedStudentKDTrainer(overrides=overrides)

    # Critical: inject the structurally-pruned nn.Module itself. Never rebuild it from YAML.
    trainer.model = student

    print("\n" + "=" * 80)
    print(f"Starting formal {EPOCHS}-epoch native KD experiment")
    print("=" * 80)
    print(f"Teacher: {TEACHER}")
    print(f"Student: {PRUNED_STUDENT}")
    print(f"KD weight dis: {DISTILL_WEIGHT}")

    start_time = time.perf_counter()
    trainer.train()
    elapsed_seconds = time.perf_counter() - start_time

    results_csv = trainer.save_dir / "results.csv"
    if not results_csv.is_file():
        raise FileNotFoundError(f"Training results not found:\n{results_csv}")
    results_summary = read_results_csv(results_csv)

    if trainer.best.is_file():
        checkpoint_path = trainer.best
    elif trainer.last.is_file():
        checkpoint_path = trainer.last
    else:
        raise FileNotFoundError("Neither best.pt nor last.pt was produced.")

    trained_student = load_student_from_checkpoint(checkpoint_path)
    params_after = count_parameters(trained_student)
    structure_after = conv_channel_signature(trained_student)
    params_preserved = params_before == params_after == EXPECTED_STUDENT_PARAMS
    structure_preserved = structure_before == structure_after

    checks = {
        "native_distillation_model_wrapper": trainer.distillation_wrapper_verified,
        "teacher_frozen": trainer.teacher_frozen_verified,
        "student_backward_propagation": trainer.student_backward_verified,
        "student_optimizer_parameter_update": trainer.student_optimizer_update_verified,
        "kd_projector_backward_propagation": trainer.projector_backward_verified,
        "finite_kd_loss": results_summary["distillation_loss_finite"],
        "nonzero_kd_loss": results_summary["distillation_loss_nonzero"],
        "student_parameter_count_preserved": params_preserved,
        "student_conv2d_structure_preserved": structure_preserved,
    }

    report = {
        "experiment": "10% structurally-pruned Student native KD 30e",
        "teacher": str(TEACHER),
        "student_source": str(PRUNED_STUDENT),
        "output_dir": str(trainer.save_dir),
        "results_csv": str(results_csv),
        "checkpoint": str(checkpoint_path),
        "epochs_requested": EPOCHS,
        "imgsz": IMAGE_SIZE,
        "batch": BATCH_SIZE,
        "device": DEVICE,
        "workers": WORKERS,
        "optimizer": OPTIMIZER,
        "lr0": LR0,
        "weight_decay": WEIGHT_DECAY,
        "distillation_weight_dis": DISTILL_WEIGHT,
        "amp": True,
        "cache": False,
        "seed": 0,
        "deterministic": True,
        "pretrained": False,
        "resume": False,
        "freeze": None,
        "distillation_wrapper_verified": trainer.distillation_wrapper_verified,
        "teacher_frozen_verified": trainer.teacher_frozen_verified,
        "student_backward_verified": trainer.student_backward_verified,
        "student_optimizer_update_verified": trainer.student_optimizer_update_verified,
        "projector_backward_verified": trainer.projector_backward_verified,
        "first_updated_student_parameter": trainer.first_updated_student_parameter,
        "first_student_update_max_abs_delta": trainer.first_student_update_max_abs_delta,
        "params_before": params_before,
        "params_after": params_after,
        "params_preserved": params_preserved,
        "conv_structure_preserved": structure_preserved,
        "elapsed_seconds": elapsed_seconds,
        "elapsed_minutes": elapsed_seconds / 60.0,
        "elapsed_hours": elapsed_seconds / 3600.0,
        **results_summary,
        "checks": checks,
        "kd_loss_monotonicity_is_not_a_failure_condition": True,
    }

    report_path = trainer.save_dir / "kd_summary.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n" + "=" * 80)
    print("FORMAL KD SUMMARY")
    print("=" * 80)
    for name, passed in checks.items():
        print(f"{'PASS' if passed else 'FAIL':<4}  {name}")

    best = results_summary["best_epoch_by_mask_map50_95"]
    last = results_summary["last_epoch"]
    print("\nBest epoch selected by Mask mAP50-95:")
    print(f"  Epoch:         {best['epoch']}")
    print(f"  Box mAP50:     {best['box_map50']:.5f}")
    print(f"  Box mAP50-95:  {best['box_map50_95']:.5f}")
    print(f"  Mask mAP50:    {best['mask_map50']:.5f}")
    print(f"  Mask mAP50-95: {best['mask_map50_95']:.5f}")

    print("\nLast epoch:")
    print(f"  Epoch:         {last['epoch']}")
    print(f"  Box mAP50:     {last['box_map50']:.5f}")
    print(f"  Box mAP50-95:  {last['box_map50_95']:.5f}")
    print(f"  Mask mAP50:    {last['mask_map50']:.5f}")
    print(f"  Mask mAP50-95: {last['mask_map50_95']:.5f}")

    print("\nIndependent metric peaks:")
    for metric, peak in results_summary["independent_peaks"].items():
        print(f"  {metric}: {peak['value']:.5f} at epoch {peak['epoch']}")

    print("\nDistillation loss endpoints (informational, not a monotonicity check):")
    print(f"  First epoch: {results_summary['distillation_loss_first']:.6f}")
    print(f"  Last epoch:  {results_summary['distillation_loss_last']:.6f}")
    print(f"\nElapsed: {elapsed_seconds / 60.0:.1f} min ({elapsed_seconds / 3600.0:.3f} h)")
    print(f"Summary: {report_path}")
    print(f"Checkpoint: {checkpoint_path}")

    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise RuntimeError("Formal KD training failed checks: " + ", ".join(failed))

    print("\nFormal 30-epoch KD training completed successfully.")


if __name__ == "__main__":
    main()
