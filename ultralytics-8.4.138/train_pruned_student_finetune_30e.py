import csv
import json
import math
import time
from pathlib import Path

import torch
import torch.nn as nn

from ultralytics.models.yolo.segment import SegmentationTrainer
from ultralytics.nn.tasks import SegmentationModel


# ============================================================
# Paths
# ============================================================

PRUNED_STUDENT = Path(
    "D:/CrackSegRuns/"
    "pruning_dryrun_10_v2/"
    "teacher_pruned_10_v2_raw.pth"
)

DATA = "D:/crack-seg/crack-seg-local.yaml"

PROJECT = "D:/CrackSegRuns"
NAME = "pruned_student_10_finetune_30e"


# ============================================================
# Training settings
# ============================================================

EPOCHS = 30

IMAGE_SIZE = 640
BATCH_SIZE = 16
DEVICE = 0

# Windows 下稳定优先
WORKERS = 0

OPTIMIZER = "AdamW"
LR0 = 0.001
WEIGHT_DECAY = 0.0005

FRACTION = 1.0
PLOTS = True


# ============================================================
# Structural guardrails
# ============================================================

# 当前保存的 pruned Student 已经 fuse
EXPECTED_RAW_PARAMS = 3_742_841

# 原 Teacher 参数量
TEACHER_PARAMS = 4_494_915


# ============================================================
# Utility functions
# ============================================================

def count_parameters(model):
    return sum(
        parameter.numel()
        for parameter in model.parameters()
    )


def count_trainable_tensors(model):
    return sum(
        parameter.requires_grad
        for parameter in model.parameters()
    )


def conv_channel_signature(model):
    """
    记录所有 Conv2d 的结构。

    Fine-tune 之后这些结构必须保持一致，
    因为 Fine-tune 只能修改参数值，
    不能把已经剪掉的通道重新长回来。
    """

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


# ============================================================
# Load structurally pruned Student
# ============================================================

def load_pruned_student():

    if not PRUNED_STUDENT.is_file():
        raise FileNotFoundError(
            f"Pruned Student not found:\n{PRUNED_STUDENT}"
        )

    student = torch.load(
        PRUNED_STUDENT,
        map_location="cpu",
        weights_only=False,
    )

    # --------------------------------------------------------
    # Confirm model type
    # --------------------------------------------------------

    if not isinstance(student, SegmentationModel):
        raise TypeError(
            "Expected complete SegmentationModel, got "
            f"{type(student).__module__}."
            f"{type(student).__name__}"
        )

    if getattr(student, "task", None) != "segment":
        raise RuntimeError(
            "Expected task='segment', got "
            f"{getattr(student, 'task', None)!r}"
        )

    # --------------------------------------------------------
    # Parameter count guardrail
    # --------------------------------------------------------

    params = count_parameters(student)

    print("=" * 80)
    print("Loading structurally pruned Student")
    print("=" * 80)

    print(f"Source: {PRUNED_STUDENT}")
    print(
        f"Type:   "
        f"{type(student).__module__}."
        f"{type(student).__name__}"
    )
    print(
        f"Params: {params:,} "
        f"({params / 1e6:.6f} M)"
    )

    if params != EXPECTED_RAW_PARAMS:
        raise RuntimeError(
            "\nUnexpected Student parameter count.\n"
            f"Expected: {EXPECTED_RAW_PARAMS:,}\n"
            f"Actual:   {params:,}\n\n"
            "Stop training to avoid accidentally using "
            "a rebuilt or different network."
        )

    if params >= TEACHER_PARAMS:
        raise RuntimeError(
            "Loaded model is not smaller than Teacher. "
            "The pruning structure may have been lost."
        )

    # --------------------------------------------------------
    # Check whether model is fused
    # --------------------------------------------------------

    batchnorm_count = sum(
        isinstance(module, nn.BatchNorm2d)
        for module in student.modules()
    )

    print(
        f"BatchNorm2d layers: {batchnorm_count}"
    )

    if batchnorm_count == 0:
        print(
            "NOTE: Student is already fused. "
            "This formal experiment fine-tunes the fused "
            "structurally pruned Student."
        )

    # --------------------------------------------------------
    # Segment head dimensionality guardrail
    # --------------------------------------------------------

    head = student.model[-1]

    proto_channels = (
        head.proto.cv3.conv.out_channels
    )

    coefficient_channels = [
        branch[-1].out_channels
        for branch in head.cv4
    ]

    print(
        f"Mask prototype channels:   "
        f"{proto_channels}"
    )

    print(
        f"Mask coefficient channels: "
        f"{coefficient_channels}"
    )

    if proto_channels != int(head.nm):
        raise RuntimeError(
            "Prototype channel count does not "
            "match head.nm."
        )

    if any(
        channels != int(head.nm)
        for channels in coefficient_channels
    ):
        raise RuntimeError(
            "Mask coefficient output dimension "
            "does not match head.nm."
        )

    # --------------------------------------------------------
    # Enable training
    # --------------------------------------------------------

    before_enable = count_trainable_tensors(
        student
    )

    print(
        "Trainable parameter tensors "
        f"before enable: {before_enable}"
    )

    student.requires_grad_(True)

    after_enable = count_trainable_tensors(
        student
    )

    print(
        "Trainable parameter tensors "
        f"after enable:  {after_enable}"
    )

    if after_enable == 0:
        raise RuntimeError(
            "No trainable parameters after "
            "requires_grad_(True)."
        )

    # Rebuild task loss for fine-tuning settings
    student.criterion = None

    student.train()

    return student


# ============================================================
# Trainer
# ============================================================

class PrunedStudentFinetuneTrainer(
    SegmentationTrainer
):
    """
    Fine-tune the already-pruned in-memory model.

    Important:
    The model must NOT be reconstructed from YAML.
    """

    def __init__(self, *args, **kwargs):

        super().__init__(
            *args,
            **kwargs
        )

        self.backward_verified = False
        self.optimizer_update_verified = False

        self.first_updated_parameter = None
        self.first_update_max_abs_delta = 0.0

    def setup_model(self):

        if not isinstance(
            self.model,
            SegmentationModel
        ):
            raise TypeError(
                "Trainer did not receive the "
                "in-memory pruned SegmentationModel. "
                "Refusing to rebuild the model from YAML."
            )

        print(
            "\nTrainer setup: using the "
            "in-memory structurally pruned Student."
        )

        print(
            "No YAML model reconstruction will occur."
        )

        return super().setup_model()

    def optimizer_step(self):

        tracked_name = None
        tracked_parameter = None

        # ----------------------------------------------------
        # Verify gradients actually exist
        # ----------------------------------------------------

        for name, parameter in (
            self.model.named_parameters()
        ):

            gradient = parameter.grad

            if (
                parameter.requires_grad
                and gradient is not None
                and torch.isfinite(gradient).all()
                and torch.count_nonzero(
                    gradient
                ).item() > 0
            ):
                self.backward_verified = True

                tracked_name = name
                tracked_parameter = parameter

                break

        # ----------------------------------------------------
        # Snapshot parameter before optimizer update
        # ----------------------------------------------------

        before = None

        if (
            tracked_parameter is not None
            and not self.optimizer_update_verified
        ):
            before = (
                tracked_parameter
                .detach()
                .clone()
            )

        # Actual Ultralytics optimizer step
        super().optimizer_step()

        # ----------------------------------------------------
        # Confirm parameter actually changed
        # ----------------------------------------------------

        if before is not None:

            delta = (
                tracked_parameter.detach()
                - before
            ).abs().max().item()

            self.first_update_max_abs_delta = (
                delta
            )

            self.first_updated_parameter = (
                tracked_name
            )

            if delta > 0.0:
                self.optimizer_update_verified = (
                    True
                )


# ============================================================
# Results parsing
# ============================================================

def read_results_csv(results_csv):

    with results_csv.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as file:

        rows = [
            {
                key.strip(): value.strip()
                for key, value in row.items()
            }
            for row in csv.DictReader(file)
        ]

    if not rows:
        raise RuntimeError(
            "results.csv contains no epochs."
        )

    print(
        f"\nresults.csv epochs recorded: "
        f"{len(rows)}"
    )

    # --------------------------------------------------------
    # Find metric column names robustly
    # --------------------------------------------------------

    columns = list(rows[0].keys())

    def find_column(
        contains_all,
        starts_with=None,
    ):
        for column in columns:

            if (
                starts_with is not None
                and not column.startswith(
                    starts_with
                )
            ):
                continue

            if all(
                token in column
                for token in contains_all
            ):
                return column

        return None

    box_map50 = find_column(
        ["mAP50(B)"],
        "metrics/",
    )

    box_map5095 = find_column(
        ["mAP50-95(B)"],
        "metrics/",
    )

    mask_map50 = find_column(
        ["mAP50(M)"],
        "metrics/",
    )

    mask_map5095 = find_column(
        ["mAP50-95(M)"],
        "metrics/",
    )

    required = {
        "Box mAP50": box_map50,
        "Box mAP50-95": box_map5095,
        "Mask mAP50": mask_map50,
        "Mask mAP50-95": mask_map5095,
    }

    missing = [
        name
        for name, key in required.items()
        if key is None
    ]

    if missing:
        raise RuntimeError(
            "Could not locate metric columns: "
            + ", ".join(missing)
        )

    # --------------------------------------------------------
    # Convert useful columns to float
    # --------------------------------------------------------

    parsed_rows = []

    for index, row in enumerate(
        rows,
        start=1,
    ):

        parsed = {
            "epoch": index,
            "box_map50": float(
                row[box_map50]
            ),
            "box_map50_95": float(
                row[box_map5095]
            ),
            "mask_map50": float(
                row[mask_map50]
            ),
            "mask_map50_95": float(
                row[mask_map5095]
            ),
        }

        # Validate metric values
        if not all(
            math.isfinite(value)
            for key, value in parsed.items()
            if key != "epoch"
        ):
            raise RuntimeError(
                f"Non-finite metric at epoch "
                f"{index}"
            )

        parsed_rows.append(parsed)

    # --------------------------------------------------------
    # Best epoch selected by Mask mAP50-95
    # --------------------------------------------------------

    best_row = max(
        parsed_rows,
        key=lambda item: item[
            "mask_map50_95"
        ],
    )

    last_row = parsed_rows[-1]

    # --------------------------------------------------------
    # Independent metric peaks
    # Useful for training log only
    # --------------------------------------------------------

    peak_box_map50 = max(
        parsed_rows,
        key=lambda item: item["box_map50"]
    )

    peak_box_map5095 = max(
        parsed_rows,
        key=lambda item: item[
            "box_map50_95"
        ]
    )

    peak_mask_map50 = max(
        parsed_rows,
        key=lambda item: item["mask_map50"]
    )

    peak_mask_map5095 = max(
        parsed_rows,
        key=lambda item: item[
            "mask_map50_95"
        ]
    )

    return {
        "epochs_recorded": len(rows),

        "best_epoch_by_mask_map50_95":
            best_row,

        "last_epoch":
            last_row,

        "independent_peaks": {
            "box_map50": peak_box_map50,
            "box_map50_95":
                peak_box_map5095,
            "mask_map50":
                peak_mask_map50,
            "mask_map50_95":
                peak_mask_map5095,
        },
    }


# ============================================================
# Checkpoint loading
# ============================================================

def load_checkpoint_model(
    checkpoint_path
):

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )

    model = (
        checkpoint.get("ema")
        or checkpoint.get("model")
    )

    if model is None:
        raise RuntimeError(
            "No EMA/model object found in "
            f"{checkpoint_path}"
        )

    return model.float()


# ============================================================
# Main
# ============================================================

def main():

    # --------------------------------------------------------
    # Load untouched raw pruned Student
    # --------------------------------------------------------

    student = load_pruned_student()

    params_before = count_parameters(
        student
    )

    signature_before = (
        conv_channel_signature(
            student
        )
    )

    # --------------------------------------------------------
    # Trainer overrides
    # --------------------------------------------------------

    overrides = {

        # Metadata only.
        # trainer.model will be replaced by the
        # in-memory pruned model before train().
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

        "pretrained": False,
        "resume": False,
        "freeze": None,

        "project": PROJECT,
        "name": NAME,

        # If directory already exists,
        # Ultralytics may reuse it.
        # For a formal experiment, make sure
        # the previous directory is not mixed
        # with an old run.
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

    trainer = (
        PrunedStudentFinetuneTrainer(
            overrides=overrides
        )
    )

    # Critical:
    # overwrite trainer.model with the
    # actual structurally-pruned nn.Module
    trainer.model = student

    print("\n" + "=" * 80)

    print(
        f"Starting formal {EPOCHS}-epoch "
        "Pruned Student fine-tuning"
    )

    print("=" * 80)

    start_time = time.time()

    trainer.train()

    elapsed_seconds = (
        time.time() - start_time
    )

    # --------------------------------------------------------
    # Basic training verification
    # --------------------------------------------------------

    if not trainer.backward_verified:
        raise RuntimeError(
            "Backward propagation was never "
            "verified."
        )

    if not (
        trainer.optimizer_update_verified
    ):
        raise RuntimeError(
            "Optimizer parameter update was "
            "never verified."
        )

    # --------------------------------------------------------
    # Parse results.csv
    # --------------------------------------------------------

    results_csv = (
        trainer.save_dir
        / "results.csv"
    )

    if not results_csv.is_file():
        raise FileNotFoundError(
            f"Training results not found:\n"
            f"{results_csv}"
        )

    results_summary = (
        read_results_csv(
            results_csv
        )
    )

    # --------------------------------------------------------
    # Use Ultralytics-selected best.pt
    # --------------------------------------------------------

    if trainer.best.is_file():
        final_checkpoint = (
            trainer.best
        )
    elif trainer.last.is_file():
        final_checkpoint = (
            trainer.last
        )
    else:
        raise FileNotFoundError(
            "Neither best.pt nor last.pt "
            "was produced."
        )

    trained_student = (
        load_checkpoint_model(
            final_checkpoint
        )
    )

    # --------------------------------------------------------
    # Verify pruning structure survived training
    # --------------------------------------------------------

    params_after = (
        count_parameters(
            trained_student
        )
    )

    signature_after = (
        conv_channel_signature(
            trained_student
        )
    )

    params_preserved = (
        params_after
        == params_before
    )

    structure_preserved = (
        signature_after
        == signature_before
    )

    if not params_preserved:
        raise RuntimeError(
            "Parameter count changed during "
            "fine-tuning."
        )

    if not structure_preserved:
        raise RuntimeError(
            "Conv2d channel structure changed "
            "during fine-tuning."
        )

    # --------------------------------------------------------
    # Create final report
    # --------------------------------------------------------

    report = {

        "experiment":
            "10% pruned Student "
            "30e fine-tune",

        "source_model":
            str(PRUNED_STUDENT),

        "output_dir":
            str(trainer.save_dir),

        "final_checkpoint":
            str(final_checkpoint),

        "epochs_requested":
            EPOCHS,

        "optimizer":
            OPTIMIZER,

        "lr0":
            LR0,

        "weight_decay":
            WEIGHT_DECAY,

        "imgsz":
            IMAGE_SIZE,

        "batch":
            BATCH_SIZE,

        "workers":
            WORKERS,

        "backward_verified":
            trainer.backward_verified,

        "optimizer_update_verified":
            trainer.optimizer_update_verified,

        "first_updated_parameter":
            trainer.first_updated_parameter,

        "first_update_max_abs_delta":
            trainer.first_update_max_abs_delta,

        "params_before":
            params_before,

        "params_after":
            params_after,

        "params_preserved":
            params_preserved,

        "conv_structure_preserved":
            structure_preserved,

        "elapsed_seconds":
            elapsed_seconds,

        "elapsed_hours":
            elapsed_seconds / 3600.0,

        **results_summary,
    }

    report_path = (
        trainer.save_dir
        / "finetune_summary.json"
    )

    report_path.write_text(
        json.dumps(
            report,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    # --------------------------------------------------------
    # Console summary
    # --------------------------------------------------------

    best = (
        results_summary[
            "best_epoch_by_mask_map50_95"
        ]
    )

    last = (
        results_summary[
            "last_epoch"
        ]
    )

    print("\n" + "=" * 80)
    print("FORMAL FINE-TUNE SUMMARY")
    print("=" * 80)

    print(
        f"Source Params: "
        f"{params_before:,}"
    )

    print(
        f"Final Params:  "
        f"{params_after:,}"
    )

    print(
        f"Structure preserved: "
        f"{structure_preserved}"
    )

    print()

    print(
        "Best epoch by Mask "
        "mAP50-95:"
    )

    print(
        f"  Epoch:          "
        f"{best['epoch']}"
    )

    print(
        f"  Box mAP50:      "
        f"{best['box_map50']:.5f}"
    )

    print(
        f"  Box mAP50-95:   "
        f"{best['box_map50_95']:.5f}"
    )

    print(
        f"  Mask mAP50:     "
        f"{best['mask_map50']:.5f}"
    )

    print(
        f"  Mask mAP50-95:  "
        f"{best['mask_map50_95']:.5f}"
    )

    print()

    print("Last epoch:")

    print(
        f"  Epoch:          "
        f"{last['epoch']}"
    )

    print(
        f"  Box mAP50:      "
        f"{last['box_map50']:.5f}"
    )

    print(
        f"  Box mAP50-95:   "
        f"{last['box_map50_95']:.5f}"
    )

    print(
        f"  Mask mAP50:     "
        f"{last['mask_map50']:.5f}"
    )

    print(
        f"  Mask mAP50-95:  "
        f"{last['mask_map50_95']:.5f}"
    )

    print()

    print(
        f"Elapsed: "
        f"{elapsed_seconds / 60:.1f} min"
    )

    print(
        f"Summary: "
        f"{report_path}"
    )

    print(
        f"Checkpoint: "
        f"{final_checkpoint}"
    )

    print(
        "\nFine-tuning completed successfully."
    )


if __name__ == "__main__":
    main()