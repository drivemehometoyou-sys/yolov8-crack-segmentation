import csv
import json
import math
from pathlib import Path

import torch
import torch.nn as nn

from ultralytics.models.yolo.segment import SegmentationTrainer
from ultralytics.nn.tasks import SegmentationModel


PRUNED_STUDENT = Path(
    "D:/CrackSegRuns/pruning_dryrun_10_v2/teacher_pruned_10_v2_raw.pth"
)
DATA = "D:/crack-seg/crack-seg-local.yaml"
PROJECT = "D:/CrackSegRuns"
NAME = "pruned_student_10_finetune_smoke"

EPOCHS = 3
IMAGE_SIZE = 640
BATCH_SIZE = 16
DEVICE = 0
WORKERS = 0
FRACTION = 1.0
PLOTS = True

OPTIMIZER = "AdamW"
LR0 = 0.001
WEIGHT_DECAY = 0.0005

# Structural guardrail. The saved .pth is already fused, so its exact parameter
# count is 3,742,841 rather than the pre-fusion Torch-Pruning count 3,747,983.
EXPECTED_RAW_PARAMS = 3_742_841
TEACHER_PARAMS = 4_494_915


def count_parameters(model):
    return sum(parameter.numel() for parameter in model.parameters())


def conv_channel_signature(model):
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


def count_trainable_tensors(model):
    return sum(parameter.requires_grad for parameter in model.parameters())


def load_pruned_student():
    if not PRUNED_STUDENT.is_file():
        raise FileNotFoundError(f"Pruned Student not found: {PRUNED_STUDENT}")

    student = torch.load(
        PRUNED_STUDENT,
        map_location="cpu",
        weights_only=False,
    )

    if not isinstance(student, SegmentationModel):
        raise TypeError(
            "Expected a complete ultralytics.nn.tasks.SegmentationModel, "
            f"got {type(student).__module__}.{type(student).__name__}."
        )

    if getattr(student, "task", None) != "segment":
        raise RuntimeError(
            f"Expected task='segment', got {getattr(student, 'task', None)!r}."
        )

    params = count_parameters(student)
    print("=" * 80)
    print("Loading structurally pruned Student")
    print("=" * 80)
    print(f"Source: {PRUNED_STUDENT}")
    print(f"Type:   {type(student).__module__}.{type(student).__name__}")
    print(f"Params: {params:,} ({params / 1e6:.6f} M)")

    if params != EXPECTED_RAW_PARAMS:
        raise RuntimeError(
            "Unexpected raw Student parameter count. "
            f"Expected {EXPECTED_RAW_PARAMS:,}, got {params:,}. "
            "Stop to avoid training a rebuilt or different model."
        )

    if params >= TEACHER_PARAMS:
        raise RuntimeError(
            "Loaded model is not smaller than the Teacher. "
            "The pruned structure may have been replaced or rebuilt."
        )

    batchnorm_count = sum(
        isinstance(module, nn.BatchNorm2d)
        for module in student.modules()
    )
    print(f"BatchNorm2d layers: {batchnorm_count}")
    if batchnorm_count == 0:
        print(
            "NOTE: this saved Student is already fused. The smoke test will "
            "verify training of the fused, structurally pruned model object."
        )

    head = student.model[-1]
    proto_channels = head.proto.cv3.conv.out_channels
    coefficient_channels = [branch[-1].out_channels for branch in head.cv4]
    print(f"Mask prototype channels:   {proto_channels}")
    print(f"Mask coefficient channels: {coefficient_channels}")
    if proto_channels != int(head.nm) or any(
        channels != int(head.nm)
        for channels in coefficient_channels
    ):
        raise RuntimeError(
            "Mask prototype/coefficient dimensions do not match head.nm."
        )

    before_enable = count_trainable_tensors(student)
    print(f"Trainable parameter tensors before enable: {before_enable}")
    student.requires_grad_(True)
    after_enable = count_trainable_tensors(student)
    print(f"Trainable parameter tensors after enable:  {after_enable}")

    if after_enable == 0:
        raise RuntimeError("No trainable parameters after requires_grad_(True).")

    # Build a fresh loss object against the fine-tuning hyperparameters.
    student.criterion = None
    student.train()
    return student


class PrunedStudentSmokeTrainer(SegmentationTrainer):
    """SegmentationTrainer with explicit backward/update verification."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.backward_verified = False
        self.optimizer_update_verified = False
        self.first_update_max_abs_delta = 0.0
        self.first_updated_parameter = None

    def setup_model(self):
        if not isinstance(self.model, SegmentationModel):
            raise TypeError(
                "Trainer did not receive the in-memory pruned "
                "SegmentationModel. Refusing to rebuild from YAML."
            )
        print(
            "Trainer setup: using the in-memory pruned Student directly; "
            "no YAML model reconstruction."
        )
        return super().setup_model()

    def optimizer_step(self):
        tracked_name = None
        tracked_parameter = None

        for name, parameter in self.model.named_parameters():
            gradient = parameter.grad
            if (
                parameter.requires_grad
                and gradient is not None
                and torch.isfinite(gradient).all()
                and torch.count_nonzero(gradient).item() > 0
            ):
                self.backward_verified = True
                tracked_name = name
                tracked_parameter = parameter
                break

        before = (
            tracked_parameter.detach().clone()
            if tracked_parameter is not None
            and not self.optimizer_update_verified
            else None
        )

        super().optimizer_step()

        if before is not None:
            delta = (tracked_parameter.detach() - before).abs().max().item()
            self.first_update_max_abs_delta = delta
            self.first_updated_parameter = tracked_name
            if delta > 0.0:
                self.optimizer_update_verified = True


def read_results_csv(results_csv):
    with results_csv.open("r", encoding="utf-8-sig", newline="") as file:
        rows = [
            {key.strip(): value.strip() for key, value in row.items()}
            for row in csv.DictReader(file)
        ]

    if len(rows) != EPOCHS:
        raise RuntimeError(
            f"Expected {EPOCHS} results.csv rows, found {len(rows)}."
        )

    loss_keys = [
        key
        for key in rows[0]
        if key.startswith("train/") and key.endswith("_loss")
    ]
    metric_keys = [key for key in rows[0] if key.startswith("metrics/")]

    if not loss_keys:
        raise RuntimeError("No training-loss columns found in results.csv.")
    if not metric_keys:
        raise RuntimeError("No validation metric columns found in results.csv.")

    losses = []
    print("\nThree-epoch loss trace")
    print("-" * 80)
    for epoch_index, row in enumerate(rows, start=1):
        epoch_losses = {key: float(row[key]) for key in loss_keys}
        if not all(math.isfinite(value) for value in epoch_losses.values()):
            raise RuntimeError(f"Non-finite loss detected at epoch {epoch_index}.")
        losses.append(epoch_losses)
        formatted = ", ".join(
            f"{key.split('/', 1)[1]}={value:.6f}"
            for key, value in epoch_losses.items()
        )
        print(f"Epoch {epoch_index}: {formatted}")

    first_total = sum(losses[0].values())
    last_total = sum(losses[-1].values())
    loss_trend_evaluable = len(losses) >= 2
    loss_decreased = last_total < first_total if loss_trend_evaluable else None

    final_metrics = {key: float(rows[-1][key]) for key in metric_keys}
    validation_finite = all(
        math.isfinite(value)
        for value in final_metrics.values()
    )
    if not validation_finite:
        raise RuntimeError("Non-finite validation metric detected.")

    return {
        "loss_keys": loss_keys,
        "first_total_train_loss": first_total,
        "last_total_train_loss": last_total,
        "loss_trend_evaluable": loss_trend_evaluable,
        "loss_decreased": loss_decreased,
        "final_validation_metrics": final_metrics,
        "validation_finite": validation_finite,
    }


def load_checkpoint_model(checkpoint_path):
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )
    model = checkpoint.get("ema") or checkpoint.get("model")
    if model is None:
        raise RuntimeError(f"No EMA/model object found in {checkpoint_path}.")
    return model.float()


def main():
    student = load_pruned_student()
    params_before = count_parameters(student)
    signature_before = conv_channel_signature(student)

    overrides = {
        # This path is metadata only. trainer.model is replaced with the loaded
        # nn.Module before train(), and setup_model() verifies that fact.
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
        "exist_ok": True,
        "val": True,
        "save": True,
        "plots": PLOTS,
        "cache": False,
        "fraction": FRACTION,
        "seed": 0,
        "deterministic": True,
        "compile": False,
    }

    trainer = PrunedStudentSmokeTrainer(overrides=overrides)
    trainer.model = student

    print("\n" + "=" * 80)
    print(f"Starting {EPOCHS}-epoch Pruned Student fine-tune smoke test")
    print("=" * 80)
    trainer.train()

    results_csv = trainer.save_dir / "results.csv"
    if not results_csv.is_file():
        raise FileNotFoundError(f"Training results not found: {results_csv}")
    results_check = read_results_csv(results_csv)

    final_checkpoint = trainer.best if trainer.best.is_file() else trainer.last
    if not final_checkpoint.is_file():
        raise FileNotFoundError("Neither best.pt nor last.pt was produced.")
    trained_student = load_checkpoint_model(final_checkpoint)

    params_after = count_parameters(trained_student)
    signature_after = conv_channel_signature(trained_student)
    params_preserved = params_after == params_before
    structure_preserved = signature_after == signature_before

    report = {
        "source_model": str(PRUNED_STUDENT),
        "output_dir": str(trainer.save_dir),
        "final_checkpoint": str(final_checkpoint),
        "epochs": EPOCHS,
        "optimizer": OPTIMIZER,
        "lr0": LR0,
        "weight_decay": WEIGHT_DECAY,
        "backward_verified": trainer.backward_verified,
        "optimizer_update_verified": trainer.optimizer_update_verified,
        "first_updated_parameter": trainer.first_updated_parameter,
        "first_update_max_abs_delta": trainer.first_update_max_abs_delta,
        "losses_finite": True,
        **results_check,
        "params_before": params_before,
        "params_after": params_after,
        "params_preserved": params_preserved,
        "conv_structure_preserved": structure_preserved,
    }

    report_path = trainer.save_dir / "smoke_verification.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    checks = {
        "backward propagation": trainer.backward_verified,
        "optimizer parameter update": trainer.optimizer_update_verified,
        "finite losses": True,
        "overall loss decreased": (
            results_check["loss_decreased"]
            if results_check["loss_trend_evaluable"]
            else True
        ),
        "validation completed": results_check["validation_finite"],
        "parameter count preserved": params_preserved,
        "Conv2d structure preserved": structure_preserved,
    }

    print("\n" + "=" * 80)
    print("SMOKE TEST VERIFICATION")
    print("=" * 80)
    for check_name, passed in checks.items():
        print(f"{'PASS' if passed else 'FAIL':<4}  {check_name}")
    print(f"Params before: {params_before:,}")
    print(f"Params after:  {params_after:,}")
    print(f"Report:        {report_path}")

    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise RuntimeError("Smoke test failed checks: " + ", ".join(failed))

    print("\nAll smoke-test checks passed. The pruned Student is trainable.")


if __name__ == "__main__":
    main()
