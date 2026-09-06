import csv
import json
import math
from pathlib import Path

import torch
import torch.nn as nn

from ultralytics.models.yolo.segment import SegmentationTrainer
from ultralytics.nn.distill_model import DistillationModel
from ultralytics.nn.tasks import SegmentationModel
from ultralytics.utils.torch_utils import unwrap_model


# ============================================================
# Paths
# ============================================================

# Teacher：未剪枝的 SPDConv + SPPF-MSA 最优模型
TEACHER = Path(
    "D:/CrackSegRuns/"
    "yolov8n_seg_spdconv_sppfmsa_100e/"
    "weights/"
    "best.pt"
)

# Student：原始 10% 结构化剪枝模型
# 注意：必须从 raw pruned Student 开始，而不是 Fine-tune 的 best.pt
PRUNED_STUDENT = Path(
    "D:/CrackSegRuns/"
    "pruning_dryrun_10_v2/"
    "teacher_pruned_10_v2_raw.pth"
)

DATA = "D:/crack-seg/crack-seg-local.yaml"

PROJECT = "D:/CrackSegRuns"
NAME = "pruned_student_10_kd_smoke"


# ============================================================
# Smoke-test settings
# ============================================================

EPOCHS = 3

IMAGE_SIZE = 640
BATCH_SIZE = 16
DEVICE = 0

# Windows 下稳定优先
WORKERS = 0

OPTIMIZER = "AdamW"
LR0 = 0.001
WEIGHT_DECAY = 0.0005

# Ultralytics DistillationModel 中真正使用的
# feature distillation loss 权重
DISTILL_WEIGHT = 6.0

FRACTION = 1.0
PLOTS = True


# ============================================================
# Structural guardrails
# ============================================================

# 当前 raw pruned Student 已经过 fuse
EXPECTED_STUDENT_PARAMS = 3_742_841

# 原 Teacher 未 fuse 复杂度统计时参数量
# 这里只用来确认 Student 确实更小
TEACHER_REFERENCE_PARAMS = 4_494_915


# ============================================================
# Utilities
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
    保存所有 Conv2d 的输入/输出通道结构。

    KD 训练完成以后 Student 的结构必须保持一致。
    Projector 是训练辅助模块，不属于最终 Student。
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
# Load raw structurally-pruned Student
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
    # Confirm correct model type
    # --------------------------------------------------------

    if not isinstance(student, SegmentationModel):
        raise TypeError(
            "Expected a complete SegmentationModel, got "
            f"{type(student).__module__}."
            f"{type(student).__name__}"
        )

    if getattr(student, "task", None) != "segment":
        raise RuntimeError(
            "Expected task='segment', got "
            f"{getattr(student, 'task', None)!r}"
        )

    params = count_parameters(student)

    print("=" * 80)
    print("Loading raw structurally-pruned Student")
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

    # --------------------------------------------------------
    # Guard against accidental YAML reconstruction
    # --------------------------------------------------------

    if params != EXPECTED_STUDENT_PARAMS:
        raise RuntimeError(
            "\nUnexpected Student parameter count.\n"
            f"Expected: {EXPECTED_STUDENT_PARAMS:,}\n"
            f"Actual:   {params:,}\n\n"
            "Stop training to avoid accidentally using "
            "a rebuilt or different network."
        )

    if params >= TEACHER_REFERENCE_PARAMS:
        raise RuntimeError(
            "Loaded Student is not smaller than Teacher. "
            "The pruned structure may have been lost."
        )

    # --------------------------------------------------------
    # Check fused state
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
            "KD smoke test will train the fused, "
            "structurally-pruned Student."
        )

    # --------------------------------------------------------
    # Verify segmentation head dimensions
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
            "Mask prototype output dimension "
            "does not match head.nm."
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
    # Enable Student training
    # --------------------------------------------------------

    before_enable = count_trainable_tensors(
        student
    )

    print(
        "Trainable Student parameter tensors "
        f"before enable: {before_enable}"
    )

    student.requires_grad_(True)

    after_enable = count_trainable_tensors(
        student
    )

    print(
        "Trainable Student parameter tensors "
        f"after enable:  {after_enable}"
    )

    if after_enable == 0:
        raise RuntimeError(
            "No trainable Student parameters after "
            "requires_grad_(True)."
        )

    # Rebuild segmentation criterion against
    # the current training hyperparameters
    student.criterion = None

    student.train()

    return student


# ============================================================
# KD Trainer
# ============================================================

class PrunedStudentKDSmokeTrainer(
    SegmentationTrainer
):
    """
    SegmentationTrainer used to verify:

    1. raw pruned Student is used directly;
    2. native Ultralytics DistillationModel is created;
    3. Teacher remains frozen;
    4. Student receives gradients;
    5. Student parameters actually update;
    6. KD projector receives gradients.
    """

    def __init__(self, *args, **kwargs):

        super().__init__(
            *args,
            **kwargs
        )

        self.distillation_wrapper_verified = False
        self.teacher_frozen_verified = False

        self.student_backward_verified = False
        self.student_optimizer_update_verified = False

        self.projector_backward_verified = False

        self.first_updated_student_parameter = None
        self.first_student_update_max_abs_delta = 0.0

    def setup_model(self):

        # ----------------------------------------------------
        # Before parent setup:
        # Trainer must still contain our in-memory
        # structurally-pruned Student.
        # ----------------------------------------------------

        if not isinstance(
            self.model,
            SegmentationModel
        ):
            raise TypeError(
                "Trainer did not receive the in-memory "
                "pruned SegmentationModel before setup. "
                "Refusing to rebuild Student from YAML."
            )

        student_params_before_wrap = (
            count_parameters(
                self.model
            )
        )

        if (
            student_params_before_wrap
            != EXPECTED_STUDENT_PARAMS
        ):
            raise RuntimeError(
                "Student parameter count changed "
                "before DistillationModel wrapping."
            )

        print(
            "\nTrainer setup:"
        )

        print(
            "  Using in-memory raw pruned Student."
        )

        print(
            "  Native Ultralytics distillation "
            "will be enabled later inside _setup_train()."
        )

        # For an in-memory nn.Module the parent method intentionally
        # returns without rebuilding the model. The native KD wrapper
        # is created later by BaseTrainer._setup_train().
        return super().setup_model()

    def _setup_train(self):

        # Let native Ultralytics complete the full setup sequence:
        # - preserve the in-memory pruned Student
        # - move it to the selected device
        # - attach the current training args
        # - validate imgsz
        # - construct DistillationModel
        # - build optimizer, validator, and EMA
        super()._setup_train()

        kd_model = unwrap_model(self.model)

        # ----------------------------------------------------
        # Verify native KD wrapper after the parent _setup_train
        # ----------------------------------------------------

        if not isinstance(
            kd_model,
            DistillationModel
        ):
            raise RuntimeError(
                "\nNative DistillationModel was NOT created "
                "after SegmentationTrainer._setup_train().\n"
                f"distill_model={self.args.distill_model!r}"
            )

        self.distillation_wrapper_verified = True

        print(
            "\nDistillationModel successfully created."
        )

        # ----------------------------------------------------
        # Verify the wrapper still contains our pruned Student
        # ----------------------------------------------------

        wrapped_student_params = (
            count_parameters(
                kd_model.student_model
            )
        )

        print(
            "Wrapped Student params: "
            f"{wrapped_student_params:,}"
        )

        if (
            wrapped_student_params
            != EXPECTED_STUDENT_PARAMS
        ):
            raise RuntimeError(
                "The Student inside DistillationModel "
                "is not the expected structurally "
                "pruned Student."
            )

        # ----------------------------------------------------
        # Verify Teacher exists and is frozen
        # ----------------------------------------------------

        teacher = kd_model.teacher_model

        if teacher is None:
            raise RuntimeError(
                "DistillationModel contains no Teacher."
            )

        teacher_params = (
            count_parameters(
                teacher
            )
        )

        teacher_trainable = (
            count_trainable_tensors(
                teacher
            )
        )

        print(
            f"Teacher params: "
            f"{teacher_params:,}"
        )

        print(
            "Teacher trainable parameter tensors: "
            f"{teacher_trainable}"
        )

        self.teacher_frozen_verified = (
            teacher_trainable == 0
        )

        if not self.teacher_frozen_verified:
            raise RuntimeError(
                "Teacher is not fully frozen."
            )

        # ----------------------------------------------------
        # Show automatically-selected distillation layers
        # ----------------------------------------------------

        print(
            "Distillation feature layer indices: "
            f"{kd_model.feats_idx}"
        )

        # ----------------------------------------------------
        # Show Student -> Teacher channel projection
        # ----------------------------------------------------

        print(
            "KD projectors:"
        )

        for i, projector in enumerate(
            kd_model.projector
        ):
            first_conv = projector[0]
            second_conv = projector[2]

            print(
                f"  Level {i}: "
                f"{first_conv.in_channels} "
                f"-> {first_conv.out_channels} "
                f"-> {second_conv.out_channels}"
            )

        print(
            f"Distillation weight dis = "
            f"{kd_model.dis}"
        )

    def optimizer_step(self):

        tracked_name = None
        tracked_parameter = None
        kd_model = unwrap_model(self.model)

        # ----------------------------------------------------
        # Verify Student gradient
        # ----------------------------------------------------

        if isinstance(
            kd_model,
            DistillationModel
        ):

            for name, parameter in (
                kd_model.student_model
                .named_parameters()
            ):

                gradient = parameter.grad

                if (
                    parameter.requires_grad
                    and gradient is not None
                    and torch.isfinite(
                        gradient
                    ).all()
                    and torch.count_nonzero(
                        gradient
                    ).item() > 0
                ):
                    self.student_backward_verified = (
                        True
                    )

                    tracked_name = name
                    tracked_parameter = parameter

                    break

            # ------------------------------------------------
            # Verify KD projector gradient
            # ------------------------------------------------

            for parameter in (
                kd_model.projector.parameters()
            ):

                gradient = parameter.grad

                if (
                    parameter.requires_grad
                    and gradient is not None
                    and torch.isfinite(
                        gradient
                    ).all()
                    and torch.count_nonzero(
                        gradient
                    ).item() > 0
                ):
                    self.projector_backward_verified = (
                        True
                    )
                    break

        # ----------------------------------------------------
        # Snapshot one Student parameter
        # ----------------------------------------------------

        before = None

        if (
            tracked_parameter is not None
            and not
            self.student_optimizer_update_verified
        ):
            before = (
                tracked_parameter
                .detach()
                .clone()
            )

        # Normal Ultralytics optimizer step
        super().optimizer_step()

        # ----------------------------------------------------
        # Verify Student parameter really changed
        # ----------------------------------------------------

        if before is not None:

            delta = (
                tracked_parameter.detach()
                - before
            ).abs().max().item()

            self.first_student_update_max_abs_delta = (
                delta
            )

            self.first_updated_student_parameter = (
                tracked_name
            )

            if delta > 0.0:
                self.student_optimizer_update_verified = (
                    True
                )


# ============================================================
# Read results.csv
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
    # Find KD loss column
    # --------------------------------------------------------

    columns = list(
        rows[0].keys()
    )

    print(
        "\nTraining result columns:"
    )

    for column in columns:
        print(
            f"  {column}"
        )

    dis_loss_key = None

    for column in columns:
        if "dis_loss" in column:
            dis_loss_key = column
            break

    if dis_loss_key is None:
        raise RuntimeError(
            "\nNo dis_loss column was found "
            "in results.csv.\n"
            "Native knowledge distillation may "
            "not actually be active."
        )

    print(
        f"\nDetected KD loss column: "
        f"{dis_loss_key}"
    )

    # --------------------------------------------------------
    # Validate loss values
    # --------------------------------------------------------

    loss_keys = [
        key
        for key in columns
        if (
            key.startswith("train/")
            and key.endswith("_loss")
        )
    ]

    if not loss_keys:
        raise RuntimeError(
            "No training loss columns found."
        )

    kd_losses = []

    print(
        "\nKD smoke loss trace"
    )

    print(
        "-" * 80
    )

    for epoch_index, row in enumerate(
        rows,
        start=1,
    ):

        epoch_losses = {}

        for key in loss_keys:

            value = float(
                row[key]
            )

            if not math.isfinite(
                value
            ):
                raise RuntimeError(
                    "Non-finite training loss "
                    f"at epoch {epoch_index}: "
                    f"{key}={value}"
                )

            epoch_losses[key] = value

        kd_value = float(
            row[dis_loss_key]
        )

        if not math.isfinite(
            kd_value
        ):
            raise RuntimeError(
                "Non-finite KD loss at "
                f"epoch {epoch_index}."
            )

        kd_losses.append(
            kd_value
        )

        formatted = ", ".join(
            f"{key.split('/', 1)[-1]}"
            f"={value:.6f}"
            for key, value
            in epoch_losses.items()
        )

        print(
            f"Epoch {epoch_index}: "
            f"{formatted}"
        )

    kd_loss_finite = all(
        math.isfinite(value)
        for value in kd_losses
    )

    kd_loss_nonzero = any(
        abs(value) > 0.0
        for value in kd_losses
    )

    # --------------------------------------------------------
    # Validation metrics
    # --------------------------------------------------------

    metric_keys = [
        key
        for key in columns
        if key.startswith(
            "metrics/"
        )
    ]

    if not metric_keys:
        raise RuntimeError(
            "No validation metrics found."
        )

    final_metrics = {
        key: float(
            rows[-1][key]
        )
        for key in metric_keys
    }

    validation_finite = all(
        math.isfinite(value)
        for value
        in final_metrics.values()
    )

    if not validation_finite:
        raise RuntimeError(
            "Non-finite validation metric "
            "detected."
        )

    return {
        "epochs_recorded":
            len(rows),

        "distillation_loss_column":
            dis_loss_key,

        "distillation_losses":
            kd_losses,

        "distillation_loss_finite":
            kd_loss_finite,

        "distillation_loss_nonzero":
            kd_loss_nonzero,

        "final_validation_metrics":
            final_metrics,

        "validation_finite":
            validation_finite,
    }


# ============================================================
# Load Student from saved checkpoint
# ============================================================

def load_student_from_checkpoint(
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

    model = model.float()

    # Some Ultralytics checkpoints may still
    # contain DistillationModel.
    # For final deployment/evaluation we only
    # care about the Student.
    if isinstance(
        model,
        DistillationModel
    ):
        model = (
            model.student_model
        )

    if not isinstance(
        model,
        SegmentationModel
    ):
        raise TypeError(
            "Saved checkpoint does not contain "
            "a SegmentationModel Student."
        )

    return model


# ============================================================
# Main
# ============================================================

def main():

    # --------------------------------------------------------
    # Check Teacher path
    # --------------------------------------------------------

    if not TEACHER.is_file():
        raise FileNotFoundError(
            f"Teacher not found:\n{TEACHER}"
        )

    print("=" * 80)
    print("Teacher")
    print("=" * 80)

    print(
        f"Teacher checkpoint: "
        f"{TEACHER}"
    )

    # --------------------------------------------------------
    # Load untouched raw Student
    # --------------------------------------------------------

    student = (
        load_pruned_student()
    )

    params_before = (
        count_parameters(
            student
        )
    )

    structure_before = (
        conv_channel_signature(
            student
        )
    )

    # --------------------------------------------------------
    # Trainer arguments
    # --------------------------------------------------------

    overrides = {

        # Metadata path only.
        # trainer.model is explicitly replaced
        # by our in-memory pruned Student.
        "model":
            str(PRUNED_STUDENT),

        "data":
            DATA,

        "task":
            "segment",

        "epochs":
            EPOCHS,

        "imgsz":
            IMAGE_SIZE,

        "batch":
            BATCH_SIZE,

        "device":
            DEVICE,

        "workers":
            WORKERS,

        "optimizer":
            OPTIMIZER,

        "lr0":
            LR0,

        "weight_decay":
            WEIGHT_DECAY,

        # ----------------------------------------------------
        # Native Ultralytics knowledge distillation
        # ----------------------------------------------------

        "distill_model":
            str(TEACHER),

        "dis":
            DISTILL_WEIGHT,

        # ----------------------------------------------------

        "pretrained":
            False,

        "resume":
            False,

        "freeze":
            None,

        "project":
            PROJECT,

        "name":
            NAME,

        "exist_ok":
            True,

        "val":
            True,

        "save":
            True,

        "plots":
            PLOTS,

        "amp":
            True,

        "cache":
            False,

        "fraction":
            FRACTION,

        "seed":
            0,

        "deterministic":
            True,

        "compile":
            False,
    }

    trainer = (
        PrunedStudentKDSmokeTrainer(
            overrides=overrides
        )
    )

    # Critical:
    # feed actual structurally-pruned
    # nn.Module directly into trainer.
    trainer.model = student

    print(
        "\n" + "=" * 80
    )

    print(
        "Starting 3-epoch native KD smoke test"
    )

    print(
        "=" * 80
    )

    print(
        f"Teacher: "
        f"{TEACHER}"
    )

    print(
        f"Student: "
        f"{PRUNED_STUDENT}"
    )

    print(
        f"KD weight dis: "
        f"{DISTILL_WEIGHT}"
    )

    trainer.train()

    # --------------------------------------------------------
    # Read results.csv
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

    results_check = (
        read_results_csv(
            results_csv
        )
    )

    # --------------------------------------------------------
    # Locate checkpoint
    # --------------------------------------------------------

    if trainer.best.is_file():
        checkpoint_path = (
            trainer.best
        )

    elif trainer.last.is_file():
        checkpoint_path = (
            trainer.last
        )

    else:
        raise FileNotFoundError(
            "Neither best.pt nor last.pt "
            "was produced."
        )

    # --------------------------------------------------------
    # Extract final Student
    # --------------------------------------------------------

    trained_student = (
        load_student_from_checkpoint(
            checkpoint_path
        )
    )

    params_after = (
        count_parameters(
            trained_student
        )
    )

    structure_after = (
        conv_channel_signature(
            trained_student
        )
    )

    params_preserved = (
        params_after
        == params_before
    )

    structure_preserved = (
        structure_after
        == structure_before
    )

    # --------------------------------------------------------
    # Final checks
    # --------------------------------------------------------

    checks = {

        "native DistillationModel wrapper":
            trainer.distillation_wrapper_verified,

        "Teacher frozen":
            trainer.teacher_frozen_verified,

        "Student backward propagation":
            trainer.student_backward_verified,

        "Student optimizer parameter update":
            trainer.student_optimizer_update_verified,

        "KD projector backward propagation":
            trainer.projector_backward_verified,

        "finite KD loss":
            results_check[
                "distillation_loss_finite"
            ],

        "non-zero KD loss":
            results_check[
                "distillation_loss_nonzero"
            ],

        "validation completed":
            results_check[
                "validation_finite"
            ],

        "Student parameter count preserved":
            params_preserved,

        "Student Conv2d structure preserved":
            structure_preserved,
    }

    # --------------------------------------------------------
    # JSON report
    # --------------------------------------------------------

    report = {

        "experiment":
            "10% pruned Student "
            "native KD smoke test",

        "teacher":
            str(TEACHER),

        "student_source":
            str(PRUNED_STUDENT),

        "output_dir":
            str(trainer.save_dir),

        "checkpoint":
            str(checkpoint_path),

        "epochs":
            EPOCHS,

        "optimizer":
            OPTIMIZER,

        "lr0":
            LR0,

        "weight_decay":
            WEIGHT_DECAY,

        "distillation_weight_dis":
            DISTILL_WEIGHT,

        "distillation_wrapper_verified":
            trainer.distillation_wrapper_verified,

        "teacher_frozen_verified":
            trainer.teacher_frozen_verified,

        "student_backward_verified":
            trainer.student_backward_verified,

        "student_optimizer_update_verified":
            trainer.student_optimizer_update_verified,

        "projector_backward_verified":
            trainer.projector_backward_verified,

        "first_updated_student_parameter":
            trainer.first_updated_student_parameter,

        "first_student_update_max_abs_delta":
            trainer.first_student_update_max_abs_delta,

        "params_before":
            params_before,

        "params_after":
            params_after,

        "params_preserved":
            params_preserved,

        "conv_structure_preserved":
            structure_preserved,

        **results_check,

        "checks":
            checks,
    }

    report_path = (
        trainer.save_dir
        / "kd_smoke_verification.json"
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
    # Print summary
    # --------------------------------------------------------

    print(
        "\n" + "=" * 80
    )

    print(
        "KD SMOKE TEST VERIFICATION"
    )

    print(
        "=" * 80
    )

    for name, passed in (
        checks.items()
    ):
        print(
            f"{'PASS' if passed else 'FAIL':<4}  "
            f"{name}"
        )

    print()

    print(
        f"Student params before KD: "
        f"{params_before:,}"
    )

    print(
        f"Student params after KD:  "
        f"{params_after:,}"
    )

    print(
        f"First updated Student parameter: "
        f"{trainer.first_updated_student_parameter}"
    )

    print(
        "First Student max abs update: "
        f"{trainer.first_student_update_max_abs_delta}"
    )

    print(
        f"Report: "
        f"{report_path}"
    )

    print(
        f"Checkpoint: "
        f"{checkpoint_path}"
    )

    failed = [
        name
        for name, passed
        in checks.items()
        if not passed
    ]

    if failed:
        raise RuntimeError(
            "KD smoke test failed checks: "
            + ", ".join(failed)
        )

    print()

    print(
        "All KD smoke-test checks passed."
    )

    print(
        "Native Ultralytics knowledge "
        "distillation is compatible with "
        "the structurally-pruned Student."
    )


if __name__ == "__main__":
    main()
