from pathlib import Path

import torch
import torch.nn as nn
import torch_pruning as tp

from ultralytics import YOLO


TEACHER = (
    "D:/CrackSegRuns/yolov8n_seg_spdconv_sppfmsa_100e/weights/best.pt"
)
DATA = "D:/crack-seg/crack-seg-local.yaml"
OUTPUT_DIR = Path("D:/CrackSegRuns/pruning_dryrun_10_v2")
SAVE_PATH = OUTPUT_DIR / "teacher_pruned_10_v2_raw.pth"

PRUNING_RATIO = 0.10
IMAGE_SIZE = 640
BATCH_SIZE = 16


class ValidatedMagnitudePruner(tp.pruner.MagnitudePruner):
    """
    Skip dependency groups whose mapped indices exceed a target channel axis.

    Torch-Pruning 1.6.0 can produce invalid index mappings for some nested C2f
    split/concat paths. Calling MagnitudeImportance on such a group otherwise
    raises IndexError (or a CUDA device-side assertion) before valid groups are
    pruned. These groups are not added to ignored_layers; they are rejected only
    when TP's own dependency mapping is internally inconsistent.
    """

    def __init__(self, *args, **kwargs):
        self.skipped_invalid_groups = []
        super().__init__(*args, **kwargs)

    def estimate_importance(self, group):
        invalid_items = []
        for dependency, indices in group:
            if not indices:
                continue

            if self.DG.is_out_channel_pruning_fn(dependency.handler):
                channel_count = self.DG.get_out_channels(dependency.target.module)
                axis = "out"
            elif self.DG.is_in_channel_pruning_fn(dependency.handler):
                channel_count = self.DG.get_in_channels(dependency.target.module)
                axis = "in"
            else:
                continue

            max_index = max(indices)
            if channel_count is not None and max_index >= channel_count:
                invalid_items.append((dependency.target.module, axis, max_index, channel_count))

        if invalid_items:
            root_module = group[0][0].target.module
            self.skipped_invalid_groups.append((root_module, invalid_items))
            return None

        return super().estimate_importance(group)


def qualified_module_names(model):
    """Map module objects to stable, human-readable names."""
    return {module: name for name, module in model.named_modules()}


def describe_segment_head(head):
    """Print the actual cv2/cv3/cv4 container and branch structure."""
    print("\nSegment head structure")
    print("-" * 80)
    print(f"Head type: {type(head).__module__}.{type(head).__name__}")

    for attr in ("nc", "reg_max", "nm", "npr", "nl", "end2end"):
        print(f"{attr}: {getattr(head, attr, None)}")

    for attr in ("cv2", "cv3", "cv4"):
        container = getattr(head, attr, None)
        if container is None:
            raise RuntimeError(f"Segment head has no {attr} container.")

        print(f"\n{attr}: {type(container).__module__}.{type(container).__name__}")
        for branch_index, branch in enumerate(container):
            children = list(branch.children())
            child_types = ", ".join(
                f"{i}:{type(child).__module__}.{type(child).__name__}"
                for i, child in enumerate(children)
            )
            print(
                f"  {attr}[{branch_index}]: "
                f"{type(branch).__module__}.{type(branch).__name__} -> {child_types}"
            )


def final_output_conv(branch, branch_name):
    """Return one branch's final prediction Conv2d, rejecting ambiguity."""
    children = list(branch.children())
    if not children:
        raise RuntimeError(f"{branch_name} has no child layers.")

    final_layer = children[-1]
    if isinstance(final_layer, nn.Conv2d):
        return final_layer

    wrapped_conv = getattr(final_layer, "conv", None)
    if isinstance(wrapped_conv, nn.Conv2d):
        return wrapped_conv

    raise TypeError(
        f"{branch_name}'s final layer is "
        f"{type(final_layer).__module__}.{type(final_layer).__name__}, "
        "not an nn.Conv2d or a wrapper containing .conv."
    )


def collect_protected_output_convs(model):
    """
    Protect only fixed-semantic prediction outputs in the Segment head.

    For this Ultralytics 8.4.138 checkpoint these are the final Conv2d layers
    of every cv2 (box distribution), cv3 (class), and cv4 (mask coefficient)
    branch, plus proto.cv3.conv (mask prototypes). Both the mask coefficient
    outputs and prototype outputs must remain equal to head.nm.

    Intermediate head convolutions and all backbone/neck modules stay eligible
    for dependency-aware structured pruning.
    """
    head = model.model[-1]
    describe_segment_head(head)

    if not all(hasattr(head, attr) for attr in ("cv2", "cv3", "cv4")):
        raise TypeError(
            "The final model layer is not the expected Segment head with "
            "cv2/cv3/cv4 branches."
        )

    expected_outputs = {
        "cv2": 4 * int(head.reg_max),
        "cv3": int(head.nc),
        "cv4": int(head.nm),
    }

    protected = []
    for attr in ("cv2", "cv3", "cv4"):
        branches = getattr(head, attr)
        if len(branches) != int(head.nl):
            raise RuntimeError(
                f"Expected {head.nl} {attr} branches, found {len(branches)}."
            )

        for branch_index, branch in enumerate(branches):
            conv = final_output_conv(branch, f"{attr}[{branch_index}]")
            if conv.out_channels != expected_outputs[attr]:
                raise RuntimeError(
                    f"Unexpected {attr}[{branch_index}] output width: "
                    f"got {conv.out_channels}, expected {expected_outputs[attr]}."
                )
            protected.append(conv)

    # Segment post-processing multiplies mask coefficients shaped [N, nm]
    # by mask prototypes shaped [nm, H * W]. Protecting cv4 alone is not
    # enough: proto.cv3 must also keep exactly nm output channels.
    proto = getattr(head, "proto", None)
    proto_cv3 = getattr(proto, "cv3", None)
    proto_output_conv = (
        proto_cv3
        if isinstance(proto_cv3, nn.Conv2d)
        else getattr(proto_cv3, "conv", None)
    )

    if not isinstance(proto_output_conv, nn.Conv2d):
        raise TypeError(
            "Expected Segment head proto.cv3 to be an nn.Conv2d or an "
            "Ultralytics Conv wrapper containing .conv."
        )

    if proto_output_conv.out_channels != int(head.nm):
        raise RuntimeError(
            "Unexpected proto.cv3 output width: "
            f"got {proto_output_conv.out_channels}, expected nm={head.nm}."
        )

    protected.append(proto_output_conv)

    return list(dict.fromkeys(protected))


def snapshot_conv_channels(model):
    """Capture every Conv2d's current input/output channel counts."""
    return {
        name: (module.in_channels, module.out_channels)
        for name, module in model.named_modules()
        if isinstance(module, nn.Conv2d)
    }


def print_conv_channels(title, channels):
    print("\n" + title)
    print("-" * 80)
    for name, (in_channels, out_channels) in channels.items():
        print(f"{name:<64} in={in_channels:>4}  out={out_channels:>4}")


def print_conv_channel_changes(before, after):
    print("\nConv2d channel changes (before -> after)")
    print("-" * 80)
    for name, before_shape in before.items():
        after_shape = after.get(name)
        if after_shape is None:
            print(f"{name:<64} removed")
            continue

        marker = "CHANGED" if before_shape != after_shape else "same"
        print(
            f"{name:<64} "
            f"in {before_shape[0]:>4} -> {after_shape[0]:>4}, "
            f"out {before_shape[1]:>4} -> {after_shape[1]:>4}  [{marker}]"
        )

    for name, after_shape in after.items():
        if name not in before:
            print(
                f"{name:<64} added "
                f"(in={after_shape[0]}, out={after_shape[1]})"
            )


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    validation_device = 0 if device.type == "cuda" else "cpu"

    print("=" * 80)
    print("Loading Teacher")
    print("=" * 80)
    print(f"Checkpoint: {TEACHER}")
    print(f"Device:     {device}")

    yolo = YOLO(TEACHER)
    model = yolo.model.to(device).eval()
    original_requires_grad = {
        name: parameter.requires_grad
        for name, parameter in model.named_parameters()
    }

    # Ultralytics loads inference checkpoints with every parameter frozen.
    # Torch-Pruning 1.6.0 traces its dependency graph through autograd; with
    # all parameters frozen, it finds zero pruning groups and silently prunes
    # nothing. Temporarily enable gradients for dependency tracing/pruning.
    trainable_before = sum(original_requires_grad.values())
    model.requires_grad_(True)
    print(
        f"Temporarily enabled gradients for Torch-Pruning graph tracing "
        f"({trainable_before} trainable parameter tensors in checkpoint)."
    )

    example_inputs = torch.randn(1, 3, IMAGE_SIZE, IMAGE_SIZE, device=device)

    before_channels = snapshot_conv_channels(model)
    print_conv_channels("Conv2d channels before pruning", before_channels)

    base_macs, base_params = tp.utils.count_ops_and_params(model, example_inputs)
    print("\nBefore pruning")
    print("-" * 80)
    print(f"Params: {base_params / 1e6:.6f} M ({base_params:,})")
    print(f"MACs:   {base_macs / 1e9:.6f} G ({base_macs:,})")

    protected_layers = collect_protected_output_convs(model)
    module_names = qualified_module_names(model)

    print("\nProtected final prediction Conv2d layers")
    print("-" * 80)
    for index, layer in enumerate(protected_layers, start=1):
        print(
            f"{index:>2}. {module_names[layer]}: "
            f"in={layer.in_channels}, out={layer.out_channels}, "
            f"kernel={layer.kernel_size}"
        )
    print(f"Total protected Conv2d layers: {len(protected_layers)}")

    importance = tp.importance.MagnitudeImportance(p=2)
    pruner = ValidatedMagnitudePruner(
        model=model,
        example_inputs=example_inputs,
        importance=importance,
        pruning_ratio=PRUNING_RATIO,
        iterative_steps=1,
        ignored_layers=protected_layers,
        round_to=None,
    )

    print("\n" + "=" * 80)
    print("Start structured channel pruning: 10%")
    print("=" * 80)
    pruner.step()

    if pruner.skipped_invalid_groups:
        print("\nSkipped Torch-Pruning groups with invalid dependency indices")
        print("-" * 80)
        seen_roots = set()
        for root_module, invalid_items in pruner.skipped_invalid_groups:
            root_name = module_names.get(root_module, repr(root_module))
            if root_name in seen_roots:
                continue
            seen_roots.add(root_name)
            details = "; ".join(
                f"{module_names.get(target, type(target).__name__)} "
                f"{axis}: max_index={max_index}, channels={channel_count}"
                for target, axis, max_index, channel_count in invalid_items
            )
            print(f"{root_name}: {details}")

    # Restore the checkpoint's original frozen/trainable state by name.
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(original_requires_grad.get(name, False))

    after_channels = snapshot_conv_channels(model)
    print_conv_channels("Conv2d channels after pruning", after_channels)
    print_conv_channel_changes(before_channels, after_channels)

    pruned_macs, pruned_params = tp.utils.count_ops_and_params(model, example_inputs)
    params_reduction = 100.0 * (1.0 - pruned_params / base_params)
    macs_reduction = 100.0 * (1.0 - pruned_macs / base_macs)

    print("\nAfter pruning")
    print("-" * 80)
    print(f"Params: {pruned_params / 1e6:.6f} M ({pruned_params:,})")
    print(f"MACs:   {pruned_macs / 1e9:.6f} G ({pruned_macs:,})")
    print("\nReduction")
    print("-" * 80)
    print(f"Params: {params_reduction:.2f}%")
    print(f"MACs:   {macs_reduction:.2f}%")

    if pruned_params >= base_params and pruned_macs >= base_macs:
        raise RuntimeError(
            "Pruning produced no Params or MACs reduction. "
            "Stop instead of reporting this as a successful dry run."
        )

    print("\n" + "=" * 80)
    print("Forward compatibility test")
    print("=" * 80)
    model.eval()
    with torch.no_grad():
        model(example_inputs)
    print("Forward PASS")

    print("\n" + "=" * 80)
    print("Validation after pruning")
    print("=" * 80)
    yolo.model = model
    metrics = yolo.val(
        data=DATA,
        split="val",
        imgsz=IMAGE_SIZE,
        batch=BATCH_SIZE,
        device=validation_device,
        workers=0,
        project=str(OUTPUT_DIR),
        name="val",
        exist_ok=True,
        plots=True,
        verbose=True,
    )

    print("\n" + "=" * 80)
    print("PRUNED MODEL RESULTS")
    print("=" * 80)
    print(f"Box mAP50:      {metrics.box.map50:.5f}")
    print(f"Box mAP50-95:   {metrics.box.map:.5f}")
    print(f"Mask mAP50:     {metrics.seg.map50:.5f}")
    print(f"Mask mAP50-95:  {metrics.seg.map:.5f}")

    model.zero_grad(set_to_none=True)
    model.cpu()
    torch.save(model, SAVE_PATH)
    print(f"\nSaved full structurally pruned model to:\n{SAVE_PATH}")


if __name__ == "__main__":
    main()
