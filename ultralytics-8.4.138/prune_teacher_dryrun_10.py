from pathlib import Path

import torch
import torch.nn as nn
import torch_pruning as tp

from ultralytics import YOLO


# ============================================================
# Configuration
# ============================================================

TEACHER = (
    "D:/CrackSegRuns/"
    "yolov8n_seg_spdconv_sppfmsa_100e/"
    "weights/best.pt"
)

DATA = "D:/crack-seg/crack-seg-local.yaml"

OUTPUT_DIR = Path(
    "D:/CrackSegRuns/pruning_dryrun_10"
)


# ============================================================
# Collect protected layers
# ============================================================

def collect_ignored_layers(model):
    """
    Conservative first pruning experiment.

    Protect:
    1. SPDConv
    2. SPPF_MSA
    3. Segment head
    4. Public output channels of C2f blocks
    """

    ignored_layers = []

    # --------------------------------------------------------
    # Protect custom / high-risk modules
    #
    # Combined model layer layout:
    #
    # 3,5,7 : SPDConv
    # 9     : SPPF_MSA
    # last  : Segment
    # --------------------------------------------------------

    protected_module_indices = [
        3,
        5,
        7,
        9,
        len(model.model) - 1,
    ]

    for idx in protected_module_indices:
        module = model.model[idx]

        for submodule in module.modules():
            if isinstance(submodule, nn.Conv2d):
                ignored_layers.append(submodule)

    # --------------------------------------------------------
    # Protect C2f public output widths
    #
    # Internal channels are still allowed to be pruned.
    # --------------------------------------------------------

    c2f_indices = [
        2,
        4,
        6,
        8,
        12,
        15,
        18,
        21,
    ]

    for idx in c2f_indices:
        module = model.model[idx]

        # C2f.cv2 is Ultralytics Conv
        # C2f.cv2.conv is the real nn.Conv2d
        if hasattr(module, "cv2") and hasattr(module.cv2, "conv"):
            ignored_layers.append(module.cv2.conv)

    # Remove duplicates
    ignored_layers = list(dict.fromkeys(ignored_layers))

    return ignored_layers


# ============================================================
# Main
# ============================================================

def main():

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    device = torch.device(
        "cuda:0"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("=" * 80)
    print("Loading Teacher")
    print("=" * 80)

    yolo = YOLO(TEACHER)

    model = yolo.model
    model.to(device)
    model.eval()

    # Example input used for dependency graph tracing
    example_inputs = torch.randn(
        1,
        3,
        640,
        640,
        device=device
    )

    # --------------------------------------------------------
    # Before pruning
    # --------------------------------------------------------

    base_macs, base_params = (
        tp.utils.count_ops_and_params(
            model,
            example_inputs
        )
    )

    print("\nBefore pruning:")
    print(
        f"Params: {base_params / 1e6:.3f} M"
    )
    print(
        f"MACs:   {base_macs / 1e9:.3f} G"
    )

    # --------------------------------------------------------
    # Layers that we do NOT want to prune in first experiment
    # --------------------------------------------------------

    ignored_layers = collect_ignored_layers(
        model
    )

    print(
        f"\nProtected Conv layers: "
        f"{len(ignored_layers)}"
    )

    # --------------------------------------------------------
    # Importance criterion
    # --------------------------------------------------------

    importance = (
        tp.importance.MagnitudeImportance(
            p=2
        )
    )

    # --------------------------------------------------------
    # Structured channel pruner
    # --------------------------------------------------------

    pruner = tp.pruner.MagnitudePruner(
        model=model,
        example_inputs=example_inputs,

        importance=importance,

        # FIRST TEST ONLY: 10%
        pruning_ratio=0.10,

        iterative_steps=1,

        ignored_layers=ignored_layers,

        # Keep channel numbers hardware-friendly
        round_to=8,
    )

    print("\n" + "=" * 80)
    print("Start structured pruning: 10%")
    print("=" * 80)

    # Do NOT put this inside torch.no_grad()
    pruner.step()

    # --------------------------------------------------------
    # After pruning
    # --------------------------------------------------------

    pruned_macs, pruned_params = (
        tp.utils.count_ops_and_params(
            model,
            example_inputs
        )
    )

    print("\nAfter pruning:")
    print(
        f"Params: "
        f"{pruned_params / 1e6:.3f} M"
    )

    print(
        f"MACs:   "
        f"{pruned_macs / 1e9:.3f} G"
    )

    print("\nReduction:")

    print(
        f"Params: "
        f"{100 * (1 - pruned_params / base_params):.2f}%"
    )

    print(
        f"MACs:   "
        f"{100 * (1 - pruned_macs / base_macs):.2f}%"
    )

    # --------------------------------------------------------
    # Forward test
    # --------------------------------------------------------

    print("\n" + "=" * 80)
    print("Forward compatibility test")
    print("=" * 80)

    model.eval()

    with torch.no_grad():
        output = model(example_inputs)

    print("Forward PASS")

    # --------------------------------------------------------
    # Validation
    # --------------------------------------------------------

    print("\n" + "=" * 80)
    print("Validation after pruning")
    print("=" * 80)

    # yolo.model and model refer to the same object.
    yolo.model = model

    metrics = yolo.val(
        data=DATA,
        split="val",

        imgsz=640,
        batch=16,
        device=0,

        # Windows: safest for this experiment
        workers=0,

        project=(
            "D:/CrackSegRuns/"
            "pruning_dryrun_10"
        ),

        name="val",

        plots=True,
        verbose=True,
    )

    print("\n" + "=" * 80)
    print("PRUNED MODEL RESULTS")
    print("=" * 80)

    print(
        f"Box mAP50:      "
        f"{metrics.box.map50:.5f}"
    )

    print(
        f"Box mAP50-95:   "
        f"{metrics.box.map:.5f}"
    )

    print(
        f"Mask mAP50:     "
        f"{metrics.seg.map50:.5f}"
    )

    print(
        f"Mask mAP50-95:  "
        f"{metrics.seg.map:.5f}"
    )

    # --------------------------------------------------------
    # Save the ENTIRE structurally-pruned model
    # --------------------------------------------------------

    model.zero_grad()
    model.cpu()

    save_path = (
        OUTPUT_DIR
        / "teacher_pruned_10_raw.pth"
    )

    torch.save(
        model,
        save_path
    )

    print(
        f"\nSaved full pruned model to:\n"
        f"{save_path}"
    )


if __name__ == "__main__":
    main()