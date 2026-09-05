from ultralytics import YOLO


def evaluate(name, weight_path):
    print("\n" + "=" * 80)
    print(f"Evaluating: {name}")
    print(f"Weights: {weight_path}")
    print("=" * 80)

    model = YOLO(weight_path)

    metrics = model.val(
        data="D:/crack-seg/crack-seg-local.yaml",
        split="val",

        imgsz=640,
        batch=16,
        device=0,
        workers=0,

        project="D:/CrackSegRuns/unified_best_eval",
        name=name,

        plots=True,
        save_json=False,
        verbose=True,
    )

    result = {
        "Model": name,
        "Box_P": metrics.box.mp,
        "Box_R": metrics.box.mr,
        "Box_mAP50": metrics.box.map50,
        "Box_mAP50-95": metrics.box.map,
        "Mask_P": metrics.seg.mp,
        "Mask_R": metrics.seg.mr,
        "Mask_mAP50": metrics.seg.map50,
        "Mask_mAP50-95": metrics.seg.map,
    }

    return result


def main():

    models = {
        "baseline_best": (
            "D:/CrackSegRuns/"
            "yolov8n_seg_baseline/"
            "weights/best.pt"
        ),

        "spdconv_best": (
            "D:/CrackSegRuns/"
            "yolov8n_seg_spdconv_100e/"
            "weights/best.pt"
        ),

        "spdconv_sppfmsa_best": (
            "D:/CrackSegRuns/"
            "yolov8n_seg_spdconv_sppfmsa_100e/"
            "weights/best.pt"
        ),
    }

    results = []

    for name, weight_path in models.items():
        results.append(
            evaluate(name, weight_path)
        )

    print("\n\n")
    print("=" * 120)
    print("UNIFIED BEST.PT EVALUATION RESULTS")
    print("=" * 120)

    header = (
        f"{'Model':<28}"
        f"{'Box P':>10}"
        f"{'Box R':>10}"
        f"{'Box mAP50':>12}"
        f"{'Box mAP50-95':>16}"
        f"{'Mask P':>10}"
        f"{'Mask R':>10}"
        f"{'Mask mAP50':>13}"
        f"{'Mask mAP50-95':>17}"
    )

    print(header)
    print("-" * 120)

    for r in results:
        print(
            f"{r['Model']:<28}"
            f"{r['Box_P']:>10.5f}"
            f"{r['Box_R']:>10.5f}"
            f"{r['Box_mAP50']:>12.5f}"
            f"{r['Box_mAP50-95']:>16.5f}"
            f"{r['Mask_P']:>10.5f}"
            f"{r['Mask_R']:>10.5f}"
            f"{r['Mask_mAP50']:>13.5f}"
            f"{r['Mask_mAP50-95']:>17.5f}"
        )


if __name__ == "__main__":
    main()