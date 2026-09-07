# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Quantify connected crack regions for the complete independent test set."""

from __future__ import annotations

import csv
import json
import math
import sys
from pathlib import Path

import cv2
import numpy as np

try:
    import skimage
    from skimage.morphology import skeletonize
except ImportError:
    print("=" * 80)
    print("ERROR: scikit-image is not installed.")
    print("Install it with: pip install scikit-image")
    print("=" * 80)
    sys.exit(1)

from ultralytics import YOLO


MODEL_PATH = Path("D:/CrackSegRuns/pruned_student_10_kd_30e/weights/best.pt")
INPUT_ROOT = Path("D:/crack-seg/images/test")
OUTPUT_ROOT = Path("D:/CrackSegRuns/crack_quantification_test_all")
VIS_DIR = OUTPUT_ROOT / "visualizations"
CSV_PATH = OUTPUT_ROOT / "crack_measurements.csv"
SUMMARY_PATH = OUTPUT_ROOT / "quantification_summary.json"

IMGSZ = 640
CONF = 0.25
DEVICE = 0
MIN_COMPONENT_AREA = 20
PANEL_WIDTH = 340
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
COLORS = [
    (0, 255, 0),
    (255, 0, 0),
    (0, 0, 255),
    (0, 255, 255),
    (255, 0, 255),
    (255, 255, 0),
    (0, 165, 255),
    (255, 0, 128),
]
CSV_FIELDS = [
    "image",
    "connected_crack_region_id",
    "max_confidence",
    "supporting_instances",
    "length_px",
    "mean_width_px",
    "max_width_px",
    "area_px2",
]


def load_model():
    """Load the final KD student as a segmentation model."""
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

    print(f"scikit-image check: OK (version {skimage.__version__})")
    print(f"Loading final KD Student: {MODEL_PATH}")
    model = YOLO(str(MODEL_PATH), task="segment")
    if getattr(model.model, "task", None) is None:
        print("NOTE: model.task metadata is missing; setting model.task='segment'.")
        model.model.task = "segment"
    return model


def collect_images():
    """Recursively collect supported image files in stable order."""
    if not INPUT_ROOT.is_dir():
        raise FileNotFoundError(f"Test image directory not found: {INPUT_ROOT}")

    images = sorted(
        (path for path in INPUT_ROOT.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS),
        key=lambda path: path.relative_to(INPUT_ROOT).as_posix().lower(),
    )
    if not images:
        raise RuntimeError(f"No supported images found under: {INPUT_ROOT}")
    return images


def resize_mask_to_original(mask, original_height, original_width):
    """Resize a predicted mask to the original image resolution."""
    if mask.shape == (original_height, original_width):
        return mask.astype(bool)
    resized = cv2.resize(
        mask.astype(np.uint8),
        (original_width, original_height),
        interpolation=cv2.INTER_NEAREST,
    )
    return resized.astype(bool)


def calculate_skeleton_length(skeleton):
    """Calculate corrected 8-neighbor skeleton graph length in pixels.

    Horizontal and vertical edges have weight 1. Diagonal edges have weight
    sqrt(2), but a diagonal is omitted when either orthogonal intermediate
    pixel exists, preventing a diagonal shortcut from being counted in
    addition to an existing orthogonal path.
    """
    sk = skeleton.astype(bool)
    if np.count_nonzero(sk) <= 1:
        return 0.0

    horizontal_count = np.count_nonzero(sk[:, :-1] & sk[:, 1:])
    vertical_count = np.count_nonzero(sk[:-1, :] & sk[1:, :])

    diagonal_1 = sk[:-1, :-1] & sk[1:, 1:]
    diagonal_1_valid = diagonal_1 & ~(sk[:-1, 1:] | sk[1:, :-1])

    diagonal_2 = sk[:-1, 1:] & sk[1:, :-1]
    diagonal_2_valid = diagonal_2 & ~(sk[:-1, :-1] | sk[1:, 1:])

    diagonal_count = np.count_nonzero(diagonal_1_valid) + np.count_nonzero(diagonal_2_valid)
    return float(horizontal_count + vertical_count + math.sqrt(2.0) * diagonal_count)


def calculate_width(binary_mask, skeleton):
    """Return mean and maximum distance-transform width on skeleton pixels."""
    if not np.any(skeleton):
        return 0.0, 0.0
    distance_map = cv2.distanceTransform(binary_mask.astype(np.uint8) * 255, cv2.DIST_L2, 5)
    distances = distance_map[skeleton]
    if distances.size == 0:
        return 0.0, 0.0
    widths = 2.0 * distances
    return float(np.mean(widths)), float(np.max(widths))


def measure_connected_region(region_mask):
    """Measure foreground area, corrected skeleton length, and width."""
    region_mask = region_mask.astype(bool)
    area_px2 = int(np.count_nonzero(region_mask))
    if area_px2 == 0:
        return {
            "skeleton": np.zeros_like(region_mask, dtype=bool),
            "length_px": 0.0,
            "mean_width_px": 0.0,
            "max_width_px": 0.0,
            "area_px2": 0,
        }

    skeleton = skeletonize(region_mask)
    mean_width_px, max_width_px = calculate_width(region_mask, skeleton)
    return {
        "skeleton": skeleton,
        "length_px": calculate_skeleton_length(skeleton),
        "mean_width_px": mean_width_px,
        "max_width_px": max_width_px,
        "area_px2": area_px2,
    }


def merge_instance_masks(instance_masks):
    """Union all original-resolution YOLO instance masks."""
    if not instance_masks:
        return None
    merged_mask = np.zeros_like(instance_masks[0], dtype=bool)
    for mask in instance_masks:
        merged_mask |= mask
    return merged_mask


def extract_connected_crack_regions(merged_mask, min_component_area=MIN_COMPONENT_AREA):
    """Extract 8-connected regions and count components below the area limit."""
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(merged_mask.astype(np.uint8), connectivity=8)
    regions = []
    filtered_small_components = 0
    for label_id in range(1, num_labels):
        area = int(stats[label_id, cv2.CC_STAT_AREA])
        if area < min_component_area:
            filtered_small_components += 1
            continue
        x = int(stats[label_id, cv2.CC_STAT_LEFT])
        y = int(stats[label_id, cv2.CC_STAT_TOP])
        width = int(stats[label_id, cv2.CC_STAT_WIDTH])
        height = int(stats[label_id, cv2.CC_STAT_HEIGHT])
        regions.append(
            {
                "mask": labels == label_id,
                "component_area": area,
                "bbox": (x, y, x + width - 1, y + height - 1),
            }
        )
    regions.sort(key=lambda item: item["component_area"], reverse=True)
    return regions, filtered_small_components


def find_supporting_instances(region_mask, instance_masks, confidences):
    """Find YOLO instances that overlap a connected union region."""
    supporting_confidences = [
        float(confidences[index])
        for index, instance_mask in enumerate(instance_masks)
        if np.any(region_mask & instance_mask)
    ]
    return len(supporting_confidences), max(supporting_confidences, default=0.0)


def overlay_mask(image, mask, color, alpha=0.35):
    """Blend one colored region mask onto an image."""
    overlay = image.copy()
    overlay[mask] = color
    image[:] = cv2.addWeighted(overlay, alpha, image, 1.0 - alpha, 0)


def create_visualization_canvas(image):
    """Create an image canvas with a dark information panel on the right."""
    height, width = image.shape[:2]
    canvas = np.zeros((height, width + PANEL_WIDTH, 3), dtype=np.uint8)
    canvas[:, :width] = image
    canvas[:, width:] = (30, 30, 30)
    return canvas


def draw_panel_information(canvas, original_width, measurements, raw_instance_count):
    """Draw detection and geometry information in the right-side panel."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    x = original_width + 15
    y = 28

    cv2.putText(canvas, "Crack Quantification", (x, y), font, 0.58, (255, 255, 255), 1, cv2.LINE_AA)
    y += 28
    cv2.putText(canvas, f"YOLO instances: {raw_instance_count}", (x, y), font, 0.45, (200, 200, 200), 1, cv2.LINE_AA)
    y += 20
    cv2.putText(canvas, f"Connected regions: {len(measurements)}", (x, y), font, 0.45, (200, 200, 200), 1, cv2.LINE_AA)
    y += 30

    if raw_instance_count == 0:
        cv2.putText(canvas, "No crack detected", (x, y), font, 0.58, (80, 180, 255), 1, cv2.LINE_AA)
        return
    if not measurements:
        cv2.putText(canvas, "No retained region", (x, y), font, 0.50, (80, 180, 255), 1, cv2.LINE_AA)
        y += 22
        cv2.putText(
            canvas, f"Area threshold: {MIN_COMPONENT_AREA}", (x, y), font, 0.42, (200, 200, 200), 1, cv2.LINE_AA
        )
        return

    for item in measurements:
        color = item["color"]
        lines = [
            f"Region #{item['connected_crack_region_id']}",
            f"Max conf: {item['max_confidence']:.2f}",
            f"Supporting instances: {item['supporting_instances']}",
            f"L: {item['length_px']:.1f} px",
            f"Wmean: {item['mean_width_px']:.1f} px",
            f"Wmax: {item['max_width_px']:.1f} px",
            f"Area: {item['area_px2']} px^2",
        ]
        block_height = 18 * len(lines) + 10
        if y + block_height >= canvas.shape[0]:
            cv2.putText(canvas, "...", (x, min(y, canvas.shape[0] - 10)), font, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
            break
        cv2.rectangle(
            canvas,
            (x - 5, y - 15),
            (original_width + PANEL_WIDTH - 10, y + 18 * len(lines) + 3),
            color,
            1,
        )
        for line in lines:
            cv2.putText(canvas, line, (x, y), font, 0.42, (255, 255, 255), 1, cv2.LINE_AA)
            y += 18
        y += 15


def process_image(model, image_path):
    """Run inference and quantify every retained connected region in one image."""
    image = cv2.imread(str(image_path))
    if image is None:
        raise RuntimeError("OpenCV could not read the image")
    height, width = image.shape[:2]

    results = model.predict(
        source=str(image_path),
        imgsz=IMGSZ,
        conf=CONF,
        device=DEVICE,
        retina_masks=True,
        save=False,
        verbose=False,
    )
    result = results[0] if results else None
    if result is None or result.masks is None or result.boxes is None or len(result.boxes) == 0:
        canvas = create_visualization_canvas(image)
        draw_panel_information(canvas, width, [], 0)
        return [], canvas, 0, 0

    raw_masks = result.masks.data.detach().cpu().numpy()
    confidences = result.boxes.conf.detach().cpu().numpy()
    raw_instance_count = min(len(raw_masks), len(confidences))
    instance_masks = [
        resize_mask_to_original(raw_masks[index] > 0.5, height, width) for index in range(raw_instance_count)
    ]
    valid_confidences = [float(confidences[index]) for index in range(raw_instance_count)]
    merged_mask = merge_instance_masks(instance_masks)

    if merged_mask is None:
        canvas = create_visualization_canvas(image)
        draw_panel_information(canvas, width, [], raw_instance_count)
        return [], canvas, raw_instance_count, 0

    regions, filtered_small_components = extract_connected_crack_regions(merged_mask)
    visualization = image.copy()
    measurements = []
    relative_image = image_path.relative_to(INPUT_ROOT).as_posix()

    for region_index, region in enumerate(regions, start=1):
        region_mask = region["mask"]
        geometry = measure_connected_region(region_mask)
        supporting_instances, max_confidence = find_supporting_instances(region_mask, instance_masks, valid_confidences)
        color = COLORS[(region_index - 1) % len(COLORS)]
        overlay_mask(visualization, region_mask, color)
        visualization[geometry["skeleton"]] = (255, 255, 255)
        x1, y1, x2, y2 = region["bbox"]
        cv2.rectangle(visualization, (x1, y1), (x2, y2), color, 1)
        measurements.append(
            {
                "image": relative_image,
                "connected_crack_region_id": region_index,
                "max_confidence": max_confidence,
                "supporting_instances": supporting_instances,
                "length_px": geometry["length_px"],
                "mean_width_px": geometry["mean_width_px"],
                "max_width_px": geometry["max_width_px"],
                "area_px2": geometry["area_px2"],
                "color": color,
            }
        )

    canvas = create_visualization_canvas(visualization)
    draw_panel_information(canvas, width, measurements, raw_instance_count)
    return measurements, canvas, raw_instance_count, filtered_small_components


def visualization_path(image_path):
    """Preserve relative subdirectories and make visualization names unique."""
    relative = image_path.relative_to(INPUT_ROOT)
    return VIS_DIR / relative.parent / f"{relative.name}_quantified.jpg"


def save_visualization(image_path, visualization):
    """Save one visualization and fail explicitly if OpenCV cannot write it."""
    output_path = visualization_path(image_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), visualization):
        raise RuntimeError(f"Failed to save visualization: {output_path}")
    return output_path


def save_csv(measurements):
    """Write one CSV row per retained connected crack region."""
    with CSV_PATH.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for item in measurements:
            writer.writerow(
                {
                    "image": item["image"],
                    "connected_crack_region_id": item["connected_crack_region_id"],
                    "max_confidence": f"{item['max_confidence']:.6f}",
                    "supporting_instances": item["supporting_instances"],
                    "length_px": f"{item['length_px']:.4f}",
                    "mean_width_px": f"{item['mean_width_px']:.4f}",
                    "max_width_px": f"{item['max_width_px']:.4f}",
                    "area_px2": item["area_px2"],
                }
            )


def save_summary(summary):
    """Write batch-level counts and failure details as JSON."""
    with SUMMARY_PATH.open("w", encoding="utf-8") as file:
        json.dump(summary, file, ensure_ascii=False, indent=2)
        file.write("\n")


def main():
    """Run robust full-test quantification without stopping on single-image errors."""
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    VIS_DIR.mkdir(parents=True, exist_ok=True)
    model = load_model()
    images = collect_images()

    print("=" * 80)
    print("FULL TEST CONNECTED CRACK REGION QUANTIFICATION")
    print("=" * 80)
    print(f"Input images: {len(images)}")
    print(f"Input root: {INPUT_ROOT}")
    print(f"Output root: {OUTPUT_ROOT}")
    print(f"imgsz={IMGSZ}, conf={CONF}, device={DEVICE}, retina_masks=True")
    print(f"8-connectivity, minimum component area={MIN_COMPONENT_AREA}px^2")
    print("No erosion, dilation, or closing is applied.\n")

    all_measurements = []
    stats = {
        "processed_images": 0,
        "images_with_detections": 0,
        "images_without_detections": 0,
        "raw_yolo_instances_total": 0,
        "connected_regions_total": 0,
        "filtered_small_components_total": 0,
        "failed_images": 0,
        "failed_image_paths": [],
    }

    for index, image_path in enumerate(images, start=1):
        relative_image = image_path.relative_to(INPUT_ROOT).as_posix()
        print(f"[{index}/{len(images)}] {relative_image}")
        try:
            measurements, visualization, raw_count, filtered_count = process_image(model, image_path)
            output_path = save_visualization(image_path, visualization)
        except Exception as exc:
            stats["failed_images"] += 1
            stats["failed_image_paths"].append(relative_image)
            print(f"  ERROR: {type(exc).__name__}: {exc}")
            print("  Continuing with the next image.\n")
            continue

        stats["processed_images"] += 1
        stats["raw_yolo_instances_total"] += raw_count
        stats["connected_regions_total"] += len(measurements)
        stats["filtered_small_components_total"] += filtered_count
        if raw_count > 0:
            stats["images_with_detections"] += 1
        else:
            stats["images_without_detections"] += 1
        all_measurements.extend(measurements)
        print(
            f"  YOLO instances={raw_count}, connected regions={len(measurements)}, "
            f"filtered small components={filtered_count}"
        )
        print(f"  Saved: {output_path}\n")

    save_csv(all_measurements)
    save_summary(stats)

    print("=" * 80)
    print("QUANTIFICATION COMPLETED")
    print("=" * 80)
    for key in (
        "processed_images",
        "images_with_detections",
        "images_without_detections",
        "raw_yolo_instances_total",
        "connected_regions_total",
        "filtered_small_components_total",
        "failed_images",
    ):
        print(f"{key}: {stats[key]}")
    if stats["failed_image_paths"]:
        print("Failed image paths:")
        for path in stats["failed_image_paths"]:
            print(f"  - {path}")
    print(f"CSV: {CSV_PATH}")
    print(f"Summary: {SUMMARY_PATH}")
    print(f"Visualizations: {VIS_DIR}")


if __name__ == "__main__":
    main()
