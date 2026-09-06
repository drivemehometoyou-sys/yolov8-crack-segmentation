from pathlib import Path
import csv
import math
import sys

import cv2
import numpy as np

try:
    from skimage.morphology import skeletonize
except ImportError:
    print("=" * 70)
    print("ERROR: scikit-image is not installed.")
    print("Install it with:")
    print("pip install scikit-image")
    print("=" * 70)
    sys.exit(1)

from ultralytics import YOLO


# ============================================================
# Configuration
# ============================================================

MODEL_PATH = Path(
    "D:/CrackSegRuns/"
    "pruned_student_10_kd_30e/"
    "weights/best.pt"
)

INPUT_PATHS = [
    Path(
        "D:/crack-seg/images/test/"
        "2413.rf.669f2c0f7a1d89a840314907bb8be452.jpg"
    ),
    Path(
        "D:/crack-seg/images/test/"
        "1686.rf.809fb1b51c607e5cf787e44ef4ddd7b8.jpg"
    ),
    Path(
        "D:/crack-seg/images/test/"
        "1616.rf.c868709931a671796794fdbb95352c5a.jpg"
    ),
    Path(
        "D:/crack-seg/images/test/"
        "3848.rf.eebe99038cd40502695607594e000258.jpg"
    ),
    Path(
        "D:/crack-seg/images/test/"
        "3885.rf.5d0b08090bb3c2fcd3bc8fe36db9e51a.jpg"
    ),
]

OUTPUT_ROOT = Path(
    "D:/CrackSegRuns/"
    "crack_quantification_merged_v2_selected5"
)

VIS_DIR = OUTPUT_ROOT / "visualizations"

CSV_PATH = OUTPUT_ROOT / "crack_measurements.csv"


# ============================================================
# Inference settings
# ============================================================

IMGSZ = 640
CONF = 0.25
DEVICE = 0


# ============================================================
# Connected-component filtering
# ============================================================

# Only remove extremely tiny isolated prediction fragments.
# No erosion / dilation / closing is used.
MIN_COMPONENT_AREA = 20


# ============================================================
# Visualization
# ============================================================

PANEL_WIDTH = 340

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


# ============================================================
# Load final KD Student
# ============================================================

def load_model():

    if not MODEL_PATH.is_file():
        raise FileNotFoundError(
            f"Model not found:\n{MODEL_PATH}"
        )

    print("=" * 80)
    print("Loading final KD Student")
    print("=" * 80)
    print(f"Model: {MODEL_PATH}")

    model = YOLO(
        str(MODEL_PATH),
        task="segment",
    )

    if getattr(
        model.model,
        "task",
        None,
    ) is None:

        print(
            "NOTE: model.task metadata is missing; "
            "setting model.task='segment'."
        )

        model.model.task = "segment"

    print("Model loaded successfully.")
    print()

    return model


# ============================================================
# Collect selected images
# ============================================================

def collect_images():

    valid_images = []

    for image_path in INPUT_PATHS:

        if not image_path.is_file():

            print(
                f"WARNING: image not found:\n"
                f"{image_path}"
            )

            continue

        valid_images.append(
            image_path
        )

    if not valid_images:
        raise RuntimeError(
            "No valid input images found."
        )

    return valid_images


# ============================================================
# Resize mask to original image resolution
# ============================================================

def resize_mask_to_original(
    mask,
    height,
    width,
):

    if mask.shape == (
        height,
        width,
    ):
        return mask.astype(bool)

    resized = cv2.resize(
        mask.astype(np.uint8),
        (
            width,
            height,
        ),
        interpolation=cv2.INTER_NEAREST,
    )

    return resized.astype(bool)


# ============================================================
# Improved skeleton length
# ============================================================

def calculate_skeleton_length(
    skeleton,
):
    """
    Calculate total length of an 8-connected skeleton.

    Horizontal / vertical edge:
        weight = 1

    Diagonal edge:
        weight = sqrt(2)

    Important correction:

    A diagonal edge is counted only if neither of the two
    possible orthogonal intermediate pixels exists.

    This avoids shortcut double-counting such as:

        A .
        B C

    where A-B and B-C already describe the local skeleton path,
    so A-C should NOT also be counted as a diagonal edge.

    Unit:
        pixels
    """

    sk = skeleton.astype(bool)

    if np.count_nonzero(sk) <= 1:
        return 0.0

    # --------------------------------------------------------
    # Horizontal edges
    # --------------------------------------------------------

    horizontal = np.logical_and(
        sk[:, :-1],
        sk[:, 1:],
    )

    horizontal_count = np.count_nonzero(
        horizontal
    )

    # --------------------------------------------------------
    # Vertical edges
    # --------------------------------------------------------

    vertical = np.logical_and(
        sk[:-1, :],
        sk[1:, :],
    )

    vertical_count = np.count_nonzero(
        vertical
    )

    # --------------------------------------------------------
    # Main diagonal
    #
    # A .
    # . B
    #
    # Orthogonal intermediate candidates:
    #
    # . X
    # X .
    # --------------------------------------------------------

    diagonal_1 = np.logical_and(
        sk[:-1, :-1],
        sk[1:, 1:],
    )

    bridge_1_a = sk[:-1, 1:]
    bridge_1_b = sk[1:, :-1]

    no_orthogonal_bridge_1 = np.logical_not(
        np.logical_or(
            bridge_1_a,
            bridge_1_b,
        )
    )

    diagonal_1_valid = np.logical_and(
        diagonal_1,
        no_orthogonal_bridge_1,
    )

    # --------------------------------------------------------
    # Anti-diagonal
    #
    # . A
    # B .
    #
    # Orthogonal intermediate candidates:
    #
    # X .
    # . X
    # --------------------------------------------------------

    diagonal_2 = np.logical_and(
        sk[:-1, 1:],
        sk[1:, :-1],
    )

    bridge_2_a = sk[:-1, :-1]
    bridge_2_b = sk[1:, 1:]

    no_orthogonal_bridge_2 = np.logical_not(
        np.logical_or(
            bridge_2_a,
            bridge_2_b,
        )
    )

    diagonal_2_valid = np.logical_and(
        diagonal_2,
        no_orthogonal_bridge_2,
    )

    diagonal_count = (
        np.count_nonzero(
            diagonal_1_valid
        )
        +
        np.count_nonzero(
            diagonal_2_valid
        )
    )

    total_length = (
        horizontal_count
        +
        vertical_count
        +
        math.sqrt(2.0)
        * diagonal_count
    )

    return float(total_length)


# ============================================================
# Width using Euclidean distance transform
# ============================================================

def calculate_width(
    binary_mask,
    skeleton,
):
    """
    Estimate local crack width.

    For every skeleton pixel:

        local_width ≈ 2 * distance_to_background

    Returns:
        mean_width_px
        max_width_px
    """

    if np.count_nonzero(
        skeleton
    ) == 0:

        return 0.0, 0.0

    mask_uint8 = (
        binary_mask.astype(
            np.uint8
        )
        * 255
    )

    distance_map = cv2.distanceTransform(
        mask_uint8,
        cv2.DIST_L2,
        5,
    )

    skeleton_distances = (
        distance_map[
            skeleton
        ]
    )

    if skeleton_distances.size == 0:
        return 0.0, 0.0

    local_widths = (
        2.0
        * skeleton_distances
    )

    mean_width_px = float(
        np.mean(
            local_widths
        )
    )

    max_width_px = float(
        np.max(
            local_widths
        )
    )

    return (
        mean_width_px,
        max_width_px,
    )


# ============================================================
# Measure one connected crack region
# ============================================================

def measure_connected_region(
    region_mask,
):

    region_mask = (
        region_mask.astype(
            bool
        )
    )

    # --------------------------------------------------------
    # Area
    # --------------------------------------------------------

    area_px2 = int(
        np.count_nonzero(
            region_mask
        )
    )

    if area_px2 == 0:

        return {
            "skeleton":
                np.zeros_like(
                    region_mask,
                    dtype=bool,
                ),

            "length_px":
                0.0,

            "mean_width_px":
                0.0,

            "max_width_px":
                0.0,

            "area_px2":
                0,
        }

    # --------------------------------------------------------
    # Skeleton
    # --------------------------------------------------------

    skeleton = skeletonize(
        region_mask
    )

    # --------------------------------------------------------
    # Length
    # --------------------------------------------------------

    length_px = (
        calculate_skeleton_length(
            skeleton
        )
    )

    # --------------------------------------------------------
    # Width
    # --------------------------------------------------------

    (
        mean_width_px,
        max_width_px,
    ) = calculate_width(
        region_mask,
        skeleton,
    )

    return {
        "skeleton":
            skeleton,

        "length_px":
            length_px,

        "mean_width_px":
            mean_width_px,

        "max_width_px":
            max_width_px,

        "area_px2":
            area_px2,
    }


# ============================================================
# Merge all YOLO crack-instance masks
# ============================================================

def merge_instance_masks(
    instance_masks,
):

    if not instance_masks:
        return None

    merged_mask = np.zeros_like(
        instance_masks[0],
        dtype=bool,
    )

    for mask in instance_masks:

        merged_mask = np.logical_or(
            merged_mask,
            mask,
        )

    return merged_mask


# ============================================================
# Connected crack regions
# ============================================================

def extract_connected_crack_regions(
    merged_mask,
):
    """
    Split merged crack prediction using
    8-connected connected-component analysis.
    """

    mask_uint8 = (
        merged_mask.astype(
            np.uint8
        )
    )

    (
        num_labels,
        labels,
        stats,
        centroids,
    ) = cv2.connectedComponentsWithStats(
        mask_uint8,
        connectivity=8,
    )

    regions = []

    # Label 0 is background
    for label_id in range(
        1,
        num_labels,
    ):

        area = int(
            stats[
                label_id,
                cv2.CC_STAT_AREA
            ]
        )

        if area < MIN_COMPONENT_AREA:
            continue

        region_mask = (
            labels
            == label_id
        )

        x = int(
            stats[
                label_id,
                cv2.CC_STAT_LEFT
            ]
        )

        y = int(
            stats[
                label_id,
                cv2.CC_STAT_TOP
            ]
        )

        w = int(
            stats[
                label_id,
                cv2.CC_STAT_WIDTH
            ]
        )

        h = int(
            stats[
                label_id,
                cv2.CC_STAT_HEIGHT
            ]
        )

        regions.append(
            {
                "mask":
                    region_mask,

                "component_area":
                    area,

                "bbox":
                    (
                        x,
                        y,
                        x + w - 1,
                        y + h - 1,
                    ),
            }
        )

    # Largest region first
    regions.sort(
        key=lambda item:
        item["component_area"],
        reverse=True,
    )

    return regions


# ============================================================
# Find original YOLO instances supporting one region
# ============================================================

def find_supporting_instances(
    region_mask,
    instance_masks,
    confidences,
):

    supporting_indices = []
    supporting_confidences = []

    for index, instance_mask in enumerate(
        instance_masks
    ):

        overlap = np.logical_and(
            region_mask,
            instance_mask,
        )

        if np.any(overlap):

            supporting_indices.append(
                index
            )

            supporting_confidences.append(
                float(
                    confidences[index]
                )
            )

    if supporting_confidences:

        max_confidence = max(
            supporting_confidences
        )

    else:

        max_confidence = 0.0

    return (
        supporting_indices,
        max_confidence,
    )


# ============================================================
# Visualization mask overlay
# ============================================================

def overlay_mask(
    image,
    mask,
    color,
    alpha=0.35,
):

    overlay = image.copy()

    overlay[
        mask
    ] = color

    blended = cv2.addWeighted(
        overlay,
        alpha,
        image,
        1.0 - alpha,
        0,
    )

    image[:] = blended


# ============================================================
# Canvas with right information panel
# ============================================================

def create_visualization_canvas(
    image,
):

    height, width = (
        image.shape[:2]
    )

    canvas = np.zeros(
        (
            height,
            width + PANEL_WIDTH,
            3,
        ),
        dtype=np.uint8,
    )

    canvas[
        :height,
        :width
    ] = image

    canvas[
        :height,
        width:
    ] = (
        30,
        30,
        30,
    )

    return canvas


# ============================================================
# Draw right panel
# ============================================================

def draw_panel_information(
    canvas,
    original_width,
    measurements,
    raw_instance_count,
):

    font = (
        cv2.FONT_HERSHEY_SIMPLEX
    )

    x = (
        original_width
        + 15
    )

    y = 28

    # --------------------------------------------------------
    # Title
    # --------------------------------------------------------

    cv2.putText(
        canvas,
        "Crack Quantification",
        (
            x,
            y,
        ),
        font,
        0.58,
        (
            255,
            255,
            255,
        ),
        1,
        cv2.LINE_AA,
    )

    y += 28

    # --------------------------------------------------------
    # Raw instance count
    # --------------------------------------------------------

    cv2.putText(
        canvas,
        (
            f"YOLO instances: "
            f"{raw_instance_count}"
        ),
        (
            x,
            y,
        ),
        font,
        0.45,
        (
            200,
            200,
            200,
        ),
        1,
        cv2.LINE_AA,
    )

    y += 20

    # --------------------------------------------------------
    # Connected region count
    # --------------------------------------------------------

    cv2.putText(
        canvas,
        (
            f"Connected regions: "
            f"{len(measurements)}"
        ),
        (
            x,
            y,
        ),
        font,
        0.45,
        (
            200,
            200,
            200,
        ),
        1,
        cv2.LINE_AA,
    )

    y += 30

    # --------------------------------------------------------
    # Each connected crack region
    # --------------------------------------------------------

    for item in measurements:

        color = item[
            "color"
        ]

        lines = [
            (
                f"Region "
                f"#{item['connected_crack_region_id']}"
            ),
            (
                f"Max conf: "
                f"{item['max_confidence']:.2f}"
            ),
            (
                f"Supporting instances: "
                f"{item['supporting_instances']}"
            ),
            (
                f"L: "
                f"{item['length_px']:.1f} px"
            ),
            (
                f"Wmean: "
                f"{item['mean_width_px']:.1f} px"
            ),
            (
                f"Wmax: "
                f"{item['max_width_px']:.1f} px"
            ),
            (
                f"Area: "
                f"{item['area_px2']} px^2"
            ),
        ]

        block_height = (
            18
            * len(lines)
            + 10
        )

        # If panel is full, stop drawing further blocks.
        if (
            y
            + block_height
            >= canvas.shape[0]
        ):

            cv2.putText(
                canvas,
                "...",
                (
                    x,
                    min(
                        y,
                        canvas.shape[0] - 10,
                    ),
                ),
                font,
                0.5,
                (
                    255,
                    255,
                    255,
                ),
                1,
                cv2.LINE_AA,
            )

            break

        cv2.rectangle(
            canvas,
            (
                x - 5,
                y - 15,
            ),
            (
                original_width
                + PANEL_WIDTH
                - 10,
                y
                + 18
                * len(lines)
                + 3,
            ),
            color,
            1,
        )

        for line in lines:

            cv2.putText(
                canvas,
                line,
                (
                    x,
                    y,
                ),
                font,
                0.42,
                (
                    255,
                    255,
                    255,
                ),
                1,
                cv2.LINE_AA,
            )

            y += 18

        y += 15


# ============================================================
# Process one image
# ============================================================

def process_image(
    model,
    image_path,
):

    image = cv2.imread(
        str(image_path)
    )

    if image is None:

        raise RuntimeError(
            f"Failed to read image:\n"
            f"{image_path}"
        )

    height, width = (
        image.shape[:2]
    )

    # --------------------------------------------------------
    # YOLO segmentation inference
    # --------------------------------------------------------

    results = model.predict(
        source=str(
            image_path
        ),
        imgsz=IMGSZ,
        conf=CONF,
        device=DEVICE,
        retina_masks=True,
        save=False,
        verbose=False,
    )

    if not results:

        canvas = (
            create_visualization_canvas(
                image
            )
        )

        return (
            [],
            canvas,
            0,
        )

    result = (
        results[0]
    )

    # --------------------------------------------------------
    # No detections
    # --------------------------------------------------------

    if (
        result.masks is None
        or
        result.boxes is None
        or
        len(result.boxes) == 0
    ):

        canvas = (
            create_visualization_canvas(
                image
            )
        )

        draw_panel_information(
            canvas,
            width,
            [],
            0,
        )

        return (
            [],
            canvas,
            0,
        )

    # --------------------------------------------------------
    # Raw YOLO masks
    # --------------------------------------------------------

    raw_masks = (
        result.masks.data
        .detach()
        .cpu()
        .numpy()
    )

    confidences = (
        result.boxes.conf
        .detach()
        .cpu()
        .numpy()
    )

    raw_instance_count = min(
        len(raw_masks),
        len(confidences),
    )

    instance_masks = []
    valid_confidences = []

    # --------------------------------------------------------
    # Convert every YOLO instance to original-resolution mask
    # --------------------------------------------------------

    for i in range(
        raw_instance_count
    ):

        binary_mask = (
            raw_masks[i]
            > 0.5
        )

        binary_mask = (
            resize_mask_to_original(
                binary_mask,
                height,
                width,
            )
        )

        instance_masks.append(
            binary_mask
        )

        valid_confidences.append(
            float(
                confidences[i]
            )
        )

    # --------------------------------------------------------
    # Merge all crack masks
    # --------------------------------------------------------

    merged_mask = (
        merge_instance_masks(
            instance_masks
        )
    )

    if merged_mask is None:

        canvas = (
            create_visualization_canvas(
                image
            )
        )

        return (
            [],
            canvas,
            raw_instance_count,
        )

    # --------------------------------------------------------
    # Connected-component analysis
    # --------------------------------------------------------

    regions = (
        extract_connected_crack_regions(
            merged_mask
        )
    )

    visualization = (
        image.copy()
    )

    measurements = []

    # --------------------------------------------------------
    # Measure each connected crack region
    # --------------------------------------------------------

    for region_index, region in enumerate(
        regions,
        start=1,
    ):

        region_mask = (
            region[
                "mask"
            ]
        )

        geometry = (
            measure_connected_region(
                region_mask
            )
        )

        (
            supporting_indices,
            max_confidence,
        ) = find_supporting_instances(
            region_mask,
            instance_masks,
            valid_confidences,
        )

        color = COLORS[
            (
                region_index
                - 1
            )
            % len(COLORS)
        ]

        # ----------------------------------------------------
        # Mask overlay
        # ----------------------------------------------------

        overlay_mask(
            visualization,
            region_mask,
            color,
            alpha=0.35,
        )

        # ----------------------------------------------------
        # Skeleton
        # ----------------------------------------------------

        skeleton = (
            geometry[
                "skeleton"
            ]
        )

        visualization[
            skeleton
        ] = (
            255,
            255,
            255,
        )

        # ----------------------------------------------------
        # Bounding rectangle
        # ----------------------------------------------------

        (
            x1,
            y1,
            x2,
            y2,
        ) = region[
            "bbox"
        ]

        cv2.rectangle(
            visualization,
            (
                x1,
                y1,
            ),
            (
                x2,
                y2,
            ),
            color,
            1,
        )

        # ----------------------------------------------------
        # Store result
        # ----------------------------------------------------

        measurements.append(
            {
                "image":
                    image_path.name,

                "connected_crack_region_id":
                    region_index,

                "max_confidence":
                    max_confidence,

                "supporting_instances":
                    len(
                        supporting_indices
                    ),

                "length_px":
                    geometry[
                        "length_px"
                    ],

                "mean_width_px":
                    geometry[
                        "mean_width_px"
                    ],

                "max_width_px":
                    geometry[
                        "max_width_px"
                    ],

                "area_px2":
                    geometry[
                        "area_px2"
                    ],

                "color":
                    color,
            }
        )

    # --------------------------------------------------------
    # Build final visualization
    # --------------------------------------------------------

    canvas = (
        create_visualization_canvas(
            visualization
        )
    )

    draw_panel_information(
        canvas,
        width,
        measurements,
        raw_instance_count,
    )

    return (
        measurements,
        canvas,
        raw_instance_count,
    )


# ============================================================
# Save CSV
# ============================================================

def save_csv(
    measurements,
):

    fieldnames = [
        "image",
        "connected_crack_region_id",
        "max_confidence",
        "supporting_instances",
        "length_px",
        "mean_width_px",
        "max_width_px",
        "area_px2",
    ]

    with CSV_PATH.open(
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as file:

        writer = (
            csv.DictWriter(
                file,
                fieldnames=fieldnames,
            )
        )

        writer.writeheader()

        for item in measurements:

            writer.writerow(
                {
                    "image":
                        item[
                            "image"
                        ],

                    "connected_crack_region_id":
                        item[
                            "connected_crack_region_id"
                        ],

                    "max_confidence":
                        (
                            f"{item['max_confidence']:.6f}"
                        ),

                    "supporting_instances":
                        item[
                            "supporting_instances"
                        ],

                    "length_px":
                        (
                            f"{item['length_px']:.4f}"
                        ),

                    "mean_width_px":
                        (
                            f"{item['mean_width_px']:.4f}"
                        ),

                    "max_width_px":
                        (
                            f"{item['max_width_px']:.4f}"
                        ),

                    "area_px2":
                        item[
                            "area_px2"
                        ],
                }
            )


# ============================================================
# Main
# ============================================================

def main():

    OUTPUT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    VIS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    model = load_model()

    images = (
        collect_images()
    )

    print("=" * 80)
    print(
        "Connected crack region quantification"
    )
    print("=" * 80)

    print(
        f"Selected images: "
        f"{len(images)}"
    )

    print(
        f"Output: "
        f"{OUTPUT_ROOT}"
    )

    print(
        f"Confidence threshold: "
        f"{CONF}"
    )

    print(
        f"Min component area: "
        f"{MIN_COMPONENT_AREA}px^2"
    )

    print()

    all_measurements = []

    total_raw_instances = 0
    total_connected_regions = 0

    # --------------------------------------------------------
    # Process selected images
    # --------------------------------------------------------

    for index, image_path in enumerate(
        images,
        start=1,
    ):

        print(
            f"[{index}/{len(images)}] "
            f"{image_path.name}"
        )

        try:

            (
                measurements,
                visualization,
                raw_instance_count,
            ) = process_image(
                model,
                image_path,
            )

        except Exception as exc:

            print(
                f"  ERROR: {exc}"
            )

            continue

        total_raw_instances += (
            raw_instance_count
        )

        total_connected_regions += (
            len(
                measurements
            )
        )

        all_measurements.extend(
            measurements
        )

        print(
            f"  Raw YOLO instances: "
            f"{raw_instance_count}"
        )

        print(
            f"  Connected crack regions: "
            f"{len(measurements)}"
        )

        for item in measurements:

            print(
                "  "
                f"Region "
                f"#{item['connected_crack_region_id']} | "
                f"Conf="
                f"{item['max_confidence']:.3f} | "
                f"Support="
                f"{item['supporting_instances']} | "
                f"L="
                f"{item['length_px']:.2f}px | "
                f"Wmean="
                f"{item['mean_width_px']:.2f}px | "
                f"Wmax="
                f"{item['max_width_px']:.2f}px | "
                f"Area="
                f"{item['area_px2']}px^2"
            )

        if visualization is not None:

            output_path = (
                VIS_DIR
                /
                (
                    image_path.stem
                    + "_merged_v2_quantified.jpg"
                )
            )

            success = cv2.imwrite(
                str(
                    output_path
                ),
                visualization,
            )

            if success:

                print(
                    f"  Saved: "
                    f"{output_path.name}"
                )

            else:

                print(
                    "  WARNING: failed to save visualization."
                )

        print()

    # ========================================================
    # CSV
    # ========================================================

    save_csv(
        all_measurements
    )

    # ========================================================
    # Final summary
    # ========================================================

    print("=" * 80)
    print(
        "CONNECTED CRACK REGION QUANTIFICATION COMPLETED"
    )
    print("=" * 80)

    print(
        f"Processed images: "
        f"{len(images)}"
    )

    print(
        f"Raw YOLO instances: "
        f"{total_raw_instances}"
    )

    print(
        f"Connected crack regions: "
        f"{total_connected_regions}"
    )

    print()

    print("CSV:")
    print(CSV_PATH)

    print()

    print("Visualizations:")
    print(VIS_DIR)

    print()

    print("Pipeline:")
    print(
        "YOLO instance masks"
    )
    print(
        "  -> mask union"
    )
    print(
        "  -> 8-connected components"
    )
    print(
        "  -> connected crack regions"
    )
    print(
        "  -> skeletonization"
    )
    print(
        "  -> corrected graph length"
    )
    print(
        "  -> distance-transform width"
    )
    print(
        "  -> pixel area"
    )

    print()

    print("Units:")
    print("  Length     = px")
    print("  Mean Width = px")
    print("  Max Width  = px")
    print("  Area       = px^2")

    print()

    print(
        "NOTE:"
    )

    print(
        "No erosion, dilation or closing "
        "is applied."
    )

    print(
        "No physical scale calibration "
        "is applied."
    )

    print(
        "Therefore all measurements remain "
        "in pixel-domain units."
    )


if __name__ == "__main__":
    main()