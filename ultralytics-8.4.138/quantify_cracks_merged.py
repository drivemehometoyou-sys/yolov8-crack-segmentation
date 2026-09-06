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
    print("Please install it with:")
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
    "crack_quantification_merged_selected5"
)

VIS_DIR = OUTPUT_ROOT / "visualizations"

CSV_PATH = OUTPUT_ROOT / "crack_measurements.csv"


# ============================================================
# Inference
# ============================================================

IMGSZ = 640
CONF = 0.25
DEVICE = 0


# ============================================================
# Connected-component filtering
# ============================================================

# Only removes extremely tiny isolated components.
# No erosion / dilation is used.
MIN_COMPONENT_AREA = 20


# ============================================================
# Visualization
# ============================================================

PANEL_WIDTH = 330

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
# Model
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
# Input check
# ============================================================

def collect_images():

    valid_images = []

    for path in INPUT_PATHS:

        if not path.is_file():

            print(
                f"WARNING: missing image:\n{path}"
            )

            continue

        valid_images.append(path)

    if not valid_images:
        raise RuntimeError(
            "No valid images found."
        )

    return valid_images


# ============================================================
# Resize mask
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
# Skeleton length
# ============================================================

def calculate_skeleton_length(
    skeleton,
):
    """
    8-connected skeleton graph length.

    Horizontal / vertical:
        1 px

    Diagonal:
        sqrt(2) px
    """

    sk = skeleton.astype(bool)

    if np.count_nonzero(sk) <= 1:
        return 0.0

    horizontal = np.logical_and(
        sk[:, :-1],
        sk[:, 1:],
    )

    vertical = np.logical_and(
        sk[:-1, :],
        sk[1:, :],
    )

    diagonal_1 = np.logical_and(
        sk[:-1, :-1],
        sk[1:, 1:],
    )

    diagonal_2 = np.logical_and(
        sk[:-1, 1:],
        sk[1:, :-1],
    )

    n_horizontal = np.count_nonzero(
        horizontal
    )

    n_vertical = np.count_nonzero(
        vertical
    )

    n_diagonal = (
        np.count_nonzero(
            diagonal_1
        )
        +
        np.count_nonzero(
            diagonal_2
        )
    )

    length = (
        n_horizontal
        +
        n_vertical
        +
        math.sqrt(2.0)
        * n_diagonal
    )

    return float(length)


# ============================================================
# Width
# ============================================================

def calculate_width(
    binary_mask,
    skeleton,
):

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

    distances = distance_map[
        skeleton
    ]

    if distances.size == 0:
        return 0.0, 0.0

    local_widths = (
        2.0
        * distances
    )

    mean_width = float(
        np.mean(
            local_widths
        )
    )

    max_width = float(
        np.max(
            local_widths
        )
    )

    return (
        mean_width,
        max_width,
    )


# ============================================================
# Geometry measurement
# ============================================================

def measure_physical_crack(
    mask,
):

    mask = mask.astype(bool)

    area_px2 = int(
        np.count_nonzero(
            mask
        )
    )

    if area_px2 == 0:

        return {
            "skeleton":
                np.zeros_like(
                    mask,
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

    skeleton = skeletonize(
        mask
    )

    length_px = (
        calculate_skeleton_length(
            skeleton
        )
    )

    (
        mean_width_px,
        max_width_px,
    ) = calculate_width(
        mask,
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
# Merge YOLO instances
# ============================================================

def merge_instance_masks(
    instance_masks,
):

    if not instance_masks:

        return None

    merged = np.zeros_like(
        instance_masks[0],
        dtype=bool,
    )

    for mask in instance_masks:

        merged = np.logical_or(
            merged,
            mask,
        )

    return merged


# ============================================================
# Connected components
# ============================================================

def extract_physical_cracks(
    merged_mask,
):
    """
    Split merged mask into physical crack regions
    using 8-connectivity.
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

    components = []

    # label 0 = background
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

        component_mask = (
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

        components.append(
            {
                "mask":
                    component_mask,

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

    # Largest physical crack first
    components.sort(
        key=lambda item:
        item[
            "component_area"
        ],
        reverse=True,
    )

    return components


# ============================================================
# Find supporting YOLO instances
# ============================================================

def find_supporting_instances(
    physical_mask,
    instance_masks,
    confidences,
):
    """
    A YOLO instance supports a physical crack
    if its mask overlaps that connected component.
    """

    supporting_indices = []

    supporting_confidences = []

    for index, instance_mask in enumerate(
        instance_masks
    ):

        overlap = np.logical_and(
            physical_mask,
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
# Visualization
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
# Draw side information panel
# ============================================================

def create_visualization_canvas(
    image,
):

    height, width = image.shape[:2]

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


def draw_panel_information(
    canvas,
    original_width,
    measurements,
    raw_instance_count,
):

    font = cv2.FONT_HERSHEY_SIMPLEX

    x = original_width + 15

    y = 28

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

    cv2.putText(
        canvas,
        (
            f"Physical cracks: "
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

    for item in measurements:

        color = item[
            "color"
        ]

        lines = [
            (
                f"Crack #{item['physical_crack_id']}"
            ),
            (
                f"Conf: {item['max_confidence']:.2f}"
            ),
            (
                f"Instances: "
                f"{item['supporting_instances']}"
            ),
            (
                f"L: {item['length_px']:.1f} px"
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
                + 5,
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
                0.43,
                (
                    255,
                    255,
                    255,
                ),
                1,
                cv2.LINE_AA,
            )

            y += 18

        y += 14


# ============================================================
# Process image
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

        return (
            [],
            image,
            0,
        )

    result = results[0]

    if (
        result.masks is None
        or
        result.boxes is None
        or
        len(result.boxes) == 0
    ):

        return (
            [],
            image,
            0,
        )

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
    # Resize all instance masks
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
    # Merge all predicted crack masks
    # --------------------------------------------------------

    merged_mask = (
        merge_instance_masks(
            instance_masks
        )
    )

    if merged_mask is None:

        return (
            [],
            image,
            raw_instance_count,
        )

    # --------------------------------------------------------
    # Connected physical cracks
    # --------------------------------------------------------

    components = (
        extract_physical_cracks(
            merged_mask
        )
    )

    visualization = image.copy()

    measurements = []

    # --------------------------------------------------------
    # Measure each physical crack
    # --------------------------------------------------------

    for index, component in enumerate(
        components,
        start=1,
    ):

        physical_mask = (
            component[
                "mask"
            ]
        )

        geometry = (
            measure_physical_crack(
                physical_mask
            )
        )

        (
            supporting_indices,
            max_confidence,
        ) = find_supporting_instances(
            physical_mask,
            instance_masks,
            valid_confidences,
        )

        color = COLORS[
            (index - 1)
            % len(COLORS)
        ]

        overlay_mask(
            visualization,
            physical_mask,
            color,
            alpha=0.35,
        )

        skeleton = geometry[
            "skeleton"
        ]

        visualization[
            skeleton
        ] = (
            255,
            255,
            255,
        )

        (
            x1,
            y1,
            x2,
            y2,
        ) = component[
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

        measurements.append(
            {
                "image":
                    image_path.name,

                "physical_crack_id":
                    index,

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
    # Side panel
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
# CSV
# ============================================================

def save_csv(
    all_measurements,
):

    fieldnames = [
        "image",
        "physical_crack_id",
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

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        for item in all_measurements:

            writer.writerow(
                {
                    "image":
                        item[
                            "image"
                        ],

                    "physical_crack_id":
                        item[
                            "physical_crack_id"
                        ],

                    "max_confidence":
                        f"{item['max_confidence']:.6f}",

                    "supporting_instances":
                        item[
                            "supporting_instances"
                        ],

                    "length_px":
                        f"{item['length_px']:.4f}",

                    "mean_width_px":
                        f"{item['mean_width_px']:.4f}",

                    "max_width_px":
                        f"{item['max_width_px']:.4f}",

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

    images = collect_images()

    print("=" * 80)
    print(
        "Merged physical crack quantification"
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

    total_physical_cracks = 0

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

        total_physical_cracks += (
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
            f"  Physical cracks: "
            f"{len(measurements)}"
        )

        for item in measurements:

            print(
                "  "
                f"Physical Crack "
                f"#{item['physical_crack_id']} | "
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
                    + "_merged_quantified.jpg"
                )
            )

            cv2.imwrite(
                str(
                    output_path
                ),
                visualization,
            )

            print(
                f"  Saved: "
                f"{output_path.name}"
            )

        print()

    # ========================================================
    # CSV
    # ========================================================

    save_csv(
        all_measurements
    )

    # ========================================================
    # Summary
    # ========================================================

    print("=" * 80)
    print(
        "MERGED QUANTIFICATION COMPLETED"
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
        f"Physical cracks after merge: "
        f"{total_physical_cracks}"
    )

    print()

    print(
        "CSV:"
    )

    print(
        CSV_PATH
    )

    print()

    print(
        "Visualizations:"
    )

    print(
        VIS_DIR
    )

    print()

    print(
        "Pipeline:"
    )

    print(
        "YOLO instances"
    )

    print(
        "  -> mask union"
    )

    print(
        "  -> 8-connected components"
    )

    print(
        "  -> physical cracks"
    )

    print(
        "  -> skeleton"
    )

    print(
        "  -> length / width / area"
    )

    print()

    print(
        "Units:"
    )

    print(
        "  Length = px"
    )

    print(
        "  Width  = px"
    )

    print(
        "  Area   = px^2"
    )

    print()

    print(
        "No physical calibration is applied."
    )


if __name__ == "__main__":
    main()