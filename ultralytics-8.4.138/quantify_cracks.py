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


# ============================================================
# Selected test images
# ============================================================

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


# ============================================================
# Output
# ============================================================

OUTPUT_ROOT = Path(
    "D:/CrackSegRuns/crack_quantification_selected5"
)

VIS_DIR = (
    OUTPUT_ROOT
    / "visualizations"
)

CSV_PATH = (
    OUTPUT_ROOT
    / "crack_measurements.csv"
)


# ============================================================
# Inference parameters
# ============================================================

IMGSZ = 640

CONF = 0.25

DEVICE = 0


IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tif",
    ".tiff",
}


# ============================================================
# Visualization colors
# OpenCV uses BGR
# ============================================================

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
# Load model
# ============================================================

def load_model():

    if not MODEL_PATH.is_file():
        raise FileNotFoundError(
            f"Model not found:\n{MODEL_PATH}"
        )

    print("=" * 80)
    print("Loading final KD Student")
    print("=" * 80)

    print(
        f"Model: {MODEL_PATH}"
    )

    model = YOLO(
        str(MODEL_PATH),
        task="segment",
    )

    # Some locally saved checkpoints may lose
    # task metadata although the model is segmentation.
    if getattr(
        model.model,
        "task",
        None,
    ) is None:

        print(
            "NOTE: model.task metadata is missing; "
            "setting model.task='segment'."
        )

        model.model.task = (
            "segment"
        )

    print(
        "Model loaded successfully."
    )

    print()

    return model


# ============================================================
# Validate selected input images
# ============================================================

def collect_images():

    images = []

    for path in INPUT_PATHS:

        if not path.exists():

            print(
                f"WARNING: image does not exist:\n"
                f"{path}"
            )

            continue

        if not path.is_file():

            print(
                f"WARNING: not a file:\n"
                f"{path}"
            )

            continue

        if (
            path.suffix.lower()
            not in IMAGE_EXTENSIONS
        ):

            print(
                f"WARNING: unsupported image type:\n"
                f"{path}"
            )

            continue

        images.append(
            path
        )

    if not images:

        raise RuntimeError(
            "No valid input images found."
        )

    return images


# ============================================================
# Resize mask to original resolution
# ============================================================

def resize_mask_to_original(
    mask,
    original_height,
    original_width,
):

    if (
        mask.shape[0]
        == original_height
        and
        mask.shape[1]
        == original_width
    ):
        return mask.astype(bool)

    resized = cv2.resize(
        mask.astype(
            np.uint8
        ),
        (
            original_width,
            original_height,
        ),
        interpolation=(
            cv2.INTER_NEAREST
        ),
    )

    return resized.astype(bool)


# ============================================================
# Skeleton length
# ============================================================

def calculate_skeleton_length(
    skeleton,
):
    """
    Calculate total length of the entire
    8-connected skeleton graph.

    Horizontal connection = 1 pixel
    Vertical connection   = 1 pixel
    Diagonal connection   = sqrt(2) pixels

    Each edge is counted once.
    """

    sk = skeleton.astype(
        bool
    )

    if np.count_nonzero(sk) <= 1:
        return 0.0

    # Horizontal:
    #
    # x x
    horizontal = np.logical_and(
        sk[:, :-1],
        sk[:, 1:],
    )

    # Vertical:
    #
    # x
    # x
    vertical = np.logical_and(
        sk[:-1, :],
        sk[1:, :],
    )

    # Main diagonal:
    #
    # x .
    # . x
    diagonal_1 = np.logical_and(
        sk[:-1, :-1],
        sk[1:, 1:],
    )

    # Anti-diagonal:
    #
    # . x
    # x .
    diagonal_2 = np.logical_and(
        sk[:-1, 1:],
        sk[1:, :-1],
    )

    horizontal_count = (
        np.count_nonzero(
            horizontal
        )
    )

    vertical_count = (
        np.count_nonzero(
            vertical
        )
    )

    diagonal_count = (
        np.count_nonzero(
            diagonal_1
        )
        +
        np.count_nonzero(
            diagonal_2
        )
    )

    length = (
        horizontal_count
        +
        vertical_count
        +
        math.sqrt(2.0)
        * diagonal_count
    )

    return float(length)


# ============================================================
# Crack width
# ============================================================

def calculate_width(
    binary_mask,
    skeleton,
):
    """
    Estimate local crack width using Euclidean
    distance transform.

    For a skeleton point:

        width ~= 2 * distance_to_background

    Returns:
        mean_width_px
        max_width_px
    """

    if (
        np.count_nonzero(
            skeleton
        )
        == 0
    ):
        return 0.0, 0.0

    mask_uint8 = (
        binary_mask.astype(
            np.uint8
        )
        * 255
    )

    distance_map = (
        cv2.distanceTransform(
            mask_uint8,
            cv2.DIST_L2,
            5,
        )
    )

    distances = (
        distance_map[
            skeleton
        ]
    )

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
# Measure one crack instance
# ============================================================

def measure_crack(
    binary_mask,
):

    binary_mask = (
        binary_mask.astype(
            bool
        )
    )

    # --------------------------------------------------------
    # Area
    # --------------------------------------------------------

    area_px2 = int(
        np.count_nonzero(
            binary_mask
        )
    )

    if area_px2 == 0:

        empty_skeleton = (
            np.zeros_like(
                binary_mask,
                dtype=bool,
            )
        )

        return {
            "skeleton":
                empty_skeleton,

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
        binary_mask
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
        binary_mask,
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
# Bounding box from mask
# ============================================================

def get_mask_bbox(
    mask,
):

    ys, xs = np.nonzero(
        mask
    )

    if len(xs) == 0:
        return None

    x1 = int(
        xs.min()
    )

    y1 = int(
        ys.min()
    )

    x2 = int(
        xs.max()
    )

    y2 = int(
        ys.max()
    )

    return (
        x1,
        y1,
        x2,
        y2,
    )


# ============================================================
# Transparent mask overlay
# ============================================================

def overlay_mask(
    image,
    mask,
    color,
    alpha=0.35,
):

    overlay = (
        image.copy()
    )

    overlay[
        mask
    ] = color

    blended = (
        cv2.addWeighted(
            overlay,
            alpha,
            image,
            1.0 - alpha,
            0,
        )
    )

    image[:] = (
        blended
    )


# ============================================================
# Draw measurement text
# ============================================================

def draw_text_block(
    image,
    lines,
    x,
    y,
    color,
):

    font = (
        cv2.FONT_HERSHEY_SIMPLEX
    )

    font_scale = 0.46

    thickness = 1

    line_height = 18

    height, width = (
        image.shape[:2]
    )

    max_text_width = 0

    for line in lines:

        text_size, _ = (
            cv2.getTextSize(
                line,
                font,
                font_scale,
                thickness,
            )
        )

        max_text_width = max(
            max_text_width,
            text_size[0],
        )

    block_height = (
        line_height
        * len(lines)
        + 8
    )

    # Keep box inside image
    x = max(
        2,
        min(
            x,
            width
            - max_text_width
            - 12,
        ),
    )

    y = max(
        block_height + 2,
        min(
            y,
            height - 2,
        ),
    )

    top = (
        y
        - block_height
    )

    right = (
        x
        + max_text_width
        + 10
    )

    bottom = y

    # Black background
    cv2.rectangle(
        image,
        (
            x,
            top,
        ),
        (
            right,
            bottom,
        ),
        (
            0,
            0,
            0,
        ),
        -1,
    )

    # Colored border
    cv2.rectangle(
        image,
        (
            x,
            top,
        ),
        (
            right,
            bottom,
        ),
        color,
        1,
    )

    current_y = (
        top
        + line_height
    )

    for line in lines:

        cv2.putText(
            image,
            line,
            (
                x + 5,
                current_y,
            ),
            font,
            font_scale,
            (
                255,
                255,
                255,
            ),
            thickness,
            cv2.LINE_AA,
        )

        current_y += (
            line_height
        )


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

        print(
            f"WARNING: failed to read image:\n"
            f"{image_path}"
        )

        return [], None

    (
        original_height,
        original_width,
    ) = image.shape[:2]

    visualization = (
        image.copy()
    )

    # --------------------------------------------------------
    # Segmentation inference
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

        return (
            [],
            visualization,
        )

    result = (
        results[0]
    )

    # --------------------------------------------------------
    # No prediction
    # --------------------------------------------------------

    if (
        result.masks is None
        or
        result.boxes is None
        or
        len(result.boxes) == 0
    ):

        cv2.putText(
            visualization,
            "No crack detected",
            (
                20,
                35,
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (
                0,
                0,
                255,
            ),
            2,
            cv2.LINE_AA,
        )

        return (
            [],
            visualization,
        )

    # --------------------------------------------------------
    # Masks
    # --------------------------------------------------------

    masks = (
        result.masks.data
        .detach()
        .cpu()
        .numpy()
    )

    # --------------------------------------------------------
    # Confidence
    # --------------------------------------------------------

    confidences = (
        result.boxes.conf
        .detach()
        .cpu()
        .numpy()
    )

    mask_count = (
        masks.shape[0]
    )

    confidence_count = (
        len(confidences)
    )

    count = min(
        mask_count,
        confidence_count,
    )

    if (
        mask_count
        != confidence_count
    ):

        print(
            "WARNING: masks and confidence "
            "counts differ: "
            f"{mask_count} vs "
            f"{confidence_count}"
        )

    measurements = []

    # --------------------------------------------------------
    # Each predicted crack instance
    # --------------------------------------------------------

    for i in range(
        count
    ):

        crack_id = (
            i + 1
        )

        confidence = float(
            confidences[i]
        )

        # ----------------------------------------------------
        # Binary mask
        # ----------------------------------------------------

        binary_mask = (
            masks[i]
            > 0.5
        )

        binary_mask = (
            resize_mask_to_original(
                binary_mask,
                original_height,
                original_width,
            )
        )

        # ----------------------------------------------------
        # Geometry
        # ----------------------------------------------------

        measurement = (
            measure_crack(
                binary_mask
            )
        )

        skeleton = (
            measurement[
                "skeleton"
            ]
        )

        length_px = (
            measurement[
                "length_px"
            ]
        )

        mean_width_px = (
            measurement[
                "mean_width_px"
            ]
        )

        max_width_px = (
            measurement[
                "max_width_px"
            ]
        )

        area_px2 = (
            measurement[
                "area_px2"
            ]
        )

        # ----------------------------------------------------
        # Record
        # ----------------------------------------------------

        row = {
            "image":
                image_path.name,

            "crack_id":
                crack_id,

            "confidence":
                confidence,

            "length_px":
                length_px,

            "mean_width_px":
                mean_width_px,

            "max_width_px":
                max_width_px,

            "area_px2":
                area_px2,
        }

        measurements.append(
            row
        )

        # ----------------------------------------------------
        # Visualization
        # ----------------------------------------------------

        color = COLORS[
            i
            % len(COLORS)
        ]

        overlay_mask(
            visualization,
            binary_mask,
            color,
            alpha=0.35,
        )

        # White skeleton centerline
        visualization[
            skeleton
        ] = (
            255,
            255,
            255,
        )

        bbox = get_mask_bbox(
            binary_mask
        )

        if bbox is not None:

            (
                x1,
                y1,
                x2,
                y2,
            ) = bbox

            # Bounding rectangle
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

            lines = [
                (
                    f"Crack #{crack_id} "
                    f"Conf={confidence:.2f}"
                ),
                (
                    f"L="
                    f"{length_px:.1f}px"
                ),
                (
                    f"Wmean="
                    f"{mean_width_px:.1f}px"
                ),
                (
                    f"Wmax="
                    f"{max_width_px:.1f}px"
                ),
                (
                    f"Area="
                    f"{area_px2}px^2"
                ),
            ]

            if y1 > 100:

                text_y = (
                    y1
                )

            else:

                text_y = min(
                    original_height - 5,
                    y2 + 100,
                )

            draw_text_block(
                visualization,
                lines,
                x1,
                text_y,
                color,
            )

    return (
        measurements,
        visualization,
    )


# ============================================================
# Save CSV
# ============================================================

def save_csv(
    measurements,
):

    fieldnames = [
        "image",
        "crack_id",
        "confidence",
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

                    "crack_id":
                        item[
                            "crack_id"
                        ],

                    "confidence":
                        (
                            f"{item['confidence']:.6f}"
                        ),

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

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    model = load_model()

    # --------------------------------------------------------
    # Images
    # --------------------------------------------------------

    images = (
        collect_images()
    )

    print("=" * 80)
    print(
        "Crack geometry quantification"
    )
    print("=" * 80)

    print(
        f"Selected images: "
        f"{len(images)}"
    )

    print(
        f"Output: {OUTPUT_ROOT}"
    )

    print(
        f"imgsz:  {IMGSZ}"
    )

    print(
        f"conf:   {CONF}"
    )

    print()

    all_measurements = []

    detected_images = 0

    total_instances = 0

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
            ) = process_image(
                model,
                image_path,
            )

        except Exception as exc:

            print(
                f"  ERROR: {exc}"
            )

            continue

        if visualization is None:
            continue

        # ----------------------------------------------------
        # Detected cracks
        # ----------------------------------------------------

        if measurements:

            detected_images += 1

            total_instances += (
                len(
                    measurements
                )
            )

            all_measurements.extend(
                measurements
            )

            for item in measurements:

                print(
                    "  "
                    f"Crack #{item['crack_id']} | "
                    f"Conf={item['confidence']:.3f} | "
                    f"L={item['length_px']:.2f}px | "
                    f"Wmean="
                    f"{item['mean_width_px']:.2f}px | "
                    f"Wmax="
                    f"{item['max_width_px']:.2f}px | "
                    f"Area="
                    f"{item['area_px2']}px^2"
                )

        else:

            print(
                "  No crack detected."
            )

        # ----------------------------------------------------
        # Save visualization
        # ----------------------------------------------------

        output_name = (
            image_path.stem
            + "_quantified.jpg"
        )

        output_path = (
            VIS_DIR
            / output_name
        )

        success = cv2.imwrite(
            str(output_path),
            visualization,
        )

        if success:

            print(
                f"  Saved visualization: "
                f"{output_path.name}"
            )

        else:

            print(
                "  WARNING: failed to "
                "save visualization."
            )

        print()

    # ========================================================
    # Save combined CSV
    # ========================================================

    save_csv(
        all_measurements
    )

    # ========================================================
    # Final summary
    # ========================================================

    print("=" * 80)
    print(
        "QUANTIFICATION COMPLETED"
    )
    print("=" * 80)

    print(
        f"Processed images:       "
        f"{len(images)}"
    )

    print(
        f"Images with detections: "
        f"{detected_images}"
    )

    print(
        f"Crack instances:        "
        f"{total_instances}"
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
        "Measurement units:"
    )

    print(
        "  Length      = px"
    )

    print(
        "  Mean Width  = px"
    )

    print(
        "  Max Width   = px"
    )

    print(
        "  Area        = px^2"
    )

    print()

    print(
        "Current measurement method:"
    )

    print(
        "  Length:"
        " 8-connected skeleton graph"
    )

    print(
        "  Width:"
        " 2 x Euclidean distance transform"
        " on skeleton"
    )

    print(
        "  Area:"
        " foreground pixel count"
    )

    print()

    print(
        "NOTE:"
    )

    print(
        "No physical scale calibration "
        "has been applied."
    )

    print(
        "Therefore results are reported "
        "only in pixel-domain units."
    )


if __name__ == "__main__":
    main()