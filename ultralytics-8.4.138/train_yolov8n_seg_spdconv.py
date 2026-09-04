from pathlib import Path

from ultralytics import YOLO


def main():
    root = Path(__file__).resolve().parent
    model_cfg = (
        root
        / "ultralytics"
        / "cfg"
        / "models"
        / "v8"
        / "yolov8n-seg-spdconv.yaml"
    )

    model = YOLO(str(model_cfg)).load("yolov8n-seg.pt")

    model.train(
        data="D:/crack-seg/crack-seg-local.yaml",
        epochs=100,
        imgsz=640,
        batch=16,
        device=0,
        workers=4,
        project="D:/CrackSegRuns",
        name="yolov8n_seg_spdconv",
        pretrained=True,
        amp=True,
        cache=False,
        plots=True,
        save=True,
        val=True,
    )


if __name__ == "__main__":
    main()