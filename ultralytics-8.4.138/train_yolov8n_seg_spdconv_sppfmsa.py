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
        / "yolov8n-seg-spdconv-sppfmsa.yaml"
    )

    weight_path = root / "yolov8n-seg.pt"

    # 仍然从官方 YOLOv8n-seg 预训练权重开始
    # 不能从 SPDConv 30e 的 best.pt 开始，否则消融不公平
    model = YOLO(str(model_cfg)).load(str(weight_path))

    model.train(
        data="D:/crack-seg/crack-seg-local.yaml",

        epochs=30,
        imgsz=640,
        batch=16,
        device=0,
        workers=4,

        project="D:/CrackSegRuns",
        name="yolov8n_seg_spdconv_sppfmsa_30e",

        pretrained=True,
        amp=True,
        cache=False,
        plots=True,
        save=True,
        val=True,

        seed=0,
        deterministic=True,
    )


if __name__ == "__main__":
    main()