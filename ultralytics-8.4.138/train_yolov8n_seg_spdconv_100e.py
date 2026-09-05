from pathlib import Path

from ultralytics import YOLO


def main():
    root = Path(__file__).resolve().parent

    # 只包含 SPDConv 的 YOLOv8n-seg 结构
    model_cfg = (
        root
        / "ultralytics"
        / "cfg"
        / "models"
        / "v8"
        / "yolov8n-seg-spdconv.yaml"
    )

    # 官方 YOLOv8n-seg 预训练权重
    weight_path = root / "yolov8n-seg.pt"

    # 从 SPDConv YAML 创建模型，
    # 再加载能够匹配上的官方 YOLOv8n-seg 预训练权重
    model = YOLO(str(model_cfg)).load(str(weight_path))

    model.train(
        data="D:/crack-seg/crack-seg-local.yaml",

        epochs=100,
        imgsz=640,
        batch=16,
        device=0,
        workers=4,

        project="D:/CrackSegRuns",
        name="yolov8n_seg_spdconv_100e",

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