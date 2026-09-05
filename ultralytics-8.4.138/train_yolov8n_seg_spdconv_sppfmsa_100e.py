from pathlib import Path

from ultralytics import YOLO


def main():
    root = Path(__file__).resolve().parent

    # SPDConv + SPPF-MSA 模型结构
    model_cfg = (
        root
        / "ultralytics"
        / "cfg"
        / "models"
        / "v8"
        / "yolov8n-seg-spdconv-sppfmsa.yaml"
    )

    # 官方 YOLOv8n-seg 预训练权重
    weight_path = root / "yolov8n-seg.pt"

    # 从模型 YAML 创建新模型，
    # 再加载官方 YOLOv8n-seg 可匹配的预训练权重
    model = YOLO(str(model_cfg)).load(str(weight_path))

    model.train(
        data="D:/crack-seg/crack-seg-local.yaml",

        # 与 baseline 正式训练保持一致
        epochs=100,
        imgsz=640,
        batch=16,
        device=0,
        workers=4,

        # 新建独立实验目录，绝不覆盖之前结果
        project="D:/CrackSegRuns",
        name="yolov8n_seg_spdconv_sppfmsa_100e",

        pretrained=True,
        amp=True,
        cache=False,

        plots=True,
        save=True,
        val=True,

        # 保持实验可重复性
        seed=0,
        deterministic=True,
    )


if __name__ == "__main__":
    main()