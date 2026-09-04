from pathlib import Path

from ultralytics import YOLO


def main():
    # 项目根目录
    root = Path(__file__).resolve().parent

    # SPDConv 改进模型配置
    model_cfg = (
        root
        / "ultralytics"
        / "cfg"
        / "models"
        / "v8"
        / "yolov8n-seg-spdconv.yaml"
    )

    # YOLOv8n-seg 官方预训练权重
    weight_path = root / "yolov8n-seg.pt"

    # 按 SPDConv YAML 创建模型，并尽可能加载原 YOLOv8n-seg 预训练参数
    model = YOLO(str(model_cfg)).load(str(weight_path))

    # 训练
    model.train(
        data="D:/crack-seg/crack-seg-local.yaml",

        # 与 baseline 保持一致
        epochs=30,
        imgsz=640,
        batch=16,
        device=0,
        workers=4,

        # 实验结果保存位置
        project="D:/CrackSegRuns",
        name="yolov8n_seg_spdconv_30e",

        # 训练设置  
        pretrained=True,
        amp=True,
        cache=False,
        plots=True,
        save=True,
        val=True,

        # 固定随机性，方便和 baseline 对比
        seed=0,
        deterministic=True,
    )


if __name__ == "__main__":
    main()