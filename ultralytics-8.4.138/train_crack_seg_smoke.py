from ultralytics import YOLO


def main():
    # 1. 加载 YOLOv8n-seg 预训练模型
    model = YOLO("yolov8n-seg.pt")

    # 2. 开始训练
    model.train(
        data="D:/crack-seg/crack-seg-local.yaml",

        # 基础训练参数
        epochs=3,          # 先做冒烟测试，确认流程能跑通
        imgsz=640,
        batch=8,
        device=0,
        workers=4,

        # 输出位置
        project="D:/CrackSegRuns",
        name="baseline_smoke",

        # 训练设置
        pretrained=True,
        amp=True,
        cache=False,
        plots=True,
        save=True,
        val=True,
    )


if __name__ == "__main__":
    main()