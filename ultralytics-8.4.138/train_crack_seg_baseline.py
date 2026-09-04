from ultralytics import YOLO


def main():
    model = YOLO("yolov8n-seg.pt")

    model.train(
        data="D:/crack-seg/crack-seg-local.yaml",
        epochs=100,
        imgsz=640,
        batch=16,
        device=0,
        workers=4,

        project="D:/CrackSegRuns",
        name="yolov8n_seg_baseline",

        pretrained=True,
        amp=True,
        cache=False,
        plots=True,
        save=True,
        val=True,
    )


if __name__ == "__main__":
    main()