from ultralytics import YOLO
import ultralytics.nn.modules as um
from modules.ghost_modules import GhostConv, C3Ghost

um.GhostConv = GhostConv
um.C3Ghost = C3Ghost


def main():
    model_yaml = "configs/ghost_yolo_wbc.yaml"
    data_yaml = "data/wbccd.yaml"

    model = YOLO(model_yaml)

    results = model.train(
        data=data_yaml,
        epochs=500,
        imgsz=800,
        batch=16,
        device=0,
        workers=8,
        optimizer="AdamW",
        lr0=0.0005,
        lrf=0.01,
        weight_decay=0.0005,
        mosaic=0.5,
        mixup=0.0,
        copy_paste=0.2,
        box=7.5,
        cls=0.5,
        dfl=1.5,
        amp=True,
        project="runs/wbc",
        name="ghost_model",
        save=True,
        plots=True,
    )

    best_model = YOLO("runs/wbc/ghost_model/weights/best.pt")

    metrics = best_model.val(
        data=data_yaml,
        imgsz=800,
        batch=16,
        device=0,
    )

    print(metrics)


if __name__ == "__main__":
    main()
