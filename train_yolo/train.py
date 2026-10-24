from ultralytics import YOLO

# Charger le modèle pré-entraîné

model = YOLO("yolo26n.pt")

# Lancer l'entraînement

model.train(
    data="dataset/data.yaml",
    epochs=50,
    imgsz=640,
    batch=4,
    device="cpu",
    workers=0,
    project="runs",
    name="detection_trous"
)