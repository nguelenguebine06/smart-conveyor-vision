from ultralytics import YOLO

# Charger le meilleur modèle entraîné
model = YOLO(
    "runs/detect/runs/detection_trous/weights/best.pt"
)

# Tester sur une image du dataset
results = model.predict(
    source="dataset/images/val/img_0310.png",
    conf=0.85,
    save=True
)

print("Détection terminée.")