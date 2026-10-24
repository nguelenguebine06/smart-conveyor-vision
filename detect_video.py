
import cv2
from functools import lru_cache
from pathlib import Path
from ultralytics import YOLO


MODEL_PATH = Path(__file__).parent / (
    "runs/detect/runs/detection_trous/weights/best.pt"
)


@lru_cache(maxsize=1)
def charger_modele():
    """Charge le modèle une seule fois."""
    return YOLO(str(MODEL_PATH))


def detecter_video(video_path, output_path, callback=None, conf=0.25):
    """
    Analyse une vidéo, enregistre les images annotées
    et transmet chaque image à l'interface si un callback existe.
    """

    model = charger_modele()
    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        raise ValueError("Impossible d'ouvrir la vidéo.")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if fps <= 0:
        fps = 25.0

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(
        str(output_path),
        fourcc,
        fps,
        (width, height)
    )

    if not out.isOpened():
        cap.release()
        raise ValueError("Impossible de créer la vidéo de sortie.")

    images_traitees = 0
    total_detections = 0

    try:
        while True:
            ret, frame = cap.read()

            if not ret:
                break

            result = model.predict(
                source=frame,
                conf=conf,
                verbose=False
            )[0]

            annotated_frame = result.plot()

            if len(result.boxes) > 0:
                message = "ANOMALIE DETECTEE"
                couleur = (0, 0, 255)
                total_detections += len(result.boxes)
            else:
                message = "CONVOYEUR NORMAL"
                couleur = (0, 255, 0)

            cv2.putText(
                annotated_frame,
                message,
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                couleur,
                2
            )

            out.write(annotated_frame)
            images_traitees += 1

            # Transmettre l'image à Streamlit
            if callback is not None:
                callback(
                    annotated_frame,
                    images_traitees,
                    total_frames
                )

    finally:
        cap.release()
        out.release()

    return {
        "images_traitees": images_traitees,
        "total_frames": total_frames,
        "total_detections": total_detections,
        "fps": fps,
        "width": width,
        "height": height
    }


if __name__ == "__main__":
    resultat = detecter_video(
        "convoyeur_en_mouvement.mp4",
        "convoyeur_detecte.mp4",
        callback=lambda image, traitees, total: (
            cv2.imshow("Detection des trous", image)
            if cv2.waitKey(1) != -1 else None
        )
    )

    cv2.destroyAllWindows()
    print("Analyse terminée :", resultat)
