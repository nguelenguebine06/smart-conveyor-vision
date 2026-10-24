from pathlib import Path
import cv2

image_path = Path("dataset/images/train/img_0101.png")
label_path = Path("dataset/labels/train/img_0101.txt")

image = cv2.imread(str(image_path))

if image is None:
    raise FileNotFoundError(f"Image introuvable : {image_path}")

height, width = image.shape[:2]

with open(label_path, "r") as file:
    for line in file:
        values = line.split()

        if len(values) != 5:
            continue

        class_id, xc, yc, bw, bh = map(float, values)

        x1 = int((xc - bw / 2) * width)
        y1 = int((yc - bh/2) * height)
        x2 = int((xc + bw / 2) * width)
        y2 = int((yc + bh/2) * height)

        print("Dimensions image :", width, height)
        print("Boîte calculée :", x1, y1, x2, y2)
        print("Annotation YOLO :", xc, yc, bw, bh)

        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            image,
            "trou",
            (x1, max(y1 - 10, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )

cv2.imshow("Verification annotation YOLO", image)
cv2.waitKey(0)
cv2.destroyAllWindows()