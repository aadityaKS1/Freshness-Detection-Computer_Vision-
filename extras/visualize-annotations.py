import os
import cv2
import random

# Change if needed
IMG_DIR = "yolo_dataset/images/train"
LBL_DIR = "yolo_dataset/labels/train"

CLASS_NAMES = {
    0: "Apple",
    1: "Banana",
    2: "Carrot",
    3: "Tomato"
}

# Pick a random image
images = [f for f in os.listdir(IMG_DIR) if f.endswith((".jpg", ".png", ".jpeg"))]
img_name = random.choice(images)

img_path = os.path.join(IMG_DIR, img_name)
lbl_path = os.path.join(LBL_DIR, os.path.splitext(img_name)[0] + ".txt")

print("Visualizing:", img_name)

img = cv2.imread(img_path)
h, w, _ = img.shape

if os.path.exists(lbl_path):
    with open(lbl_path) as f:
        for line in f:
            cls, xc, yc, bw, bh = map(float, line.split())

            x1 = int((xc - bw/2) * w)
            y1 = int((yc - bh/2) * h)
            x2 = int((xc + bw/2) * w)
            y2 = int((yc + bh/2) * h)

            cls = int(cls)
            label = CLASS_NAMES.get(cls, str(cls))

            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img, label, (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

else:
    print("❌ Label file not found")

cv2.imshow("YOLO Annotation Check", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
