import os
import cv2
from utils.detector import YOLODetector
from utils.classifier import FreshnessClassifier
from utils.visualizer import draw_result

# ----------------------------
# PATHS
# ----------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

YOLO_MODEL_PATH = os.path.join(BASE_DIR, "models", "best.pt")
CLASSIFIER_MODEL_PATH = os.path.join(BASE_DIR, "models", "testingmodel3.keras")
TEST_IMAGE_PATH = os.path.join(BASE_DIR, "Testing", "22.png")
# ----------------------------
# CONFIG
# ----------------------------
YOLO_CLASSES = ["apple", "banana", "carrot", "tomato"]

# ----------------------------
# LOAD MODELS
# ----------------------------
detector = YOLODetector(YOLO_MODEL_PATH, YOLO_CLASSES)
classifier = FreshnessClassifier(CLASSIFIER_MODEL_PATH)

# ----------------------------
# LOAD IMAGE
# ----------------------------
img = cv2.imread(TEST_IMAGE_PATH)
if img is None:
    raise FileNotFoundError("Test image not found")

# ----------------------------
# DETECT + CLASSIFY
# ----------------------------
detections = detector.detect(img)

for det in detections:
    x1, y1, x2, y2 = det["bbox"]

    crop = img[y1:y2, x1:x2]
    if crop.size == 0:
    
        continue

    label, score = classifier.predict(crop)
    draw_result(img, det["bbox"], det["label"], f"{label}/{score:.2f}")

# ----------------------------
# SAVE RESULT AS result_X.jpg
# ----------------------------
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# find next result index
existing = [f for f in os.listdir(OUTPUT_DIR) if f.startswith("result_")]
idx = len(existing) + 1

OUTPUT_PATH = os.path.join(OUTPUT_DIR, f"result_{idx}.jpg")

cv2.imwrite(OUTPUT_PATH, img)

# ----------------------------
# SIMPLE RESIZE FOR DISPLAY ONLY
# ----------------------------
DISPLAY_INCHES = 10
DPI = 96  # standard screen DPI
TARGET_SIZE = DISPLAY_INCHES * DPI  # 960 pixels

h, w = img.shape[:2]

# Scale based on the larger dimension
scale = TARGET_SIZE / max(w, h)

display = cv2.resize(img, None, fx=scale, fy=scale)

cv2.imshow("Final Result", display)
cv2.waitKey(0)
cv2.destroyAllWindows()
0