from ultralytics import YOLO

class YOLODetector:
    def __init__(self, model_path, class_names):
        self.model = YOLO(model_path)
        self.class_names = class_names

    def detect(self, image):
        results = self.model(image)[0]
        detections = []

        for box in results.boxes:
            cls_id = int(box.cls[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            detections.append({
                "label": self.class_names[cls_id],
                "bbox": (x1, y1, x2, y2)
            })

        return detections
