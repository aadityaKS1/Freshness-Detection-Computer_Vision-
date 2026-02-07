import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v3 import preprocess_input

class FreshnessClassifier:
    def __init__(self, model_path, img_size=224):
        self.model = tf.keras.models.load_model(model_path, compile=False)
        self.img_size = img_size

    def predict(self, crop):
        if crop is None or crop.size == 0:
            return "Unknown", 0.0

        # Resize first
        crop = cv2.resize(crop, (self.img_size, self.img_size))

        # BGR → RGB
        crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)

        # Add batch dimension
        crop = np.expand_dims(crop, axis=0)

        # ✅ MobileNetV3 preprocessing (CRITICAL)
        crop = preprocess_input(crop)

        # Predict
        pred = self.model.predict(crop, verbose=0)[0][0]

        # ✅ Correct label mapping (same as your working script)
        label = "Rotten" if pred >= 0.5 else "Fresh"

        return label, float(pred)
