import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v3 import preprocess_input

# -----------------------------
# 1) Load your saved model
# -----------------------------
model = tf.keras.models.load_model("testingmodel2.keras",compile=False)

# -----------------------------
# 2) Load image using OpenCV
# -----------------------------
img_path = r"C:/Users/NINJA/Desktop/Testing/test13.png"
img = cv2.imread(img_path)

if img is None:
    print("Image not found! Check path.")
    exit()

# -----------------------------
# 3) Preprocess image
# -----------------------------
img = cv2.resize(img, (224, 224))              # Resize
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)     # Convert BGR to RGB

img_array = np.expand_dims(img, axis=0)        # Shape: (1,224,224,3)
img_array = preprocess_input(img_array)        # MobileNetV3 preprocessing

# -----------------------------
# 4) Predict
# -----------------------------
pred = model.predict(img_array)[0][0]

# -----------------------------
# 5) Show result
# -----------------------------
if pred >= 0.5:
    label = "Rotten"
else:
    label = "Fresh"

print("Prediction:", label, "Confidence:", pred)

# -----------------------------
# 6) Display image with label
# -----------------------------
display_img = cv2.imread(img_path)
cv2.putText(display_img, f"{label} ({pred:.2f})", (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

cv2.imshow("Prediction", display_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
