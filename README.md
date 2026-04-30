# Fruit & Vegetable Freshness Detection

Real-time system for detecting and classifying fresh vs rotten fruits and vegetables 
using a two-stage deep learning pipeline. Built as a Minor Project at Thapathali Campus, 
IOE, Tribhuvan University (2026).

---

## Results

| Model | Metric | Score |
|---|---|---|
| YOLOv8 | mAP@50 | 92.3% |
| YOLOv8 | mAP@50-95 | 89.5% |
| YOLOv8 | Precision | 90.8% |
| YOLOv8 | Recall | 87.5% |
| MobileNetV3 | Classification Accuracy | ~92% |

---

## How It Works

Manual inspection of fruits and vegetables is slow, inconsistent, and contributes 
to food waste. This system automates that process in real time.

Two models work together in a single pipeline:
Input (Image / Video / Webcam)
↓
YOLOv8 - detects and localizes each fruit/vegetable
↓
MobileNetV3 - classifies each detected object as Fresh or Rotten
↓
Output with bounding boxes + freshness labels

- **YOLOv8** handles object detection — finds where the fruits are in the frame
- **MobileNetV3** handles classification — evaluates color, texture, and surface 
  patterns to determine freshness
- Both run simultaneously, enabling real-time multi-object assessment in a single frame

---

## Project Structure

```
Freshness-Detection-Computer_Vision/
│
├── models/              # Trained YOLOv8 and MobileNetV3 model weights
├── utils/               # Helper functions for preprocessing and inference
├── Testing/             # Test images and validation scripts
├── output/              # Sample output results
├── code.ipynb           # Main training and experimentation notebook
├── train_demo1.ipynb    # Model training walkthrough
├── final_pipeline.py    # End-to-end inference pipeline
└── testing_model_cnn.py # CNN model testing script
```
---

## Tech Stack

- **Python**
- **YOLOv8** (Ultralytics) - object detection
- **MobileNetV3** - image classification via transfer learning
- **PyTorch** - model training
- **OpenCV** - real-time video processing
- **Django + React** - web interface

---

## Setup & Run

```bash
# Clone the repo
git clone https://github.com/aadityaKS1/Freshness-Detection-Computer_Vision-
cd Freshness-Detection-Computer_Vision-

# Install dependencies
pip install -r requirements.txt

# Run the pipeline
python final_pipeline.py
```

---

## Applications

- Agriculture - quality control at harvest
- Retail & supermarkets - automated shelf inspection  
- Supply chain - freshness monitoring during transit

---

## Team

- Aaditya Kumar Singh (THA079BCT001)
- Anish Kumar Singh (THA079BCT004)
- Jeewan Bhatt (THA079BCT015)
- Ruby Kumari Sah (THA079BCT034)

Supervised by **Er. Shanta Maharjan**, Dept. of Electronics & Computer Engineering, 
Thapathali Campus, IOE

---

## Also add these to the repo (if not done yet)

- Add `requirements.txt` — list all pip packages used
- Add a screenshot or demo GIF in the `output/` folder and reference it here
- Add topics on GitHub: `computer-vision`, `yolov8`, `object-detection`, 
  `deep-learning`, `python`, `mobilenetv3`
