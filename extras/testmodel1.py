import os
import random
import shutil

SRC = "train"
CLASSES = ["Apple", "Banana", "Carrot", "Tomato"]

IMG_TRAIN = "yolo_dataset/images/train"
IMG_VAL   = "yolo_dataset/images/val"
LBL_TRAIN = "yolo_dataset/labels/train"
LBL_VAL   = "yolo_dataset/labels/val"

VAL_RATIO = 0.2

os.makedirs(IMG_TRAIN, exist_ok=True)
os.makedirs(IMG_VAL, exist_ok=True)
os.makedirs(LBL_TRAIN, exist_ok=True)
os.makedirs(LBL_VAL, exist_ok=True)

pairs = []

# collect image–label pairs
for cls in CLASSES:
    img_dir = os.path.join(SRC, cls, "images")
    lbl_dir = os.path.join(SRC, cls, "labels")

    print(f"📂 Reading {img_dir}")

    if not os.path.exists(img_dir):
        print("❌ images folder missing")
        continue

    for img in os.listdir(img_dir):
        if img.lower().endswith((".jpg", ".jpeg", ".png")):
            name = os.path.splitext(img)[0]
            lbl = name + ".txt"

            img_path = os.path.join(img_dir, img)
            lbl_path = os.path.join(lbl_dir, lbl)

            if os.path.exists(lbl_path):
                pairs.append((img_path, lbl_path))

print(f"🔎 Total pairs found: {len(pairs)}")

# shuffle & split
random.shuffle(pairs)
val_size = int(len(pairs) * VAL_RATIO)

val_pairs = pairs[:val_size]
train_pairs = pairs[val_size:]

def move(pairs, img_dst, lbl_dst):
    for img, lbl in pairs:
        shutil.copy(img, os.path.join(img_dst, os.path.basename(img)))
        shutil.copy(lbl, os.path.join(lbl_dst, os.path.basename(lbl)))

move(train_pairs, IMG_TRAIN, LBL_TRAIN)
move(val_pairs, IMG_VAL, LBL_VAL)

print("\n✅ DONE")
print(f"Train images: {len(train_pairs)}")
print(f"Val images  : {len(val_pairs)}")
