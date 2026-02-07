import os
import shutil

# OLD structure (source)
SRC_BASE = "yolo_dataset/train"
CLASSES = ["Apple", "Banana", "Carrot", "Tomato"]

# NEW structure (destination)
DST_IMG = "yolo_dataset/images/train"
DST_LBL = "yolo_dataset/labels/train"

os.makedirs(DST_IMG, exist_ok=True)
os.makedirs(DST_LBL, exist_ok=True)

img_count = 0
lbl_count = 0

for cls in CLASSES:
    cls_dir = os.path.join(SRC_BASE, cls)

    if not os.path.exists(cls_dir):
        print(f"⚠️ Folder not found: {cls_dir}")
        continue

    for file in os.listdir(cls_dir):
        src_path = os.path.join(cls_dir, file)

        if file.lower().endswith(".jpg"):
            shutil.copy(src_path, os.path.join(DST_IMG, file))
            img_count += 1

        elif file.lower().endswith(".txt"):
            shutil.copy(src_path, os.path.join(DST_LBL, file))
            lbl_count += 1

print(f"✅ Done: {img_count} images and {lbl_count} labels moved")
