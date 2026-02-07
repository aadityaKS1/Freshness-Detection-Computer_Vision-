import os
import cv2

# class name → class id
CLASS_MAP = {
    "Apple": 0,
    "Banana": 1,
    "Carrot": 2,
    "Tomato": 3
}

BASE = "yolo_dataset"

SETS = ["train", "val"]

def convert_label(img_path, lbl_path):
    img = cv2.imread(img_path)
    if img is None:
        return

    h, w, _ = img.shape
    new_lines = []

    with open(lbl_path, "r") as f:
        lines = f.readlines()

    for line in lines:
        parts = line.strip().split()

        # Open Images format: Tomato xmin ymin xmax ymax
        if len(parts) == 5 and parts[0] in CLASS_MAP:
            cls, xmin, ymin, xmax, ymax = parts
            xmin, ymin, xmax, ymax = map(float, [xmin, ymin, xmax, ymax])

            xc = ((xmin + xmax) / 2) / w
            yc = ((ymin + ymax) / 2) / h
            bw = (xmax - xmin) / w
            bh = (ymax - ymin) / h

            new_lines.append(
                f"{CLASS_MAP[cls]} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}\n"
            )

        # Already YOLO format → keep
        elif len(parts) == 5 and parts[0].isdigit():
            new_lines.append(line)

        # ignore anything else
        else:
            continue

    with open(lbl_path, "w") as f:
        f.writelines(new_lines)


for split in SETS:
    img_dir = os.path.join(BASE, "images", split)
    lbl_dir = os.path.join(BASE, "labels", split)

    print(f"📂 Processing {split}")

    for file in os.listdir(lbl_dir):
        if file.endswith(".txt"):
            name = os.path.splitext(file)[0]

            # find matching image (jpg / png / jpeg)
            img_path = None
            for ext in [".jpg", ".jpeg", ".png"]:
                candidate = os.path.join(img_dir, name + ext)
                if os.path.exists(candidate):
                    img_path = candidate
                    break

            if img_path:
                convert_label(img_path, os.path.join(lbl_dir, file))

print("✅ Labels converted correctly using YOLO dataset folders")
