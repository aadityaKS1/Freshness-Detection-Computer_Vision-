import os

BASE_DIR = "classification_dataset"
SPLITS = ["train", "val", "test"]
CLASSES = ["fresh", "rotten"]

print("\n📊 DATASET DISTRIBUTION\n")

for split in SPLITS:
    print(f"--- {split.upper()} ---")
    for cls in CLASSES:
        path = os.path.join(BASE_DIR, split, cls)
        if not os.path.exists(path):
            print(f"{cls}: 0 (folder missing)")
            continue

        count = len([
            f for f in os.listdir(path)
            if os.path.isfile(os.path.join(path, f))
        ])
        print(f"{cls}: {count}")
    print()
