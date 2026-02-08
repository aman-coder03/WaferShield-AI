import os
import shutil
import random

base_dir = r"C:\Users\Aman Srivastava\Desktop\Programs\Projects\YieldGuard\data"
train_dir = os.path.join(base_dir, "train")
val_dir = os.path.join(base_dir, "val")
test_dir = os.path.join(base_dir, "test")

os.makedirs(val_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)

classes = os.listdir(train_dir)

for cls in classes:
    cls_path = os.path.join(train_dir, cls)
    images = os.listdir(cls_path)
    random.shuffle(images)

    total = len(images)
    val_count = int(0.15 * total)
    test_count = int(0.15 * total)

    val_images = images[:val_count]
    test_images = images[val_count:val_count+test_count]

    os.makedirs(os.path.join(val_dir, cls), exist_ok=True)
    os.makedirs(os.path.join(test_dir, cls), exist_ok=True)

    for img in val_images:
        shutil.copy(os.path.join(cls_path, img),
                    os.path.join(val_dir, cls, img))

    for img in test_images:
        shutil.copy(os.path.join(cls_path, img),
                    os.path.join(test_dir, cls, img))

print("Dataset split into train/val/test successfully!")
