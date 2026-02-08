import os
import shutil
import pandas as pd
import numpy as np
from PIL import Image

pkl_path = r"C:\Users\Aman Srivastava\Desktop\Programs\Projects\YieldGuard\data_raw\LSWMD.pkl"
output_dir = r"C:\Users\Aman Srivastava\Desktop\Programs\Projects\YieldGuard\data\train"

selected_classes = [
    'Clean',
    'Center',
    'Donut',
    'Edge-Loc',
    'Edge-Ring',
    'Loc',
    'Random',
    'Scratch'
]

max_per_class = 149

print("Loading dataset...")
df = pd.read_pickle(pkl_path)

df['failureType'] = df['failureType'].astype(str)
df['failureType'] = df['failureType'].str.replace("[", "", regex=False)
df['failureType'] = df['failureType'].str.replace("]", "", regex=False)
df['failureType'] = df['failureType'].str.replace("'", "", regex=False)

df['failureType'] = df['failureType'].replace({'none': 'Clean'})

print("Available classes after cleaning:")
print(df['failureType'].unique())

df = df[df['failureType'].isin(selected_classes)]

if os.path.exists(output_dir):
    shutil.rmtree(output_dir)

os.makedirs(output_dir, exist_ok=True)

for cls in selected_classes:
    os.makedirs(os.path.join(output_dir, cls), exist_ok=True)

print("\nExtracting images...")

for cls in selected_classes:
    subset = df[df['failureType'] == cls].head(max_per_class)
    print(f"{cls} -> {len(subset)} images")

    for idx, row in subset.iterrows():
        wafer_map = row['waferMap']

        img_array = (wafer_map * 255).astype(np.uint8)
        img = Image.fromarray(img_array).convert("L")

        img.save(os.path.join(output_dir, cls, f"{cls}_{idx}.png"))

print("\nDataset extraction completed successfully!")
print(f"Total images: {len(selected_classes) * max_per_class}")
