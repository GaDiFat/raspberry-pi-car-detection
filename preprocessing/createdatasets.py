import os
import pandas as pd
import shutil
from sklearn.model_selection import train_test_split

# 1. Parámetros
IMG_H, IMG_W = 380, 676
DATASET_DIR = "./dataset"
IMAGES_DIR = os.path.join(DATASET_DIR, "images")
LABELS_DIR = os.path.join(DATASET_DIR, "labels")

# 2. Crear carpetas
for split in ["train", "val"]:
    os.makedirs(os.path.join(IMAGES_DIR, split), exist_ok=True)
    os.makedirs(os.path.join(LABELS_DIR, split), exist_ok=True)

# 3. Leer CSV
df = pd.read_csv("./originaldata/train_solution_bounding_boxes.csv")
df.rename(columns={"image":"image_id"}, inplace=True)
df["image_id"] = df["image_id"].apply(lambda x: x.split(".")[0])

# 4. Convertir a formato YOLO
df["x_center"] = (df["xmin"] + df["xmax"]) / 2 / IMG_W
df["y_center"] = (df["ymin"] + df["ymax"]) / 2 / IMG_H
df["w"] = (df["xmax"] - df["xmin"]) / IMG_W
df["h"] = (df["ymax"] - df["ymin"]) / IMG_H
df["classes"] = 0  # solo autos

# 5. Dividir train/val
image_ids = df["image_id"].unique()
train_ids, val_ids = train_test_split(image_ids, test_size=0.2, random_state=42)

def save_labels(ids, split):
    for img_id in ids:
        sub = df[df["image_id"] == img_id]
        label_path = os.path.join(LABELS_DIR, split, f"{img_id}.txt")
        with open(label_path, "w") as f:
            for _, row in sub.iterrows():
                f.write(f"{row['classes']} {row['x_center']} {row['y_center']} {row['w']} {row['h']}\n")
        # Copiar imagen correspondiente (ajusta la ruta según tu dataset)
        src_img = f"./originaldata/training_images/{img_id}.jpg"
        dst_img = os.path.join(IMAGES_DIR, split, f"{img_id}.jpg")
        if os.path.exists(src_img):
            shutil.copy(src_img, dst_img)

save_labels(train_ids, "train")
save_labels(val_ids, "val")

# 6. Crear data.yaml
with open("data.yaml", "w") as f:
    f.write(f"train: {IMAGES_DIR}/train\n")
    f.write(f"val: {IMAGES_DIR}/val\n")
    f.write("nc: 1\n")
    f.write("names: ['car']\n")

print("Dataset preparado. Ahora puedes entrenar con YOLOv5:")
print("python yolov5/train.py --img 640 --batch 16 --epochs 50 --data data.yaml --weights yolov5s.pt")