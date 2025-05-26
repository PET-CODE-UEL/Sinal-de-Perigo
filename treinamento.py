# treinamento.py

import glob
import os
import random
import shutil
import sys

import cv2
import yaml  # para gerar o arquivo data.yaml
from tqdm import tqdm

# Caminho original do dataset (ex: Gesture Image Data)
ORIGINAL_DATASET = (
    r"C:\Users\Usuario\Desktop\Faculdade\Sinal de Perigo\Gesture Image Data"
)

# Novo caminho para formato YOLO
YOLO_DATASET_DIR = "dataset_gestos"
TRAIN_SPLIT = 0.8  # 80% treino, 20% validação

# Cria estrutura de pastas
for split in ["train", "val"]:
    os.makedirs(f"{YOLO_DATASET_DIR}/images/{split}", exist_ok=True)
    os.makedirs(f"{YOLO_DATASET_DIR}/labels/{split}", exist_ok=True)

# Mapear classes
classes = sorted(os.listdir(ORIGINAL_DATASET))
class_to_index = {name: idx for idx, name in enumerate(classes)}

# Converter imagens
all_data = []

for class_name in classes:
    class_dir = os.path.join(ORIGINAL_DATASET, class_name)
    if not os.path.isdir(class_dir):
        continue
    for img_path in glob.glob(f"{class_dir}/*.*"):
        all_data.append((img_path, class_to_index[class_name]))

# Embaralhar e dividir
random.shuffle(all_data)
split_idx = int(TRAIN_SPLIT * len(all_data))
train_data = all_data[:split_idx]
val_data = all_data[split_idx:]


def convert_and_copy(data_list, split):
    for img_path, class_idx in tqdm(data_list, desc=f"Processando {split}"):
        filename = os.path.basename(img_path)
        img = cv2.imread(img_path)
        h, w = img.shape[:2]

        # Copia a imagem
        dest_img_path = os.path.join(YOLO_DATASET_DIR, "images", split, filename)
        shutil.copyfile(img_path, dest_img_path)

        # Gera label com bounding box centralizada (assumido!)
        x_center, y_center = 0.5, 0.5
        box_width, box_height = 0.8, 0.8  # valores padrão, ajustável

        label_path = os.path.join(
            YOLO_DATASET_DIR, "labels", split, filename.rsplit(".", 1)[0] + ".txt"
        )
        with open(label_path, "w") as f:
            f.write(f"{class_idx} {x_center} {y_center} {box_width} {box_height}\n")


convert_and_copy(train_data, "train")
convert_and_copy(val_data, "val")

# Criar data.yaml
data_yaml = {
    "train": os.path.abspath(YOLO_DATASET_DIR + "/images/train"),
    "val": os.path.abspath(YOLO_DATASET_DIR + "/images/val"),
    "nc": len(classes),
    "names": classes,
}

with open(os.path.join(YOLO_DATASET_DIR, "data.yaml"), "w") as f:
    yaml.dump(data_yaml, f)

print("Pré-processamento finalizado. Iniciando treinamento...")

# --- Treinamento com YOLOv5 ---

import subprocess

# Caminho para o script de treino (requer yolov5 clonado)
yolov5_dir = "yolov5"  # clone do repositório
subprocess.run(
    [
        sys.executable,
        f"{yolov5_dir}/train.py",
        "--img",
        "416",
        "--batch",
        "16",
        "--epochs",
        "30",
        "--data",
        f"{YOLO_DATASET_DIR}/data.yaml",
        "--weights",
        "yolov5s.pt",
        "--name",
        "gestos_modelo",
    ]
)
