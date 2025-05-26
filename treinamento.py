import glob
import os
import random
import shutil
import sys

import cv2
import yaml
from tqdm import tqdm

# --- CONFIGURAÇÕES ---
ORIG_IMG_DIR = r"C:\Users\Usuario\Desktop\Faculdade\Sinal de Perigo\Gesture Image Data"
ORIG_MASK_DIR = r"C:\Users\Usuario\Desktop\Faculdade\Sinal de Perigo\Gesture Image Pre-Processed Data"

YOLO_DATA_DIR = "dataset_gestos"
SPLIT = 0.8  # 80% train, 20% val
IMG_SIZE = (416, 416)  # opcional: redimensionar

# Cria pastas
for sub in ("images/train", "images/val", "labels/train", "labels/val"):
    os.makedirs(f"{YOLO_DATA_DIR}/{sub}", exist_ok=True)

# Mapeia classes
classes = sorted(os.listdir(ORIG_IMG_DIR))
cls2idx = {c: i for i, c in enumerate(classes)}

# Coleta todos os pares (imagem, máscara)
data = []
for cls in classes:
    img_folder = os.path.join(ORIG_IMG_DIR, cls)
    mask_folder = os.path.join(ORIG_MASK_DIR, cls)
    for img_path in glob.glob(f"{img_folder}/*.jpg"):
        filename = os.path.basename(img_path)
        mask_path = os.path.join(mask_folder, filename)
        if os.path.isfile(mask_path):
            data.append((img_path, mask_path, cls2idx[cls]))
        else:
            print(f"Aviso: sem máscara para {img_path}")

random.shuffle(data)
split_idx = int(len(data) * SPLIT)
splits = {
    "train": data[:split_idx],
    "val": data[split_idx:],
}


def process_split(split_name, items):
    for img_path, mask_path, cls_idx in tqdm(items, desc=f"Processando {split_name}"):
        fn = os.path.basename(img_path)
        # Carrega imagem e máscara
        img = cv2.imread(img_path)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        h, w = mask.shape

        # Encontrar contornos na máscara
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue  # sem mão detectada
        # caixa mínima que engloba tudo
        x, y, bw, bh = cv2.boundingRect(cv2.vconcat(contours))

        # opcional: redimensionar imagem para IMG_SIZE e converter bbox
        if IMG_SIZE:
            img = cv2.resize(img, IMG_SIZE)
            scale_x = IMG_SIZE[0] / w
            scale_y = IMG_SIZE[1] / h
            x *= scale_x
            bw *= scale_x
            y *= scale_y
            bh *= scale_y
            w, h = IMG_SIZE

        # normalizar para YOLO: x_center, y_center, w, h
        x_c = (x + bw / 2) / w
        y_c = (y + bh / 2) / h
        nw = bw / w
        nh = bh / h

        # copiar imagem para pasta
        dest_img = os.path.join(YOLO_DATA_DIR, "images", split_name, fn)
        cv2.imwrite(dest_img, img)

        # escrever label
        label_fn = fn.rsplit(".", 1)[0] + ".txt"
        dest_lbl = os.path.join(YOLO_DATA_DIR, "labels", split_name, label_fn)
        with open(dest_lbl, "w") as f:
            f.write(f"{cls_idx} {x_c:.6f} {y_c:.6f} {nw:.6f} {nh:.6f}\n")


# for split_name, items in splits.items():
#     process_split(split_name, items)

# gera data.yaml
cfg = {
    "train": os.path.abspath(f"{YOLO_DATA_DIR}/images/train"),
    "val": os.path.abspath(f"{YOLO_DATA_DIR}/images/val"),
    "nc": len(classes),
    "names": classes,
}
with open(f"{YOLO_DATA_DIR}/data.yaml", "w") as f:
    yaml.dump(cfg, f)


# --- chamada ao treinamento via API do Ultralytics (sem subprocess) ---
if __name__ == "__main__":
    from ultralytics import YOLO

    print("Annotations geradas. Iniciando treino...")

    model = YOLO("weights/yolov5s.pt")  # backbone pré-treinado
    results = model.train(
        data=f"{YOLO_DATA_DIR}/data.yaml",
        epochs=5,
        imgsz=416,
        batch=16,
        name="gestos_modelo",
    )
    print("Treino finalizado. Checkpoints em:", model.trainer.save_dir)
