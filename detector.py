# detector.py

import cv2
import torch

from config import CLASSES, CONFIDENCE_THRESHOLD, WINDOW_NAME, YOLO_MODEL_PATH

# Carregar modelo YOLO treinado
model = torch.hub.load(
    "ultralytics/yolov5", "custom", path=YOLO_MODEL_PATH, force_reload=True
)
model.conf = CONFIDENCE_THRESHOLD  # Threshold de confiança

# Acessar webcam (0 é geralmente a câmera padrão)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Erro ao acessar a webcam.")
    exit()

print("Iniciando detecção de gestos. Pressione 'q' para sair.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Inferência com YOLO
    results = model(frame)

    # Renderizar as detecções no próprio frame
    rendered = results.render()[0]  # Renderiza bounding boxes, classes e scores
    cv2.imshow(WINDOW_NAME, rendered)

    # Pressione 'q' para sair
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Finalizar
cap.release()
cv2.destroyAllWindows()
