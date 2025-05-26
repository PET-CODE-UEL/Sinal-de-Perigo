import cv2
from ultralytics import YOLO

from config import CONFIDENCE_THRESHOLD, WINDOW_NAME, YOLO_MODEL_PATH

# Carrega o modelo treinado
model = YOLO(YOLO_MODEL_PATH)

# Acessa a webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Erro ao acessar a webcam.")
    exit()

print("Iniciando detecção de gestos. Pressione 'q' para sair.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Inferência
    results = model.predict(source=frame, conf=CONFIDENCE_THRESHOLD, verbose=False)

    # Renderiza as detecções no próprio frame
    rendered = results[0].plot()
    cv2.imshow(WINDOW_NAME, rendered)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
