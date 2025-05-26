# Caminho para o modelo YOLO treinado (pode ser .pt)
YOLO_MODEL_PATH = "yolov5/runs/train/gestos_modelo/weights/best.pt"

# Nomes das classes que o modelo reconhece (na mesma ordem usada no treino)
CLASSES = [
    "_",
    "0",
    "1",
    "2",
    "3",
    "4",
    "5",
    "6",
    "7",
    "8",
    "9",
    "A",
    "B",
    "C",
    "D",
    "E",
    "F",
    "G",
    "H",
    "I",
    "J",
    "K",
    "L",
    "M",
    "N",
    "O",
    "P",
    "Q",
    "R",
    "S",
    "T",
    "U",
    "V",
    "W",
    "X",
    "Y",
    "Z",
]

# Confiabilidade mínima para exibir a detecção
CONFIDENCE_THRESHOLD = 0.5

# Dimensão da janela
WINDOW_NAME = "Detecção de Gestos - YOLO"
