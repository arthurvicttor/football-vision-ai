from pathlib import Path

import cv2
from ultralytics import YOLO


IMAGE_PATH = Path("data/processed/frames/frame_000000.jpg")
OUTPUT_PATH = Path("data/processed/detection_roi.jpg")


# Região da webcam
WEBCAM_X1 = 280
WEBCAM_Y1 = 260
WEBCAM_X2 = 380
WEBCAM_Y2 = 360


def main():
    model = YOLO("yolo11n.pt")

    image = cv2.imread(str(IMAGE_PATH))

    if image is None:
        print("Erro: não foi possível carregar a imagem.")
        return

    # Cria uma cópia da imagem
    processed_image = image.copy()

    # Remove a webcam
    processed_image[
        WEBCAM_Y1:WEBCAM_Y2,
        WEBCAM_X1:WEBCAM_X2
    ] = 0

    # Executa o YOLO
    results = model(processed_image)

    # Desenha as detecções
    annotated_image = results[0].plot()

    # Salva resultado
    cv2.imwrite(str(OUTPUT_PATH), annotated_image)

    print(f"Imagem salva em: {OUTPUT_PATH}")

    # Mostra resultado
    cv2.imshow("YOLO - Webcam Removida", annotated_image)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()