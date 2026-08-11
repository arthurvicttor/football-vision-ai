import cv2
from pathlib import Path


VIDEO_PATH = "data/raw/match.mp4"
OUTPUT_DIR = Path("data/processed/ball_roi")

BALL_X = 320
BALL_Y = 160

# Tamanho da região ao redor da bola
ROI_SIZE = 40

# Quantidade de frames que vamos observar
FRAMES_TO_ANALYZE = 60


def main():
    video = cv2.VideoCapture(VIDEO_PATH)

    if not video.isOpened():
        print("Erro: não foi possível abrir o vídeo.")
        return

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    frame_number = 0

    while frame_number < FRAMES_TO_ANALYZE:

        success, frame = video.read()

        if not success:
            break

        height, width = frame.shape[:2]

        x1 = max(BALL_X - ROI_SIZE, 0)
        y1 = max(BALL_Y - ROI_SIZE, 0)

        x2 = min(BALL_X + ROI_SIZE, width)
        y2 = min(BALL_Y + ROI_SIZE, height)

        roi = frame[y1:y2, x1:x2]

        output_path = OUTPUT_DIR / f"roi_{frame_number:04d}.jpg"

        cv2.imwrite(str(output_path), roi)

        frame_number += 1

    video.release()

    print(f"Frames analisados: {frame_number}")
    print(f"ROIs salvas em: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()