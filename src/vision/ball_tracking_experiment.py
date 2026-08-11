from pathlib import Path

import cv2
from ultralytics import YOLO


VIDEO_PATH = "data/raw/match.mp4"
OUTPUT_PATH = Path("data/processed/ball_experiment.mp4")

MODEL_PATH = "yolo11n.pt"

# Começaremos analisando apenas 5 segundos
EXPERIMENT_SECONDS = 5

# Classe "sports ball" do COCO
SPORTS_BALL_CLASS = 32


def main():
    model = YOLO(MODEL_PATH)

    video = cv2.VideoCapture(VIDEO_PATH)

    if not video.isOpened():
        print("Erro: não foi possível abrir o vídeo.")
        return

    fps = video.get(cv2.CAP_PROP_FPS)
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))

    total_frames = int(fps * EXPERIMENT_SECONDS)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    writer = cv2.VideoWriter(
        str(OUTPUT_PATH),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )

    frame_number = 0

    while frame_number < total_frames:

        success, frame = video.read()

        if not success:
            break

        results = model(frame, verbose=False)

        detections = results[0].boxes

        ball_found = False

        if detections is not None:
            for box in detections:

                class_id = int(box.cls[0])

                if class_id != SPORTS_BALL_CLASS:
                    continue

                confidence = float(box.conf[0])

                x1, y1, x2, y2 = map(int, box.xyxy[0])

                center_x = (x1 + x2) // 2
                center_y = (y1 + y2) // 2

                print(
                    f"Frame {frame_number}: "
                    f"bola candidata=({center_x}, {center_y}) "
                    f"conf={confidence:.2f}"
                )

                # Desenha bounding box
                cv2.rectangle(
                    frame,
                    (x1, y1),
                    (x2, y2),
                    (0, 255, 0),
                    2,
                )

                # Desenha centro
                cv2.circle(
                    frame,
                    (center_x, center_y),
                    5,
                    (0, 0, 255),
                    -1,
                )

                # Mostra confiança
                cv2.putText(
                    frame,
                    f"ball {confidence:.2f}",
                    (x1, max(y1 - 10, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    2,
                )

                ball_found = True

        if not ball_found:
            print(f"Frame {frame_number}: nenhuma bola candidata")

        writer.write(frame)

        frame_number += 1

    video.release()
    writer.release()

    print()
    print(f"Experimento finalizado.")
    print(f"Frames analisados: {frame_number}")
    print(f"Resultado: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()