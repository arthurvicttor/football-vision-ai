from pathlib import Path

import cv2
from ultralytics import YOLO


VIDEO_PATH = "data/raw/match.mp4"
OUTPUT_PATH = Path("data/processed/player_tracking.mp4")

MODEL_PATH = "yolo11n.pt"

# Classe COCO:
# 0 = person
PERSON_CLASS = 0

# Quantos segundos queremos testar
EXPERIMENT_SECONDS = 10


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

        # Tracking com ByteTrack
        results = model.track(
            frame,
            persist=True,
            tracker="bytetrack.yaml",
            classes=[PERSON_CLASS],
            verbose=False,
        )

        result = results[0]

        if result.boxes is not None:

            boxes = result.boxes

            for box in boxes:

                # ID atribuído pelo tracker
                if box.id is None:
                    continue

                track_id = int(box.id[0])

                confidence = float(box.conf[0])

                x1, y1, x2, y2 = map(
                    int,
                    box.xyxy[0]
                )

                # Centro do jogador
                center_x = (x1 + x2) // 2
                center_y = (y1 + y2) // 2

                # Caixa
                cv2.rectangle(
                    frame,
                    (x1, y1),
                    (x2, y2),
                    (0, 255, 0),
                    2,
                )

                # ID
                cv2.putText(
    frame,
    f"ID {track_id} | {confidence:.2f}",
    (x1, max(y1 - 10, 20)),
    cv2.FONT_HERSHEY_SIMPLEX,
    0.45,
    (0, 255, 0),
    2,
)

                # Centro
                cv2.circle(
                    frame,
                    (center_x, center_y),
                    4,
                    (0, 0, 255),
                    -1,
                )

                print(
                    f"Frame {frame_number}: "
                    f"Player {track_id} "
                    f"center=({center_x}, {center_y}) "
                    f"conf={confidence:.2f}"
                )

        writer.write(frame)

        frame_number += 1

    video.release()
    writer.release()

    print()
    print("Tracking finalizado.")
    print(f"Frames analisados: {frame_number}")
    print(f"Resultado: {OUTPUT_PATH}")

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()