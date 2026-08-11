from pathlib import Path
import json

import cv2
from ultralytics import YOLO


VIDEO_PATH = "data/raw/match.mp4"
OUTPUT_PATH = Path("data/processed/player_history.json")

CONFIDENCE_THRESHOLD = 0.50

MODEL_PATH = "yolo11n.pt"

PERSON_CLASS = 0

EXPERIMENT_SECONDS = 10


def main():
    model = YOLO(MODEL_PATH)

    video = cv2.VideoCapture(VIDEO_PATH)

    if not video.isOpened():
        print("Erro: não foi possível abrir o vídeo.")
        return

    fps = video.get(cv2.CAP_PROP_FPS)

    total_frames = int(
        fps * EXPERIMENT_SECONDS
    )

    players = {}

    active_ids_per_frame = []

    frame_number = 0

    while frame_number < total_frames:

        success, frame = video.read()

        if not success:
            break

        results = model.track(
            frame,
            persist=True,
            tracker="bytetrack.yaml",
            classes=[PERSON_CLASS],
            verbose=False,
        )

        result = results[0]

        # IDs válidos detectados neste frame
        current_ids = set()

        if result.boxes is not None:

            boxes = result.boxes

            for box in boxes:

                if box.id is None:
                    continue

                confidence = float(box.conf[0])

                # Ignora detecções com baixa confiança
                if confidence < CONFIDENCE_THRESHOLD:
                    continue

                track_id = int(box.id[0])

                current_ids.add(track_id)

                x1, y1, x2, y2 = map(
                    int,
                    box.xyxy[0]
                )

                center_x = (x1 + x2) // 2
                center_y = (y1 + y2) // 2

                if track_id not in players:
                    players[track_id] = []

                players[track_id].append(
                    {
                        "frame": frame_number,
                        "x": center_x,
                        "y": center_y,
                        "confidence": round(
                            confidence,
                            4
                        ),
                    }
                )

        # Salva o diagnóstico deste frame
        active_ids_per_frame.append(
            {
                "frame": frame_number,
                "active_ids": len(current_ids),
                "ids": sorted(current_ids),
            }
        )

        frame_number += 1

    video.release()

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    # Diagnóstico
    print()
    print("Diagnóstico dos primeiros frames:")
    print()

    for item in active_ids_per_frame[:30]:

        print(
            f"Frame {item['frame']}: "
            f"{item['active_ids']} IDs "
            f"{item['ids']}"
        )

    # Salva histórico
    with open(
        OUTPUT_PATH,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            players,
            file,
            indent=4,
            ensure_ascii=False
        )

    print()
    print("Histórico criado.")
    print(f"Frames analisados: {frame_number}")
    print(f"Jogadores encontrados: {len(players)}")
    print(f"Arquivo: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()