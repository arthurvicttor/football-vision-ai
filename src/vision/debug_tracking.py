from pathlib import Path

import cv2
from ultralytics import YOLO

VIDEO_PATH = "data/raw/match.mp4"
OUTPUT_PATH = Path("data/processed/debug_tracking.mp4")

MODEL_PATH = "yolo11n.pt"

CONFIDENCE_THRESHOLD = 0.50
PERSON_CLASS = 0

EXPERIMENT_SECONDS = 10

WEBCAM_X1 = 280
WEBCAM_Y1 = 260

WEBCAM_X2 = 380
WEBCAM_Y2 = 360

def is_inside_webcam(x, y):
    return (
        WEBCAM_X1 <= x <= WEBCAM_X2
        and
        WEBCAM_Y1 <= y <= WEBCAM_Y2
    )


def main():

    model = YOLO(MODEL_PATH)

    video = cv2.VideoCapture(VIDEO_PATH)

    fps = video.get(cv2.CAP_PROP_FPS)

    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))

    total_frames = int(fps * EXPERIMENT_SECONDS)

    writer = cv2.VideoWriter(
        str(OUTPUT_PATH),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height)
    )

    frame_number = 0

    while frame_number < total_frames:

        success, frame = video.read()

        if not success:
            break

        results = model.track(
            frame,
            persist=True,
            tracker="trackers/custom_bytetrack.yaml",
            classes=[PERSON_CLASS],
            verbose=False,
        )

        result = results[0]

        if result.boxes is not None:

            for box in result.boxes:

                if box.id is None:
                    continue

                confidence = float(box.conf[0])

                if confidence < CONFIDENCE_THRESHOLD:
                    continue

                track_id = int(box.id[0])

                x1, y1, x2, y2 = map(
                    int,
                    box.xyxy[0]
                )

                center_x = (x1 + x2) // 2
                center_y = (y1 + y2) // 2

                if is_inside_webcam(center_x, center_y):
                    continue

                cv2.rectangle(
                    frame,
                    (x1, y1),
                    (x2, y2),
                    (0, 255, 0),
                    2
                )

                cv2.circle(
                    frame,
                    (center_x, center_y),
                    4,
                    (0, 0, 255),
                    -1
                )

                cv2.putText(
                    frame,
                    f"ID {track_id}",
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0,255,0),
                    2
                )

        writer.write(frame)

        frame_number += 1

    video.release()
    writer.release()

    print()
    print("Vídeo criado:")
    print(OUTPUT_PATH)


if __name__ == "__main__":
    main()