from ultralytics import YOLO
import cv2

from src.vision.diagnostics.switch_detector import SwitchDiagnostics

PERSON_CLASS = 0
VIDEO_PATH = "data/raw/match.mp4"
MODEL_PATH = "yolo11n.pt"


def extract_detections(results):
    detections = []
    boxes = results[0].boxes

    if boxes.id is None:
        return detections

    for box, track_id in zip(boxes.xyxy, boxes.id):
        x1, y1, x2, y2 = box.tolist()
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        detections.append({
            "track_id": int(track_id),
            "center": (cx, cy),
        })

    return detections


def main():
    model = YOLO(MODEL_PATH)
    cap = cv2.VideoCapture(VIDEO_PATH)

    diagnostics = SwitchDiagnostics(max_gap_frames=45, max_gap_distance=100.0)

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model.track(
            frame,
            persist=True,
            tracker="bytetrack.yaml",
            classes=[PERSON_CLASS],
            verbose=False
        )

        detections = extract_detections(results)
        diagnostics.update(frame_idx, detections)

        frame_idx += 1

    cap.release()

    print(diagnostics.summary())


if __name__ == "__main__":
    main()