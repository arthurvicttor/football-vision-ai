import cv2
from pathlib import Path


VIDEO_PATH = "data/raw/match.mp4"
OUTPUT_DIR = Path("data/processed/frames")


def main():
    video = cv2.VideoCapture(VIDEO_PATH)

    if not video.isOpened():
        print("Erro: não foi possível abrir o vídeo.")
        return

    fps = video.get(cv2.CAP_PROP_FPS)
    frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))

    duration = frame_count / fps if fps > 0 else 0

    print(f"Resolução: {width}x{height}")
    print(f"FPS: {fps:.2f}")
    print(f"Frames: {frame_count}")
    print(f"Duração: {duration:.2f} segundos")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    interval_seconds = 10
    frame_interval = int(fps * interval_seconds)

    current_frame = 0
    saved_frames = 0

    while True:
        success, frame = video.read()

        if not success:
            break

        if current_frame % frame_interval == 0:
            output_path = OUTPUT_DIR / f"frame_{current_frame:06d}.jpg"

            cv2.imwrite(str(output_path), frame)

            saved_frames += 1
            print(f"Frame salvo: {output_path}")

        current_frame += 1

    video.release()

    print(f"\nTotal de frames salvos: {saved_frames}")


if __name__ == "__main__":
    main()