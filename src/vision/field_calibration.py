import cv2


VIDEO_PATH = "data/raw/match.mp4"


def main():
    video = cv2.VideoCapture(VIDEO_PATH)

    if not video.isOpened():
        print("Erro: não foi possível abrir o vídeo.")
        return

    success, frame = video.read()

    if not success:
        print("Erro: não foi possível ler o frame.")
        video.release()
        return

    height, width = frame.shape[:2]

    # Grade de 20 em 20 pixels
    for x in range(0, width, 20):
        cv2.line(
            frame,
            (x, 0),
            (x, height),
            (255, 255, 255),
            1
        )

        cv2.putText(
            frame,
            str(x),
            (x + 2, 15),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.35,
            (255, 255, 255),
            1
        )

    for y in range(0, height, 20):
        cv2.line(
            frame,
            (0, y),
            (width, y),
            (255, 255, 255),
            1
        )

        cv2.putText(
            frame,
            str(y),
            (2, max(y - 2, 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.35,
            (255, 255, 255),
            1
        )

    cv2.imshow("Field Calibration", frame)

    cv2.waitKey(0)

    video.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()