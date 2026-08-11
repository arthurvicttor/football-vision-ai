import cv2


IMAGE_PATH = "data/processed/frames/frame_000000.jpg"


def main():
    image = cv2.imread(IMAGE_PATH)

    if image is None:
        print("Erro: não foi possível carregar a imagem.")
        return

    height, width = image.shape[:2]

    # Linhas verticais
    for x in range(0, width, 20):
        cv2.line(
            image,
            (x, 0),
            (x, height),
            (255, 255, 255),
            1
        )

        cv2.putText(
            image,
            str(x),
            (x + 2, 15),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.35,
            (255, 255, 255),
            1
        )

    # Linhas horizontais
    for y in range(0, height, 20):
        cv2.line(
            image,
            (0, y),
            (width, y),
            (255, 255, 255),
            1
        )

        cv2.putText(
            image,
            str(y),
            (2, y - 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.35,
            (255, 255, 255),
            1
        )

    cv2.imshow("Ball Calibration", image)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()