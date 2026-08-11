import cv2


VIDEO_PATH = "data/raw/match.mp4"


# Região da webcam
WEBCAM_X1 = 260
WEBCAM_Y1 = 288
WEBCAM_X2 = 390
WEBCAM_Y2 = 357

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

    # Faz uma cópia para preservar o frame original
    masked_frame = frame.copy()

    # Preenche a região da webcam com preto
    masked_frame[
        WEBCAM_Y1:WEBCAM_Y2,
        WEBCAM_X1:WEBCAM_X2
    ] = 0

    cv2.imshow("Original", frame)
    cv2.imshow("Webcam Masked", masked_frame)

    cv2.waitKey(0)

    video.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()