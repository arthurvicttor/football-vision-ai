import cv2


VIDEO_PATH = "data/raw/match.mp4"

FRAMES_TO_ANALYZE = 150

FIELD_X1 = 3
FIELD_Y1 = 40
FIELD_X2 = 640
FIELD_Y2 = 356

MIN_CONTOUR_AREA = 8
MAX_CONTOUR_AREA = 1500


def main():
    video = cv2.VideoCapture(VIDEO_PATH)

    if not video.isOpened():
        print("Erro: não foi possível abrir o vídeo.")
        return

    success, previous_frame = video.read()

    if not success:
        print("Erro: não foi possível ler o primeiro frame.")
        video.release()
        return

    previous_gray = cv2.cvtColor(
        previous_frame,
        cv2.COLOR_BGR2GRAY
    )

    frame_number = 1

    while frame_number < FRAMES_TO_ANALYZE:

        success, frame = video.read()

        if not success:
            break

        current_gray = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2GRAY
        )

        # Recorta somente a área do campo
        previous_roi = previous_gray[
            FIELD_Y1:FIELD_Y2,
            FIELD_X1:FIELD_X2
        ]

        current_roi = current_gray[
            FIELD_Y1:FIELD_Y2,
            FIELD_X1:FIELD_X2
        ]

        # Diferença entre frames
        difference = cv2.absdiff(
            previous_roi,
            current_roi
        )

        # Binarização
        _, threshold = cv2.threshold(
            difference,
            25,
            255,
            cv2.THRESH_BINARY
        )

        # Pequena limpeza de ruído
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (3, 3)
        )

        threshold = cv2.morphologyEx(
            threshold,
            cv2.MORPH_OPEN,
            kernel
        )

        contours, _ = cv2.findContours(
            threshold,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        candidate_count = 0

        for contour in contours:

            area = cv2.contourArea(contour)

            if area < MIN_CONTOUR_AREA:
                continue

            if area > MAX_CONTOUR_AREA:
                continue

            x, y, w, h = cv2.boundingRect(contour)

            # Corrige coordenadas da ROI para o frame original
            x += FIELD_X1
            y += FIELD_Y1

            cv2.rectangle(
                frame,
                (x, y),
                (x + w, y + h),
                (0, 255, 255),
                1
            )

            candidate_count += 1

        cv2.putText(
            frame,
            f"Candidates: {candidate_count}",
            (10, 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 255),
            2
        )

        cv2.imshow(
            "Motion Candidates",
            frame
        )

        cv2.imshow(
            "Motion Mask",
            threshold
        )

        previous_gray = current_gray

        frame_number += 1

        if cv2.waitKey(30) & 0xFF == ord("q"):
            break

    video.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()