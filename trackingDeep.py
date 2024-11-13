import numpy as np
import cv2
from ultralytics import YOLO
from sort import Sort
import tensorflow as tf

# Verificar si la GPU está disponible
print(tf.config.list_physical_devices('GPU'))

if __name__ == '__main__':
    cap = cv2.VideoCapture("./trim.mp4")

    model = YOLO("besttrain3.pt")

    tracker = Sort()

    first_frame_saved = False  # Bandera para guardar el primer frame

    while cap.isOpened():
        status, frame = cap.read()

        if not status:
            break

        # Guardar el primer frame
        if not first_frame_saved:
            cv2.imwrite("primero.png", frame)
            print("Formato: ", frame.shape)
            first_frame_saved = True

        # Recortar la parte que no se quiere procesar (y de 100 hacia abajo)
        frame_cropped = frame[100:, :]

        # Realizar la inferencia usando la GPU, si está disponible
        with tf.device('/GPU:0'):
            results = model(frame_cropped, stream=True)

        for res in results:
            filtered_indices = np.where(res.boxes.conf.cpu().numpy() > 0.5)[0]
            boxes = res.boxes.xyxy.cpu().numpy()[filtered_indices].astype(int)

            # Ajustar las coordenadas para la imagen original
            boxes[:, 1] += 100  # Ajustar ymin
            boxes[:, 3] += 100  # Ajustar ymax

            tracks = tracker.update(boxes)
            tracks = tracks.astype(int)

            for xmin, ymin, xmax, ymax, track_id in tracks:
                cv2.putText(img=frame, text=f"Id: {track_id}", org=(
                    xmin, ymin-10), fontFace=cv2.FONT_HERSHEY_PLAIN, fontScale=2, color=(0, 255, 0), thickness=2)
                cv2.rectangle(img=frame, pt1=(xmin, ymin), pt2=(
                    xmax, ymax), color=(0, 255, 0), thickness=2)

        # Dibujar las líneas para limitar la región de interés
        cv2.line(frame, (0, 100), (frame.shape[1], 100), color=(
            0, 0, 255), thickness=2)

        # Mostrar el frame completo, sin recortar
        cv2.imshow("frame", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
