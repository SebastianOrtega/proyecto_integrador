import numpy as np
import cv2
from ultralytics import YOLO
from sort import Sort
import tensorflow as tf

# Verificar si la GPU está disponible
print(tf.config.list_physical_devices('GPU'))

if __name__ == '__main__':
    cap = cv2.VideoCapture("./video2.mp4")

    model = YOLO("besttrain3.pt")

    tracker = Sort()

    first_frame_saved = False  # Bandera para guardar el primer frame

    entradas = 0  # Contador de entradas (y de más a menos)
    salidas = 0  # Contador de salidas (y de menos a más)

    # Diccionario para almacenar la posición inicial de detección de cada ID
    detection_start_positions = {}
    counted_ids = set()  # Conjunto para almacenar los IDs que ya han sido contados

    frame_count = 0  # Contador de frames

    # Definir el codec y crear el objeto VideoWriter para guardar el video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('video_anotado.mp4', fourcc,
                          20.0, (int(cap.get(3)), int(cap.get(4))))

    while cap.isOpened():
        status, frame = cap.read()

        if not status:
            break

        # Saltar los frames pares
        if frame_count % 2 == 0:
            frame_count += 1
            continue

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
                # Dibujar el ID y la caja delimitadora
                cv2.putText(img=frame, text=f"Id: {track_id}", org=(
                    xmin, ymin-10), fontFace=cv2.FONT_HERSHEY_PLAIN, fontScale=2, color=(0, 255, 0), thickness=2)
                cv2.rectangle(img=frame, pt1=(xmin, ymin), pt2=(
                    xmax, ymax), color=(0, 255, 0), thickness=2)

                # Verificar si el ID ya tiene una posición inicial registrada
                if track_id not in detection_start_positions:
                    detection_start_positions[track_id] = ymin

                # Verificar si el ID ya fue contado
                if track_id not in counted_ids:
                    start_y = detection_start_positions[track_id]
                    # Comprobar si el objeto se ha movido más de 50 píxeles en y (de menos a más)
                    if start_y < ymin and (ymin - start_y) >= 50:
                        salidas += 1
                        counted_ids.add(track_id)
                        print(f"Salida detectada. Total salidas: {salidas}")
                    # Comprobar si el objeto se ha movido más de 50 píxeles en y (de más a menos)
                    elif start_y > ymin and (start_y - ymin) >= 50:
                        entradas += 1
                        counted_ids.add(track_id)
                        print(f"Entrada detectada. Total entradas: {entradas}")

        # Dibujar las líneas para limitar la región de interés
        cv2.line(frame, (0, 100), (frame.shape[1], 100), color=(
            0, 0, 255), thickness=2)

        # Mostrar el contador de entradas y salidas en la parte superior izquierda
        cv2.putText(img=frame, text=f"Entradas: {entradas}", org=(
            10, 50), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(0, 255, 255), thickness=2)
        cv2.putText(img=frame, text=f"Salidas: {salidas}", org=(
            10, 80), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(0, 255, 255), thickness=2)

        # Escribir el frame en el video de salida
        out.write(frame)

        # Mostrar el frame completo, sin recortar
        cv2.imshow("frame", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

        frame_count += 1

    cap.release()
    out.release()
    cv2.destroyAllWindows()
