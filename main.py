import numpy as np
from filterpy.kalman import KalmanFilter
import torch

INPUT_WIDTH = 640
INPUT_HEIGHT = 640

# Создание фильтра Калмана
def create_kalman_filter():
    kalman_filter = KalmanFilter(dim_x=4, dim_z=2)
    kalman_filter.x = np.array([0, 0, 0, 0])  # Начальное состояние фильтра
    kalman_filter.F = np.array([[1, 0, 1, 0],
                                [0, 1, 0, 1],
                                [0, 0, 1, 0],
                                [0, 0, 0, 1]])  # Матрица перехода состояний
    kalman_filter.H = np.array([[1, 0, 0, 0],
                                [0, 1, 0, 0]])  # Матрица измерений
    kalman_filter.P *= 1000  # Ковариация ошибки состояния
    kalman_filter.R = np.array([[0.1, 0],
                                [0, 0.1]])  # Ковариация шума измерений
    kalman_filter.Q = np.eye(4) * 0.01  # Ковариация шума процесса
    return kalman_filter

# Функция трекинга с использованием фильтра Калмана и YOLOv5
def kalman_filter_tracking(frame, kalman_filter, model):
    # Применение YOLOv5 для получения предсказаний объектов
    predictions = model.predict(frame)

    # Извлечение координат и классов объектов из предсказаний
    boxes = predictions.xyxy[0]  # Координаты ограничивающих рамок объектов
    classes = predictions.names[0]  # Классы объектов

    # Применение фильтра Калмана к координатам объектов
    tracked_boxes = []
    for box in boxes:
        x = (box[0] + box[2]) / 2  # Середина по оси X
        y = (box[1] + box[3]) / 2  # Середина по оси Y

        # Обновление состояния фильтра Калмана
        kalman_filter.predict()
        kalman_filter.update([x, y])

        # Получение трекированных координат объекта
        tracked_x, tracked_y = kalman_filter.x[:2]

        # Добавление трекированных координат в список
        tracked_boxes.append([tracked_x, tracked_y, tracked_x, tracked_y])

    # Возвращение трекированных координат объектов
    return np.array(tracked_boxes)



def format_yolov5(frame):
    row, col, _ = frame.shape
    _max = max(col, row)
    result = np.zeros((_max, _max, 3), np.uint8)
    result[0:row, 0:col] = frame
    return result

def detect(image, net):
    blob = cv2.dnn.blobFromImage(image, 1/255.0, (INPUT_WIDTH, INPUT_HEIGHT), swapRB=True, crop=False)
    net.setInput(blob)
    preds = net.forward()
    return preds

# Создание фильтра Калмана
kalman_filter = create_kalman_filter()
net = cv2.dnn.readNet("yolov5s.onnx")

# Загрузка модели YOLOv5
#model = load_yolov5_model()

import cv2

# Создание объекта захвата видеопотока с веб-камеры
video_capture = cv2.VideoCapture(0)

while True:

    # Обработка нажатия клавиши 'q' для выхода из цикла
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    # Захват кадра из видеопотока
    ret, frame = video_capture.read()

    if not ret:
        break

    inputImage = format_yolov5(frame)
    outs = detect(inputImage, net)   
    
    # Обработка кадра (добавьте вашу логику обработки кадра здесь)
    # Применение фильтра Калмана и YOLOv5 для трекинга объектов
    tracked_boxes = kalman_filter_tracking(frame, kalman_filter, model)

    # Визуализация трекированных объектов на кадре
    for box in tracked_boxes:
        x1, y1, x2, y2 = box.astype(int)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Отображение кадра
    cv2.imshow('Webcam', frame)

    # Обработка нажатия клавиши 'q' для выхода из цикла
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Освобождение ресурсов
video_capture.release()
cv2.destroyAllWindows()
