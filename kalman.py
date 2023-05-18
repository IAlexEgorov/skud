import cv2
import torch
import numpy as np
from filterpy.kalman import KalmanFilter


def image_resize(image, width, height, inter=cv2.INTER_AREA):
    # initialize the dimensions of the image to be resized and
    # grab the image size
    dim = None
    (h, w) = image.shape[:2]

    # if both the width and height are None, then return the
    # original image
    if width is None and height is None:
        return image

    # check to see if the width is None
    if width is None:
        # calculate the ratio of the height and construct the
        # dimensions
        r = height / float(h)
        dim = (int(w * r), height)

    # otherwise, the height is None
    else:
        # calculate the ratio of the width and construct the
        # dimensions
        r = width / float(w)
        dim = (width, int(h * r))

    # resize the image
    resized = cv2.resize(image, dim, interpolation=inter)

    # return the resized image
    return resized


MODEL_PATH = "/Users/alexegorov/skud/"


model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

# Создание объекта KalmanFilter
kf = KalmanFilter(dim_x=4, dim_z=2)
kf.F = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]])
kf.H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]])
kf.R *= np.array([[1, 1], [1, 1]])

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# Задаем параметры для сохранения видео
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
out = cv2.VideoWriter('outpy.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 10, (frame_width,frame_height))

while True:
    ret, frame = cap.read()

    # 1080 x 1920
    resize_image = image_resize(frame, 1280, 720, cv2.INTER_AREA)
    (H, W) = resize_image.shape[:2]

    result = model(resize_image)
    # print(result.pandas().xyxy[0])
    df = result.pandas().xyxy[0]

    for index, row in df.iterrows():
        if row['confidence'] > 0.5:  # Фильтрация объектов с низкой уверенностью
            x_min, y_min, x_max, y_max = int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax'])
            x_center, y_center = (x_min + x_max) / 2, (y_min + y_max) / 2
            kf.predict()
            kf.update([x_center, y_center])
            x, y = kf.x[0:2]
            cv2.rectangle(resize_image, (x_min + 10, y_min + 10), (x_max - 10, y_max - 10), (0, 255, 0), 2)

    result.save()
    cv2.imshow("Frame", resize_image)

    # Сохранение в файл
    out.write(resize_image)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()
