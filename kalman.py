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


model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

# Создание объекта KalmanFilter
kf = KalmanFilter(dim_x=4, dim_z=2)
kf.F = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]])
kf.H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]])
kf.R *= np.array([[1, 1], [1, 1]])

cap = cv2.VideoCapture(0)

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
            cv2.rectangle(resize_image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

    cv2.imshow("Frame", resize_image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
