import cv2
import torch

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

model = torch.load('/Users/alexegorov/skud/crowdhuman_yolov5m.pt')

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    resize_image = image_resize(frame, 1280, 720, cv2.INTER_AREA)
    (H, W) = resize_image.shape[:2]

    result = model(resize_image)
    result.save()
    cv2.imshow("Frame", resize_image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
