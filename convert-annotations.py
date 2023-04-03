import os
import json
from PIL import Image

# Define the paths to the annotation files and image directories
annotation_file_train = 'datasets/annotation_train.odgt'
annotation_file_val = 'datasets/annotation_val.odgt'

train_image_dir = 'datasets/train'
val_image_dir = 'datasets/val'

# Define the output directories for the YOLOv7 annotations
train_yolo_dir = 'datasets/crowdhuman/train/labels'
val_yolo_dir = 'datasets/crowdhuman/val/labels'

# Define the desired size for the YOLOv7 images
image_size = (640, 640)

# Create the output directories if they don't exist
os.makedirs(train_yolo_dir, exist_ok=True)
os.makedirs(val_yolo_dir, exist_ok=True)

# Define a function to convert a single annotation
def convert_annotation(annotation, image_dir, yolo_dir):
    # Load the image and get its size
    image_path = os.path.join(image_dir, annotation['ID']+'.jpg')
    image = Image.open(image_path)
    image_width, image_height = image.size

    # Scale the bounding boxes to match the YOLOv7 image size
    x_scale = image_size[0] / image_width
    y_scale = image_size[1] / image_height

    # Create the YOLOv7 annotation file and write the converted annotations
    yolo_file_path = os.path.join(yolo_dir, annotation['ID']+'.txt')
    with open(yolo_file_path, 'w') as yolo_file:
        for bbox in annotation['gtboxes']:
            if bbox['tag'] != 'person':  # skip non-person objects
                continue
            x_center = (bbox['fbox'][0]+bbox['fbox'][2])/2 * x_scale
            y_center = (bbox['fbox'][1]+bbox['fbox'][3])/2 * y_scale
            box_width = (bbox['fbox'][2]-bbox['fbox'][0]) * x_scale
            box_height = (bbox['fbox'][3]-bbox['fbox'][1]) * y_scale
            yolo_line = '0 {:.6f} {:.6f} {:.6f} {:.6f}\n'.format(
                x_center / image_size[0], y_center / image_size[1],
                box_width / image_size[0], box_height / image_size[1])
            yolo_file.write(yolo_line)


# Open the annotation file and loop over the annotations
with open(annotation_file_val, 'r') as f:
    for line in f:
        annotation = json.loads(line)
        # Convert the annotation and save the YOLOv7 file
        convert_annotation(annotation, val_image_dir, val_yolo_dir)

# Open the annotation file and loop over the annotations
with open(annotation_file_train, 'r') as f:
    for line in f:
        annotation = json.loads(line)
        # Convert the annotation and save the YOLOv7 file
        convert_annotation(annotation, train_image_dir, train_yolo_dir)


