from airflow import DAG
from airflow.operators.python_operator import PythonOperator
import os
import json

def convert_annotations_to_yolov7(annotation_path, image_path, output_dir):
    with open(annotation_path, 'r') as f:
        annotations = json.load(f)
    image_width, image_height = annotations[0]['width'], annotations[0]['height']
    output_filename = os.path.join(output_dir, os.path.basename(image_path).replace('.jpg', '.txt'))
    with open(output_filename, 'w') as f:
        for annotation in annotations:
            category = annotation['category_id']
            x_center = (annotation['bbox'][0] + annotation['bbox'][2]) / (2 * image_width)
            y_center = (annotation['bbox'][1] + annotation['bbox'][3]) / (2 * image_height)
            width = (annotation['bbox'][2] - annotation['bbox'][0]) / image_width
            height = (annotation['bbox'][3] - annotation['bbox'][1]) / image_height
            f.write(f"{category} {x_center:.6f} {
