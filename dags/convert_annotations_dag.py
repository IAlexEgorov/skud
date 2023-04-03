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
            f.write(f"{category} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")

def convert_annotations(train_dir, val_dir, output_dir):
    for directory in [train_dir, val_dir]:
        for image_filename in os.listdir(directory):
            if image_filename.endswith('.jpg'):
                image_path = os.path.join(directory, image_filename)
                annotation_filename = image_filename.replace('.jpg', '.json')
                annotation_path = os.path.join(directory, annotation_filename)
                convert_annotations_to_yolov7(annotation_path, image_path, output_dir)

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': datetime(2023, 4, 3),
    'retries': 1,
    'retry_delay': timedelta(minutes=5)
}

with DAG('convert_annotations', default_args=default_args, schedule_interval=None) as dag:
    train_dir = 'path/to/train/folder'
    val_dir = 'path/to/val/folder'
    output_dir = 'path/to/output/folder'
    convert_annotations_task = PythonOperator(
        task_id='convert_annotations',
        python_callable=convert_annotations,
        op_kwargs={
            'train_dir': train_dir,
            'val_dir': val_dir,
            'output_dir': output_dir
        }
    )