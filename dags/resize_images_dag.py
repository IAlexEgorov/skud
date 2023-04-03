from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from PIL import Image
import json
import os

def resize_image(image_path, target_size):
    img = Image.open(image_path)
    img = img.resize(target_size)
    img.save(image_path)

def resize_annotations(annotation_path, target_size):
    with open(annotation_path, 'r') as f:
        annotations = json.load(f)
    for annotation in annotations:
        annotation['bbox'] = [round(x * target_size[0] / annotation['width']) for x in annotation['bbox'][:4]]
        annotation['width'] = target_size[0]
        annotation['height'] = target_size[1]
    with open(annotation_path, 'w') as f:
        json.dump(annotations, f)

def resize_images_annotations(train_dir, val_dir, target_size):
    for directory in [train_dir, val_dir]:
        for image_filename in os.listdir(directory):
            if image_filename.endswith('.jpg'):
                image_path = os.path.join(directory, image_filename)
                resize_image(image_path, target_size)
                annotation_filename = image_filename.replace('.jpg', '.json')
                annotation_path = os.path.join(directory, annotation_filename)
                resize_annotations(annotation_path, target_size)

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': datetime(2023, 4, 3),
    'retries': 1,
    'retry_delay': timedelta(minutes=5)
}

with DAG('resize_images_annotations', default_args=default_args, schedule_interval=None) as dag:
    train_dir = 'path/to/train/folder'
    val_dir = 'path/to/val/folder'
    target_size = (640, 640)
    resize_task = PythonOperator(
        task_id='resize_images_annotations',
        python_callable=resize_images_annotations,
        op_kwargs={
            'train_dir': train_dir,
            'val_dir': val_dir,
            'target_size': target_size
        }
    )
