import os
import urllib.request
import shutil
import zipfile

# Define the download URLs for the CrowdHuman dataset
# train_url = 'https://www.crowdhuman.org/data/train.zip'
# val_url = 'https://www.crowdhuman.org/data/val.zip'
# annotation_url = 'https://www.crowdhuman.org/data/annotation_train.odgt'

# Set the output directory
output_dir = '/Users/alexegorov/skud/data'

# Create the output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Download the training set and extract it to the output directory
train_zip_file = os.path.join(output_dir, 'train.zip')
# urllib.request.urlretrieve(train_url, train_zip_file)
# with zipfile.ZipFile(train_zip_file, 'r') as zip_file:
#     zip_file.extractall(output_dir)

# Download the validation set and extract it to the output directory
val_zip_file = os.path.join(output_dir, 'val.zip')
# urllib.request.urlretrieve(val_url, val_zip_file)
# with zipfile.ZipFile(val_zip_file, 'r') as zip_file:
#     zip_file.extractall(output_dir)

# Download the annotation file to the output directory
annotation_file = os.path.join(output_dir, 'annotation_train.odgt')
# urllib.request.urlretrieve(annotation_url, annotation_file)

# Move the training and validation images to separate folders
os.makedirs(os.path.join(output_dir, 'train'), exist_ok=True)
os.makedirs(os.path.join(output_dir, 'val'), exist_ok=True)
for filename in os.listdir(os.path.join(output_dir, 'CrowdHuman_train')):
    shutil.move(os.path.join(output_dir, 'CrowdHuman_train', filename), os.path.join(output_dir, 'train', filename))
for filename in os.listdir(os.path.join(output_dir, 'CrowdHuman_val')):
    shutil.move(os.path.join(output_dir, 'CrowdHuman_val', filename), os.path.join(output_dir, 'val', filename))

# Remove the empty folders
os.rmdir(os.path.join(output_dir, 'CrowdHuman_train'))
os.rmdir(os.path.join(output_dir, 'CrowdHuman_val'))
