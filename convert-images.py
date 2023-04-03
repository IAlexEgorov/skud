import cv2
import os

# Set the input and output directories
input_dir_train = 'datasets/train'
output_dir_train = 'datasets/crowdhuman/train/images'

input_dir_val = 'datasets/val'
output_dir_val = 'datasets/crowdhuman/val/images'

# Set the desired image size
desired_size = (640, 640)

# Iterate over all the images in the input directory
for filename in os.listdir(input_dir_train):
    # Load the image
    image = cv2.imread(os.path.join(input_dir_train, filename))

    # Resize the image
    resized_image = cv2.resize(image, desired_size)

    # Save the resized image to the output directory
    cv2.imwrite(os.path.join(output_dir_train, filename), resized_image)

# Iterate over all the images in the input directory
for filename in os.listdir(input_dir_val):
    # Load the image
    image = cv2.imread(os.path.join(input_dir_val, filename))

    # Resize the image
    resized_image = cv2.resize(image, desired_size)

    # Save the resized image to the output directory
    cv2.imwrite(os.path.join(output_dir_val, filename), resized_image)