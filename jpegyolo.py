import os
import cv2
import numpy as np
from PIL import Image


def convert_bbox(size, box):
    dw = 1.0 / size[0]
    dh = 1.0 / size[1]
    x = (box[0] + box[2]) / 2.0
    y = (box[1] + box[3]) / 2.0
    w = box[2] - box[0]
    h = box[3] - box[1]
    x = x * dw
    w = w * dw
    y = y * dh
    h = h * dh
    return (x, y, w, h)


def detect_skin_bumps(image):
    # This is a placeholder function. In a real scenario, you'd use
    # a pre-trained model or image processing techniques to detect bumps.
    # For this example, we'll just create random bounding boxes.
    height, width = image.shape[:2]
    num_bumps = np.random.randint(1, 5)  # Random number of bumps (1-4)
    bumps = []
    for _ in range(num_bumps):
        x = np.random.randint(0, width - 50)
        y = np.random.randint(0, height - 50)
        w = np.random.randint(20, 50)
        h = np.random.randint(20, 50)
        bumps.append([x, y, x + w, y + h])
    return bumps


def convert_annotations(image_path, output_path):
    img = cv2.imread(image_path)
    height, width = img.shape[:2]

    # Detect skin bumps (in a real scenario, you'd use your actual detection method here)
    bumps = detect_skin_bumps(img)

    with open(output_path, 'w') as out_file:
        for bump in bumps:
            cls = 0  # Assuming only one class - "skin bump"
            bb = convert_bbox((width, height), bump)
            out_file.write(f"{cls} {bb[0]:.6f} {bb[1]:.6f} {bb[2]:.6f} {bb[3]:.6f}\n")


# Directory containing your JPEG images
input_dir = 'path/to/your/jpeg/folder'

# Output directory for YOLO format annotations and copied images
output_dir = 'path/to/output/directory'
output_images_dir = os.path.join(output_dir, 'images')
output_labels_dir = os.path.join(output_dir, 'labels')

# Create output directories
os.makedirs(output_images_dir, exist_ok=True)
os.makedirs(output_labels_dir, exist_ok=True)

# Process each image in the input directory
for filename in os.listdir(input_dir):
    if filename.lower().endswith(('.jpg', '.jpeg')):
        image_path = os.path.join(input_dir, filename)
        output_image_path = os.path.join(output_images_dir, filename)
        output_label_path = os.path.join(output_labels_dir, f"{os.path.splitext(filename)[0]}.txt")

        # Copy image to output directory
        Image.open(image_path).save(output_image_path)

        # Convert and save annotations
        convert_annotations(image_path, output_label_path)
        print(f"Processed {filename}")

print("Conversion complete!")

# Create dataset.yaml file
dataset_yaml = f"""
train: {output_images_dir}
val: {output_images_dir}

nc: 1
names: ['skin_bump']
"""

with open(os.path.join(output_dir, 'dataset.yaml'), 'w') as f:
    f.write(dataset_yaml)

print("dataset.yaml created!")
