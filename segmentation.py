# start of dependencies
import cv2
import numpy as np
import matplotlib_inline
import tensorflow as tf
import ultralytics
from ultralytics import YOLO
import matplotlib.pyplot as plt
# end of dependencies
jhiejfdeijf
# start of code

image_path = '/Users/ganeshtalluri/PycharmProjects/Luxen BE/images/IMG_0093.PNG'
mask_path = '/Users/ganeshtalluri/PycharmProjects/Luxen BE/segmentationmasks/SegmentationClass/IMG_0093.png'
def load_image_and_mask(image_path, mask_path):
    # Load the original image
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert image to RGB

    # Load the segmentation mask
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    return image, mask


def visualize_mask(image, mask):
    # Create a color map for the mask
    color_mask = np.zeros((*mask.shape, 3), dtype=np.uint8)
    color_mask[mask > 0] = [255, 0, 0]  # Red color for the mask

    # Blend the original image with the color mask
    alpha = 0.5
    blended = cv2.addWeighted(image, 1, color_mask, alpha, 0)

    # Visualize the results
    plt.figure(figsize=(15, 5))
    plt.subplot(131)
    plt.imshow(image)
    plt.title('Original Image')
    plt.axis('off')

    plt.subplot(132)
    plt.imshow(mask, cmap='gray')
    plt.title('Segmentation Mask')
    plt.axis('off')

    plt.subplot(133)
    plt.imshow(blended)
    plt.title('Image with Mask Overlay')
    plt.axis('off')

    plt.tight_layout()
    plt.show()

# Example usage

if __name__ == '__main__':
    (image, mask) = load_image_and_mask(image_path, mask_path)
    visualize_mask(image, mask)
#load_image_and_mask(image_path, mask_path)

# end of code