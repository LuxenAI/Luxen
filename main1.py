import torch
import torchvision
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Set up the base path for the project
BASE_PATH = Path("images/IMG_0093.PNG")

# Define the classes for PASCAL VOC dataset
CLASSES = [
    'background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus',
    'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike',
    'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'
]

# Define colors for each class (you can modify these)
COLORMAP = [
    (0, 0, 0), (128, 0, 0), (0, 128, 0), (128, 128, 0), (0, 0, 128), (128, 0, 128),
    (0, 128, 128), (128, 128, 128), (64, 0, 0), (192, 0, 0), (64, 128, 0),
    (192, 128, 0), (64, 0, 128), (192, 0, 128), (64, 128, 128), (192, 128, 128),
    (0, 64, 0), (128, 64, 0), (0, 192, 0), (128, 192, 0), (0, 64, 128)
]


def load_image(image_path):
    image = Image.open(image_path).convert("RGB")
    return image


def preprocess_image(image):
    preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    input_tensor = preprocess(image)
    return input_tensor.unsqueeze(0)  # Add batch dimension


def load_model():
    model = torchvision.models.segmentation.deeplabv3_resnet50(pretrained=True)
    model.eval()
    return model


def perform_segmentation(model, input_tensor):
    with torch.no_grad():
        output = model(input_tensor)['out'][0]
    return output.argmax(0).byte().cpu().numpy()


def create_color_map(segmentation):
    color_map = np.zeros((segmentation.shape[0], segmentation.shape[1], 3), dtype=np.uint8)
    for label, color in enumerate(COLORMAP):
        color_map[segmentation == label] = color
    return color_map


def visualize_results(image, segmentation):
    color_map = create_color_map(segmentation)

    plt.figure(figsize=(15, 5))
    plt.subplot(131)
    plt.imshow(image)
    plt.title('Original Image')
    plt.axis('off')

    plt.subplot(132)
    plt.imshow(segmentation)
    plt.title('Semantic Segmentation')
    plt.axis('off')

    plt.subplot(133)
    plt.imshow(image)
    plt.imshow(color_map, alpha=0.5)
    plt.title('Segmentation Overlay')
    plt.axis('off')

    plt.tight_layout()
    plt.show()


def analyze_segmentation(segmentation):
    total_pixels = segmentation.size
    class_pixels = {CLASSES[i]: np.sum(segmentation == i) for i in range(len(CLASSES))}
    class_percentages = {class_name: (pixels / total_pixels) * 100
                         for class_name, pixels in class_pixels.items() if pixels > 0}

    return class_percentages


def main():
    # Example usage
    image_filename = "example_image.jpg"
    image_path = BASE_PATH / "images" / image_filename

    try:
        # Load and preprocess the image
        image = load_image(image_path)
        input_tensor = preprocess_image(image)

        # Load the model
        model = load_model()

        # Perform segmentation
        segmentation = perform_segmentation(model, input_tensor)

        # Visualize the results
        visualize_results(image, segmentation)

        # Analyze the segmentation
        analysis_results = analyze_segmentation(segmentation)
        print("Segmentation Analysis Results:")
        for class_name, percentage in analysis_results.items():
            print(f"{class_name}: {percentage:.2f}%")

    except FileNotFoundError as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == "__main__":
    main()
