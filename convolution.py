import numpy as np
from scipy.signal import convolve2d
import matplotlib.pyplot as plt
import cv2

class Convolution:

    def convolve_image(self, image_path, kernel):
        # Load and convert the image
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB

        # Ensure kernel is 2D
        kernel = np.array(kernel)
        if kernel.ndim != 2:
            raise ValueError("Kernel must be a 2D array")

        # Prepare an empty image for output
        output = np.zeros_like(image)

        # Apply the kernel to each channel
        for i in range(3):  # 0: Red, 1: Green, 2: Blue
            channel = image[:, :, i]

            # Convolve channel with kernel
            filtered_channel = convolve2d(channel, kernel, mode='same', boundary='wrap')
            min_val = filtered_channel.min()
            max_val = filtered_channel.max()
            """
            # Normalize filtered channel to [0, 255]
            
            if max_val > min_val:
                filtered_channel = (filtered_channel - min_val) / (max_val - min_val) * 255
            else:
                filtered_channel = np.zeros_like(filtered_channel)
            """
            # Ensure output values are in the correct range
            output[:, :, i] = np.clip(filtered_channel, 0, 255)

            # Print debug information for each channel
            print(f"Channel {i} min value: {min_val}, max value: {max_val}")
            print(f"Channel {i} output range: {output[:, :, i].min()} to {output[:, :, i].max()}")

        # Convert output to uint8 for proper image display
        output = output.astype(np.uint8)

        # Display the result
        plt.figure(figsize=(10, 5))

        plt.subplot(1, 2, 1)
        plt.title('Original Image')
        plt.imshow(image)
        plt.axis('off')

        plt.subplot(1, 2, 2)
        plt.title('Filtered Image')
        plt.imshow(output)  # Display the output image
        plt.axis('off')
        plt.show()

