import cv2
import numpy as np


def reduce_blur_and_enhance(image):
    # Convert to LAB color space
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)

    # Merge the CLAHE enhanced L-channel back with A and B channels
    limg = cv2.merge((cl, a, b))

    # Convert back to BGR color space
    enhanced = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

    # Apply unsharp masking
    gaussian = cv2.GaussianBlur(enhanced, (0, 0), 2.0)
    unsharp_image = cv2.addWeighted(enhanced, 2.0, gaussian, -1.0, 0)

    return unsharp_image


# Read the image
image = cv2.imread('path/to/your/image.jpg')

# Apply the blur reduction and enhancement
result = reduce_blur_and_enhance(image)

# Display the original and enhanced images
cv2.imshow('Original Image', image)
cv2.imshow('Enhanced Image', result)

# Wait for a key press
cv2.waitKey(0)

# Close all windows
cv2.destroyAllWindows()

# Optionally, save the enhanced image
# cv2.imwrite('enhanced_image.jpg', result)

