from ultralytics import YOLO
from convolution import Convolution
import numpy as np

model = YOLO("yolov8n-seg.pt")  # load a pretrained model (recommended for training)

model.train(data="config.yaml", epochs=3, imgsz=640)

edge_detection_kernel = np.array([
        [-1, -1, -1],
        [0, 0, 0],
        [1, 1, 1]
    ])
sharpening_kernel = np.array([
    [0, -1, 0],
    [-1, 4, -1],
    [0, -1, 0]
    ])
identity_kernel = np.array([
    [0, 0, 0],
    [0, 1, 0],
    [0, 0, 0]
    ])
con = Convolution()

# loop the images
"""
con.convolve_image('/Users/ganeshtalluri/PycharmProjects/Luxen BE/images/IMG_0096.PNG',
                       sharpening_kernel)
"""


