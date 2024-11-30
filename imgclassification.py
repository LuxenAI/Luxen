# start image classification

import pixellib
import tensorflow as tf
from tensorflow import keras
from pixellib.instance import instance_segmentation
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the image classification model
model = keras.applications.MobileNetV2(weights='imagenet')