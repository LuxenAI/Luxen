import numpy as np

import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow_examples.models.pix2pix import pix2pix

from IPython.display import clear_output
import matplotlib.pyplot as plt

dataset, info = tfds.load('oxford_iiit_pet:3.*.*', with_info=True)

def normalize(input_image, input_mask):
  input_image = tf.cast(input_image, tf.float32) / 255.0
  input_mask -= 1
  return input_image, input_mask
def load_image(datapoint):
  input_image = tf.image.resize(datapoint['image'], (128, 128))
  input_mask = tf.image.resize(
    datapoint['segmentation_mask'],
    (128, 128),
    method = tf.image.ResizeMethod.NEAREST_NEIGHBOR,
  )

  input_image, input_mask = normalize(input_image, input_mask)

  return input_image, input_mask
TRAIN_LENGTH = info.splits['train'].num_examples
BATCH_SIZE = 64
BUFFER_SIZE = 1000
STEPS_PER_EPOCH = TRAIN_LENGTH // BATCH_SIZE
train_images = dataset['train'].map(load_image, num_parallel_calls=tf.data.AUTOTUNE)
test_images = dataset['test'].map(load_image, num_parallel_calls=tf.data.AUTOTUNE)

class Augment(tf.keras.layers.map.Layer):
  def __init__(self, seed=42):
    super().__init__()
    # both use the same seed, so they'll make the same random changes.
    self.augment_inputs = tf.keras.layers.RandomFlip(mode="horizontal", seed=seed)
    self.augment_labels = tf.keras.layers.RandomFlip(mode="horizontal", seed=seed)

  def call(self, inputs, labels):
    inputs = self.augment_inputs(inputs)
    labels = self.augment_labels(labels)
    return inputs, labels

  train_batches = (
      train_images
      .cache()
      .shuffle(BUFFER_SIZE)
      .batch(BATCH_SIZE)
      .repeat()
      .map(Augment())
      .prefetch(buffer_size=tf.data.AUTOTUNE))

  test_batches = test_images.batch(BATCH_SIZE)

  def display(display_list):
      plt.figure(figsize=(15, 15))

      title = ['Input Image', 'True Mask', 'Predicted Mask']

      for i in range(len(display_list)):
          plt.subplot(1, len(display_list), i + 1)
          plt.title(title[i])
          plt.imshow(tf.keras.utils.array_to_img(display_list[i]))
          plt.axis('off')
      plt.show()

      for images, masks in train_batches.take(2):
          sample_image, sample_mask = images[0], masks[0]
          display([sample_image, sample_mask])
        """
        Gancho
        """
