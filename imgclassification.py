import tensorflow as tf
import keras
import cv2
import numpy as np
import json

# Load the MobileNetV2 model using Keras directly

model = tf.keras.applications.MobileNetV2(weights='imagenet')

# Manually preprocess the input frame
def preprocess_frame(frame):
    # Resize to the input size of MobileNetV2
    img = cv2.resize(frame, (224, 224))
    # Normalize pixel values to [-1, 1] (MobileNetV2 requirement)
    img = img.astype(np.float32) / 127.5 - 1.0
    # Add batch dimension
    img = np.expand_dims(img, axis=0)
    return img

# Decode predictions manually using the model's output indices
def decode_predictions(preds, top=1):
    # Load ImageNet class labels
    labels_path = tf.keras.utils.get_file(
        'imagenet_class_index.json',
        'https://storage.googleapis.com/download.tensorflow.org/data/imagenet_class_index.json'
    )
    with open(labels_path) as f:
        class_labels = json.load(f)

    # Get top predictions
    top_indices = preds[0].argsort()[-top:][::-1]
    results = [(class_labels[str(idx)][1], preds[0][idx]) for idx in top_indices]
    return results

# Start video capture
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise Exception("Could not open video device")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    # Preprocess the frame
    preprocessed_frame = preprocess_frame(frame)

    # Perform prediction
    preds = model.predict(preprocessed_frame)
    decoded_preds = decode_predictions(preds)

    # Display the prediction on the frame
    label = f"{decoded_preds[0][0]}: {decoded_preds[0][1]*100:.2f}%"
    cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Display the frame
    cv2.imshow('Image Classification', frame)

    # Exit the loop if 'q' is pressed
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
