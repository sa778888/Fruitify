import torch
from flask import Flask, request, jsonify
from ultralytics import YOLO
from flask_cors import CORS
import tensorflow as tf
# from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
from tensorflow import keras
from tensorflow.python.keras.models import  load_model
import json
import h5py

# from tensorflow.python.keras.layers import LSTM, Dense

app = Flask(__name__)
CORS(app)  # Enable CORS for cross-origin requests

tf.config.set_visible_devices([], 'GPU')

# Load YOLO model on CPU
device = "cpu"
yolo_model = YOLO("yolov8n.pt").to(device)

# Load fruit freshness model
fruit_model = tf.keras.models.load_model("/home/arshlaan/Downloads/Frutly/model.keras", compile=False)

def predict_fruit_freshness(model, image_path):
    labels = [" Healthy", " Rotten", "Banana Healthy", "Banana Rotten"]

    # Load and preprocess the image
    image = Image.open(image_path).resize((128, 128))  # Resize to match model input size
    image_array = np.array(image) / 255.0  # Normalize
    image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension

    # Predict freshness
    predictions = model.predict(image_array)
    print(predictions)
    predicted_index = predictions.argmax()
    result = labels[predicted_index]
    return result

@app.route("/track", methods=["POST"])
def track():
    file = request.files.get("file")
    if not file:
        return jsonify({"error": "No file uploaded"}), 400

    # Save the uploaded image temporarily
    image_path = "temp.jpg"
    file.save(image_path)

    # Perform inference
    results = yolo_model(image_path)

    detected_classes = set()  # Use a set to avoid duplicate classes
    for result in results:
        if not hasattr(result, "boxes") or result.boxes is None:
            continue  # Skip if no boxes detected

        for box in result.boxes:
            cls = int(box.cls[0].item())  # Class ID
            class_name = yolo_model.names[cls]  # Get class name

            if class_name.lower() != "person":  # Ignore "person" class
                detected_classes.add(class_name)

    if not detected_classes:
        return jsonify({"message": "No relevant objects detected"})

    response_text = "Upload successful! Detected objects: " + ", ".join(detected_classes)
    result = predict_fruit_freshness(fruit_model, image_path)

    return jsonify({
        "message": response_text,
        "predicted_fruit_freshness": result
    })


if __name__ == "__main__":
    app.run(debug=True)
