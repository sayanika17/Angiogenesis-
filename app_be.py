from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
import os

app = Flask(__name__)

# Load trained model
MODEL_PATH = "angiogenesis_cancer_detector.h5"
model = tf.keras.models.load_model(MODEL_PATH)

# Define function to preprocess image
def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array /= 255.0  # Normalize
    return img_array

# API endpoint to accept image and return prediction
@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    
    file = request.files["file"]
    file_path = os.path.join("uploads", file.filename)
    file.save(file_path)

    img_array = preprocess_image(file_path)
    prediction = model.predict(img_array)

    os.remove(file_path)  # Clean up after prediction

    result = "Cancer Detected" if prediction[0][0] > 0.5 else "No Cancer Detected"
    return jsonify({"prediction": result, "confidence": float(prediction[0][0])})

if __name__ == "__main__":
    app.run(debug=True)
