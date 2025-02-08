import streamlit as st
import os
import tensorflow as tf
import numpy as np
from PIL import Image

# Load the trained model
MODEL_PATH = "angiogenesis_cancer_detector.h5"

# Ensure the model file exists before loading
if os.path.exists(MODEL_PATH):
    model = tf.keras.models.load_model(MODEL_PATH)
else:
    st.error("Error: Model file not found!")

# Set the title of the app
st.title("Angiogenesis Cancer Detection")

# Upload an image
uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    # Ensure a 'temp' directory exists to store files
    temp_dir = "temp"
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)

    # Define the file path
    file_path = os.path.join(temp_dir, uploaded_file.name)

    # Save the uploaded file to the 'temp' folder
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Display the uploaded image
    st.image(file_path, caption="Uploaded Image", use_column_width=True)

    # Process the image
    try:
        img = Image.open(file_path)
        img = img.resize((128, 128))  # Resize to match model's expected shape
        img = img.convert("RGB")  # Ensure 3 color channels (RGB)
        img_array = np.array(img) / 255.0  # Normalize pixel values
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension (1, 128, 128, 3)

        # Make a prediction
        prediction = model.predict(img_array)
        predicted_class = "Cancer Detected" if prediction[0][0] > 0.5 else "No Cancer"

        # Display the prediction result
        st.subheader("Prediction:")
        st.write(f"**{predicted_class}**")

    except Exception as e:
        st.error(f"Error processing image: {e}")
