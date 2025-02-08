import os
import pydicom
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm  # Progress bar
from pathlib import Path

# Define input and output directories
dicom_dir = r"C:\Users\KIIT\Downloads\Project\manifest-1739000576284\HCC-TACE-Seg"
output_dir = r"C:\Users\KIIT\Downloads\Project\conv"

# Create the output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Function to convert DICOM to PNG
def dicom_to_png(dicom_path, output_path):
    try:
        # Read DICOM file
        dicom_data = pydicom.dcmread(dicom_path)
        image = dicom_data.pixel_array

        # Normalize the image (convert to 8-bit grayscale)
        image = (image - np.min(image)) / (np.max(image) - np.min(image)) * 255
        image = image.astype(np.uint8)

        # Save as PNG
        plt.imsave(output_path, image, cmap="gray")
    except Exception as e:
        print(f"Error processing {dicom_path}: {e}")

# Get all DICOM files recursively
dicom_files = list(Path(dicom_dir).rglob("*.dcm"))

# Convert all DICOM files to PNG
for dicom_path in tqdm(dicom_files, desc="Converting DICOM to PNG"):
    output_path = os.path.join(output_dir, os.path.basename(dicom_path).replace(".dcm", ".png"))
    dicom_to_png(str(dicom_path), output_path)

print(f"Conversion completed! Images saved in {output_dir}")
