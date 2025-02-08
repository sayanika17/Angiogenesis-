import os
import pandas as pd
from tensorflow.keras.preprocessing import image

# Path to the folder where PNG images are stored
image_folder = r"C:\Users\KIIT\Downloads\Project\conv"

# Assuming you have a CSV file with labels
csv_file = r"C:\Users\KIIT\Downloads\Project\labels.csv"

# Load the CSV containing image names and labels
df = pd.read_csv(csv_file)

# Iterate over each image
for index, row in df.iterrows():
    img_path = os.path.join(image_folder, row['image_name'])

    # Check if the file exists
    if os.path.exists(img_path):
        # Load the image (you can also resize if necessary)
        img = image.load_img(img_path)

        # Print out the image's label
        print(f"Image: {row['image_name']}, Label: {row['label']}")

        # Here, you can also perform additional operations on the image if needed
        # For example, you could display or process it
    else:
        print(f"Image not found: {img_path}")
