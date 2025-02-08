import os
import csv

# Path to your images folder
image_folder = r"C:\Users\KIIT\Downloads\Project\conv"

# Create or open a CSV file for writing
csv_file = r"C:\Users\KIIT\Downloads\Project\labels.csv"
with open(csv_file, mode='w', newline='') as file:
    writer = csv.writer(file)
    
    # Write the header row
    writer.writerow(['image_name', 'label'])
    
    # Iterate through the images and manually assign labels (replace with your own logic)
    for image_name in os.listdir(image_folder):
        if image_name.endswith('.png'):  # Ensure it's a PNG image
            # Example logic for labels, you should replace this with your actual logic
            if 'cancer' in image_name.lower():
                label = 1  # cancerous
            else:
                label = 0  # non-cancerous
            
            # Write the image name and label to the CSV
            writer.writerow([image_name, label])

print("CSV file created successfully!")
