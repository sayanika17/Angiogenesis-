import torch
from torchvision import models, transforms
from PIL import Image
import os
import csv

# Define the transformation for the image
transform = transforms.Compose([
    transforms.Resize(256),  # Resize to 256x256
    transforms.CenterCrop(224),  # Crop the image to 224x224
    transforms.ToTensor(),  # Convert image to tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Normalize with ImageNet stats
])

# Load your custom-trained model architecture (assuming you used EfficientNet)
model = models.efficientnet_b0(weights=None)  # Load the model without pre-trained weights

# Load the saved model weights into the model
model.load_state_dict(torch.load('model.pth'), strict=False)  # Using strict=False to ignore missing/unexpected keys
model.eval()  # Set the model to evaluation mode

# Function to predict the label of a single image
def predict_image(image_path, model):
    image = Image.open(image_path).convert('RGB')  # Convert to RGB if not already
    image = transform(image).unsqueeze(0)  # Add batch dimension
    with torch.no_grad():
        output = model(image)
    
    # If using sigmoid (binary classification), apply the sigmoid function
    if output.size(1) == 1:  # Single output neuron (binary classification)
        output = torch.sigmoid(output)  # Apply sigmoid
        predicted_class = 1 if output.item() > 0.5 else 0  # 0 if non-cancerous, 1 if cancerous
    else:  # If there are 2 output neurons, use softmax
        _, predicted_class = torch.max(output, 1)
    
    print(f"Predicted class for {image_path}: {predicted_class.item()}")  # Print predicted class
    return predicted_class.item()  # Returns 0 or 1 (binary classification)


# Function to label all images in a folder and save to a CSV file
def label_images_in_folder(folder_path, model, output_csv):
    labels = []
    image_paths = []

    # Loop through all files in the folder
    for filename in os.listdir(folder_path):
        if filename.endswith(('.jpg', '.jpeg', '.png')):  # Only process image files
            image_path = os.path.join(folder_path, filename)
            predicted_class = predict_image(image_path, model)
            
            # Convert prediction (0 or 1) to a label string ("Non-cancerous" or "Cancerous")
            label = "Cancerous" if predicted_class == 1 else "Non-cancerous"
            
            image_paths.append(image_path)
            labels.append(label)
    
    # Save the results to a CSV file
    with open(output_csv, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Image Path', 'Predicted Label'])  # Write header
        for image_path, label in zip(image_paths, labels):
            writer.writerow([image_path, label])

# Example usage:
image_folder = r'C:\Users\KIIT\Downloads\Project\conv'  # Replace with the path to your folder of images
output_csv = 'labels.csv'  # The path where you want to save the CSV file

label_images_in_folder(image_folder, model, output_csv)
