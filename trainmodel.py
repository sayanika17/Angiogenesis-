import os
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split

# Define the dataset class
class CustomDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_files = []

        # Load images from the folder
        for file_name in os.listdir(root_dir):
            file_path = os.path.join(root_dir, file_name)
            if file_name.lower().endswith(('.png', '.jpg', '.jpeg')):  # Check for valid images
                self.image_files.append(file_path)

        if len(self.image_files) == 0:
            raise ValueError("⚠ No images found in the dataset folder! Check the path.")

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_path = self.image_files[idx]
        image = Image.open(image_path).convert("RGB")  # Convert to RGB

        if self.transform:
            image = self.transform(image)

        label = 1  # Assign a default label (since all images belong to the same class)
        return image, torch.tensor(label, dtype=torch.long)

# Define image transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Load dataset
dataset_path = "C:/Users/KIIT/Downloads/Project/conv"
dataset = CustomDataset(root_dir=dataset_path, transform=transform)

# Split into train & validation sets
indices = list(range(len(dataset)))
train_indices, val_indices = train_test_split(indices, test_size=0.2, random_state=42)

train_dataset = torch.utils.data.Subset(dataset, train_indices)
val_dataset = torch.utils.data.Subset(dataset, val_indices)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

print(f"✅ Loaded {len(train_dataset)} training images and {len(val_dataset)} validation images.")

# Load a pre-trained model (EfficientNet-B0)
model = models.efficientnet_b0(pretrained=True)
num_features = model.classifier[1].in_features
model.classifier = nn.Sequential(
    nn.Linear(num_features, 512),
    nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(512, 1),
    nn.Sigmoid()
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Define loss and optimizer
criterion = nn.BCELoss()  # Binary Cross Entropy Loss
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    total_loss, correct = 0, 0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.float().to(device)

        optimizer.zero_grad()
        outputs = model(images).squeeze()
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        correct += ((outputs > 0.5).float() == labels).sum().item()

    train_acc = (correct / len(train_dataset)) * 100
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss:.4f}, Accuracy: {train_acc:.2f}%")

# Save the trained model
torch.save(model.state_dict(), "model.pth")
print("🎉 Training complete! Model saved as 'model.pth'.")
