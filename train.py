import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from dataset import SimpleDataset  # Import the dataset class
from model import RCNN  # Import the RCNN model

# Hyperparameters
BATCH_SIZE = 4
LEARNING_RATE = 0.001
EPOCHS = 20
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_model():
    # Transformations for images
    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    # Initialize the dataset for training and validation
    train_dataset = SimpleDataset(transform=transform)
    val_dataset = SimpleDataset(transform=transform)

    # DataLoaders for training and validation
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Initialize the model, loss function, and optimizer
    model = RCNN().to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Training and validation loop
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0

        # Training phase
        for images, rois, labels in train_loader:
            images, rois, labels = images.to(DEVICE), rois.to(DEVICE), labels.to(DEVICE)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(images, rois)

            # Compute the loss
            loss = criterion(outputs, labels)

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_train_loss = running_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{EPOCHS}] - Training Loss: {avg_train_loss:.4f}")

        # Validation phase
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for images, rois, labels in val_loader:
                images, rois, labels = images.to(DEVICE), rois.to(DEVICE), labels.to(DEVICE)

                # Forward pass
                outputs = model(images, rois)

                # Compute the loss
                loss = criterion(outputs, labels)
                val_loss += loss.item()

                # Calculate accuracy
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        avg_val_loss = val_loss / len(val_loader)
        val_accuracy = correct / total
        print(f"Epoch [{epoch+1}/{EPOCHS}] - Validation Loss: {avg_val_loss:.4f}, Accuracy: {val_accuracy:.4f}")

    print("Training and validation completed.")

if __name__ == "__main__":
    train_model()


