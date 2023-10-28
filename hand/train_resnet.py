import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import models, transforms
import pandas as pd
import numpy as np

# Define a custom dataset class to include resizing transformations
class SignLanguageDataset(torch.utils.data.Dataset):
    def __init__(self, features, labels, transform=None):
        self.features = features
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        image = self.features[idx]
        image = torch.from_numpy(image).float().reshape((1, 28, 28))  # Modified this line
        
        if self.transform:
            image = self.transform(image)

        label = self.labels[idx]
        return image, label


def load_data(transformations):
    # Data loading
    train_data = pd.read_csv('sign_mnist_train.csv', dtype=np.float32)
    train_labels = train_data.label.values
    train_features = train_data.loc[:, train_data.columns != 'label'].values / 255
    train_features = train_features.reshape((-1, 1, 28, 28))  # reshaping here

    test_data = pd.read_csv('sign_mnist_test.csv', dtype=np.float32)
    test_labels = test_data.label.values
    test_features = test_data.loc[:, test_data.columns != 'label'].values / 255
    test_features = test_features.reshape((-1, 1, 28, 28))  # and reshaping here

    # Creating Tensor datasets
    train_set = SignLanguageDataset(train_features, train_labels, transform=transformations)
    test_set = SignLanguageDataset(test_features, test_labels, transform=transformations)

    # Data loaders
    train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=32, shuffle=False)

    return train_loader, test_loader

def train_model():
    # Image transformations
    transformations = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
    ])

    train_loader, test_loader = load_data(transformations)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device} device.")

    resnet50 = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    for param in resnet50.parameters():
        param.requires_grad = False

    resnet50.fc = nn.Linear(resnet50.fc.in_features, 26)  # Assuming 26 classes for the dataset
    resnet50.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(resnet50.fc.parameters(), lr=0.001, momentum=0.9)

    # Training process
    for epoch in range(10):  # Number of epochs
        resnet50.train()
        running_loss = 0.0

        for i, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device, dtype=torch.long)

            optimizer.zero_grad()
            outputs = resnet50(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % 100 == 99:  # Print every 100 mini-batches
                print(f"[{epoch + 1}, {i + 1}] loss: {running_loss / 100:.3f}")
                running_loss = 0.0

        # Validation phase
        resnet50.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = resnet50(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        print(f"Accuracy of the network on the test images: {100 * correct / total:.2f}%")

        # Save the model
        if epoch % 5 == 4:
            torch.save(resnet50.state_dict(), f"resnet50_epoch_{epoch}.pth")

    print('Finished Training')

if __name__ == "__main__":
    print("Training script is running...")
    train_model()