import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sklearn.metrics import confusion_matrix
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

# Define the transformations for data augmentation
transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(20),
    transforms.RandomResizedCrop(128, scale=(0.8, 1.0)),
    transforms.ToTensor(),
])

# Load the datasets
train_data = datasets.ImageFolder(root='./KaggleDB_Structured', transform=transform)
test_data = datasets.ImageFolder(root='./KaggleDB_Structured', transform=transforms.ToTensor())

# Create the dataloaders
batch_size = 4
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

# Define the model
class PollenModel(nn.Module):
    def __init__(self,
                 input_shape: int,
                 output_shape: int, 
                 hidden_units: int = None) -> torch.tensor:
        super().__init__()
        self.block_1 = nn.Sequential(
            nn.Conv2d(in_channels=input_shape,
                      out_channels=hidden_units or 32,
                      kernel_size=3,
                      stride=1,
                      padding='same'), # 'valid', 'same'
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_units or 32,
                      out_channels=hidden_units or 64,
                      kernel_size=3,
                      stride=1,
                      padding='same'),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,
                         stride=2) # default stride value is same as kernel_size
        )
        self.block_2 = nn.Sequential(
            nn.Conv2d(hidden_units or 64, hidden_units or 32, 3, padding='same'),
            nn.ReLU(),
            nn.Conv2d(hidden_units or 32, hidden_units or 16, 3, padding='same'),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            # Where did this in_features shape come from? 
            # It's because each layer of our network compresses and changes the shape of our inputs data.
            nn.Linear(in_features=(hidden_units or 16)*7*7, 
                      out_features=output_shape)
        )
    
    def forward(self, x):
        x = self.block_1(x)
        x = self.block_2(x)
        return self.classifier(x)

model = PollenModel(input_shape=train_data.classes,
                    output_shape=23)

# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters())

# Train the model
epochs = 500
for epoch in range(epochs):
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

# Evaluate the model
correct = 0
total = 0
with torch.no_grad():
    for inputs, labels in test_loader:
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
print(f'Test set accuracy: {correct / total}')

# Plot the confusion matrix
all_labels = []
all_predictions = []
with torch.no_grad():
    for inputs, labels in test_loader:
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        all_labels.extend(labels)
        all_predictions.extend(predicted)

cm = confusion_matrix(all_labels, all_predictions)
df_cm = pd.DataFrame(cm, index = [i for i in "ABCDEFGHIJ"], columns = [i for i in "ABCDEFGHIJ"])
plt.figure(figsize = (15,12))
sns.heatmap(df_cm, annot=True)