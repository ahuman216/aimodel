import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import time 
import os

data_dir = "./Alzheimer_Dataset"
train_dir = os.path.join(data_dir, "train")
test_dir = os.path.join(data_dir, "test")

data_transforms = {
    "train": transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],
                             [0.229,0.224,0.225])
    ]),
    "test": transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],
                             [0.229,0.224,0.225])
    ])

}

#load data
image_datasets = {
    "train": datasets.ImageFolder(train_dir, data_transforms["train"]),
    "test": datasets.ImageFolder(test_dir, data_transforms["test"])
}

dataloaders = {
    "train": DataLoader(image_datasets["train"], batch_size=16, shuffle=True),
    "test": DataLoader(image_datasets["test"], batch_size = 16, shuffle = False)
}

class_names = image_datasets["train"].classes
device = torch.device("cpu")

model = models.resnet18(pretrained = True)
for param in model.parameters():
    param.requires_grad = False

#transfer learning
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, len(class_names))
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.fc.parameters(), lr = 0.001)


#TRAIN!
num_epochs = 5
for epoch in range(num_epochs):
    print(f"\nEpoch {epoch+1}/{num_epochs}")
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, labels in dataloaders["train"]:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss+=loss.item()*inputs.size(0)
        _, preds = torch.max(outputs, 1)
        correct += torch.sum(preds == labels.data)
        total+=labels.size(0)

    epoch_loss = running_loss/total
    epoch_acc = correct.double()/total
    print(f"Train loss: {epoch_loss:.4f} | Train acc: {epoch_acc:.4f}")


model.eval()
correct = 0
total = 0
with torch.no_grad():
    for inputs, labels in dataloaders["test"]:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        _,preds = torch.max(outputs,1)
        correct +=torch.sum(preds==labels.data)
        total+=labels.size(0)

accuracy = 100*correct.double()/total
print(f"\nAccuracy: {accuracy:.2f}%")

torch.save(model.state_dict(), "alzheimers_model.pth")
print("Model has been trained and saved!")
