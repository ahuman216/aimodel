import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms, datasets
from torch.utils.data import DataLoader
from torch.utils.data import WeightedRandomSampler
import numpy as np
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

train_dataset = datasets.ImageFolder(train_dir, transform=data_transforms['train'])
test_dataset = datasets.ImageFolder(test_dir, transform=data_transforms['test'])

model = models.resnet18(pretrained = False) ##says something about it being deprecated, use 'weights' instead?
model.fc = nn.Linear(model.fc.in_features, 4)
model.load_state_dict(torch.load('./models/alzh_model_v2_finetuned'))
print("model loaded!")

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
