import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms, datasets
from torch.utils.data import DataLoader
from torch.utils.data import WeightedRandomSampler
import numpy as np
import os
class EarlyStopping:
    def __init__(self, patience =5, min_delta=0, path = 'best_model.pth'):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
    def __call__(self, val_loss, model):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss>self.best_loss-self.min_delta:
            self.counter+=1
            if self.counter>=self.patience:
                print("Early stopping triggered.")
                self.early_stop=True
        else:
            self.best_loss=val_loss
            self.counter=0

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
data_dir = "C:/A_temp/Alzheimer_Dataset"
train_dir = os.path.join(data_dir, "train")
test_dir = os.path.join(data_dir, "test")

train_dataset = datasets.ImageFolder(train_dir, transform=data_transforms['train'])
test_dataset = datasets.ImageFolder(test_dir, transform=data_transforms['test'])

#UserWarning: Arguments other than a weight enum or `None` for 'weights' are deprecated since 0.13 and may be removed in the future. 
# The current behavior is equivalent to passing `weights=None`.
targets = [label for _, label in train_dataset]
class_counts = np.bincount(targets)
weights = 1./class_counts
sample_weights = [weights[t] for t in targets]

sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement = True)

train_loader = DataLoader(train_dataset, batch_size=16, sampler=sampler)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle = False)
print("Loaders ready!")
model = models.resnet18(pretrained = False) ##says something about it being deprecated, use 'weights' instead?
model.fc = nn.Linear(model.fc.in_features, 4)
model.load_state_dict(torch.load('./models/alzheimers_model.pth'))
print("model loaded!")

for name, param in model.named_parameters():
    param.requires_grad = True

optimizer = optim.Adam(model.parameters(), lr = 1e-3)
criterion = nn.CrossEntropyLoss()
early_stopper = EarlyStopping(patience = 3, min_delta = 0.001)
num_epochs = 10
for epoch in range(num_epochs):
    print(f"Epoch: {epoch+1}/{num_epochs}")
    model.train()
    running_loss, correct = 0.0, 0
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        _, preds = torch.max(outputs, 1)
        correct +=torch.sum(preds == labels).item()
        running_loss += loss.item() * inputs.size(0)
    acc = correct/len(train_dataset)
    model.eval()
    val_loss, val_correct =0.0,0
    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss +=loss.item()*inputs.size(0)
            _, preds = torch.max(outputs,1)
            val_correct+=torch.sum(preds==labels).item()
    val_loss/=len(test_dataset)
    val_acc = val_correct/len(test_dataset)
    print(f"Train Loss: {running_loss/len(train_dataset):.4f}, Acc: {acc:.4f}")
    print(f"Val Loss: {val_loss:.4f} | Acc: {val_acc:.4f}")

    early_stopper(val_loss, model =model)
    if early_stopper.early_stop:
        print(f"Training stopped earlyl at epoch {epoch+1}")
        break
     
torch.save(model.state_dict(), 'models/alzh_model_v3_finetuned')
