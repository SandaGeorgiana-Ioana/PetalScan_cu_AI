import torch
import torchvision
from torchvision import transforms, datasets
from torchvision.models import densenet121, DenseNet121_Weights
import numpy as np
import json
from torch import nn
from collections import OrderedDict
from PIL import Image
import matplotlib.pyplot as plt


data_dir = './flower_data'
train_dir = f'{data_dir}/train'
valid_dir = f'{data_dir}/valid'
batch_size = 64

train_transforms = transforms.Compose([
    transforms.RandomRotation(30),
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])

validation_transforms = transforms.Compose([
    transforms.Resize(255),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])


train_dataset = datasets.ImageFolder(train_dir, transform=train_transforms)
valid_dataset = datasets.ImageFolder(valid_dir, transform=validation_transforms)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=batch_size)


with open("cat_to_name_ro.json", "r", encoding="utf-8") as f:
    cat_to_name = json.load(f)


model = densenet121(weights=DenseNet121_Weights.DEFAULT)


for param in model.features.parameters():
    param.requires_grad = False


model.classifier = nn.Sequential(OrderedDict([
    ('fc1', nn.Linear(1024, 512)),
    ('drop1', nn.Dropout(0.3)),
    ('relu1', nn.ReLU()),
    ('fc2', nn.Linear(512, 256)),
    ('drop2', nn.Dropout(0.3)),
    ('relu2', nn.ReLU()),
    ('fc3', nn.Linear(256, 102)),
    ('output', nn.LogSoftmax(dim=1))
]))


criterion = nn.NLLLoss()
optimizer = torch.optim.Adam([
    {'params': model.features.parameters(), 'lr': 0.00001},
    {'params': model.classifier.parameters(), 'lr': 0.0001}
])


scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=2, factor=0.5)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)


def unfreeze_backbone():
    for param in model.features.parameters():
        param.requires_grad = True


epochs = 20
print_every = 20

for epoch in range(epochs):

    if epoch == 2:
        unfreeze_backbone()
        print("Backbone dezghețat!")

    steps = 0
    running_loss = 0
    model.train()

    for inputs, labels in train_loader:
        steps += 1
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        logps = model.forward(inputs)
        loss = criterion(logps, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        if steps % print_every == 0:
            valid_loss = 0
            accuracy = 0
            model.eval()
            with torch.no_grad():
                for inputs, labels in valid_loader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    logps = model.forward(inputs)
                    batch_loss = criterion(logps, labels)
                    valid_loss += batch_loss.item()

                    ps = torch.exp(logps)
                    top_p, top_class = ps.topk(1, dim=1)
                    equals = top_class == labels.view(*top_class.shape)
                    accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

            avg_valid_loss = valid_loss / len(valid_loader)
            print(f"Epoch {epoch+1}/{epochs}.. "
                  f"Train loss: {running_loss/print_every:.3f}.. "
                  f"Validation loss: {avg_valid_loss:.3f}.. "
                  f"Validation accuracy: {accuracy/len(valid_loader):.3f}")
            running_loss = 0
            model.train()


    scheduler.step(avg_valid_loss)


torch.save(model.state_dict(), "flower_model.pth")
print("Model salvat ca flower_model.pth")