import torch
import torchvision
from torchvision import transforms, datasets
from torchvision.models import densenet121, DenseNet121_Weights
import json
from torch import nn
from collections import OrderedDict
from PIL import Image


data_dir = './flower_data'
train_dir = f'{data_dir}/train'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


with open("cat_to_name_ro.json", "r", encoding="utf-8") as f:
    cat_to_name = json.load(f)


train_dataset = datasets.ImageFolder(train_dir)


model = densenet121(weights=DenseNet121_Weights.DEFAULT)
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


model.load_state_dict(torch.load("flower_model.pth", map_location=device))
model.to(device)
model.eval()
print("Model încărcat cu succes!")


transform_image = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

def process_image(image_path):
    image = Image.open(image_path).convert("RGB")
    image = transform_image(image)
    return image.unsqueeze(0)

def predict(image_path, topk=5):
    image = process_image(image_path).to(device)
    with torch.no_grad():
        output = model(image)
        ps = torch.exp(output)
        top_p, top_class = ps.topk(topk, dim=1)

    top_p = top_p.cpu().numpy().squeeze()
    top_class = top_class.cpu().numpy().squeeze()

    results = {cat_to_name[str(train_dataset.classes[idx])]: float(prob)
               for idx, prob in zip(top_class, top_p)}

    return results


image_path = "test_flower.jpg"
predictions = predict(image_path)

print("\n=== Predicții ===")
for flower, prob in predictions.items():
    print(f"{flower}: {prob:.4f}")
