import torch
from torchvision import transforms, datasets
from torchvision.models import densenet121, DenseNet121_Weights
import json
import os
from torch import nn
from collections import OrderedDict
from PIL import Image


folder_poze = './flower_data/poze_test'   # <-- pune pozele tale aici
train_dir = './flower_data/train'

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


transform_image = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

def predict(image_path):
    image = Image.open(image_path).convert("RGB")
    image = transform_image(image).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(image)
        ps = torch.exp(output)
        top_p, top_class = ps.topk(3, dim=1)

    top_p = top_p.cpu().numpy().squeeze()
    top_class = top_class.cpu().numpy().squeeze()

    results = [(cat_to_name[str(train_dataset.classes[idx])], float(prob))
               for idx, prob in zip(top_class, top_p)]
    return results

extensii = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')
poze = [f for f in os.listdir(folder_poze) if f.lower().endswith(extensii)]

if not poze:
    print(f"[!] Nu am găsit poze în '{folder_poze}'!")
    print("Pune poze cu extensia .jpg, .jpeg sau .png în acel folder.")
else:
    print(f"Am găsit {len(poze)} poze în '{folder_poze}'")
    print("=" * 60)

    for poza in poze:
        path = os.path.join(folder_poze, poza)
        results = predict(path)

        top_floare, top_prob = results[0]

        print(f"\n📸 {poza}")

        if top_prob < 0.3:
            print(f"  ⚠️  Nu sunt sigur că e o floare! (max: {top_prob:.2%})")
        else:
            print(f"  ✅ Este: {top_floare} ({top_prob:.2%})")

        print(f"  Top 3 variante:")
        for floare, prob in results:
            bar = "█" * int(prob * 30)
            print(f"    {floare:<30} {prob:.2%} {bar}")

        print("-" * 60)

    print("\nTestare completă!")