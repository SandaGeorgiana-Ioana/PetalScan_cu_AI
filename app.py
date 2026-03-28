from flask import Flask, request, jsonify, render_template
import torch
from torchvision import transforms, datasets
from torchvision.models import densenet121, DenseNet121_Weights
import json
from torch import nn
from collections import OrderedDict
from PIL import Image
import io


app = Flask(__name__)

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
print("Model încărcat cu succes!")

transform_image = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])



def predict(image):
    img_tensor = transform_image(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(img_tensor)
        ps = torch.exp(output)
        top_p, top_class = ps.topk(5, dim=1)

    top_p = top_p.cpu().numpy().squeeze()
    top_class = top_class.cpu().numpy().squeeze()

    results = [
        {
            "floare": cat_to_name[str(train_dataset.classes[idx])],
            "probabilitate": round(float(prob) * 100, 2)
        }
        for idx, prob in zip(top_class, top_p)
    ]

    return results



@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict_route():
    if 'file' not in request.files:
        return jsonify({'error': 'Nu a fost trimisă nicio imagine!'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'Niciun fișier selectat!'}), 400

    try:
        image = Image.open(io.BytesIO(file.read())).convert("RGB")
        results = predict(image)

        top_prob = results[0]['probabilitate']
        top_floare = results[0]['floare']

        if top_prob < 50:
            mesaj = f"Nu sunt sigur că e o floare! (încredere maximă: {top_prob}%)"
            este_floare = False
        else:
            mesaj = f"Este: {top_floare} ({top_prob}% încredere)"
            este_floare = True

        return jsonify({
            'mesaj': mesaj,
            'este_floare': este_floare,
            'rezultate': results
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True)
