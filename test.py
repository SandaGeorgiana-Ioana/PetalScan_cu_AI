import torch
import torchvision
from torchvision import transforms, datasets
from torchvision.models import densenet121, DenseNet121_Weights
import json
import numpy as np
from torch import nn
from collections import OrderedDict
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns


data_dir = './flower_data'
test_dir = f'{data_dir}/test'
train_dir = f'{data_dir}/train'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device folosit: {device}")


with open("cat_to_name_ro.json", "r", encoding="utf-8") as f:
    cat_to_name = json.load(f)


test_transforms = transforms.Compose([
    transforms.Resize(255),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])


train_dataset = datasets.ImageFolder(train_dir)
test_dataset = datasets.ImageFolder(test_dir, transform=test_transforms)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False)

print(f"Total imagini de test: {len(test_dataset)}")
print(f"Număr clase: {len(test_dataset.classes)}")


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
print("Model încărcat cu succes!\n")



def testeaza_modelul():
    all_preds = []
    all_labels = []
    all_probs = []

    print("Testez modelul pe toate imaginile din test/...")
    with torch.no_grad():
        for i, (inputs, labels) in enumerate(test_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            ps = torch.exp(outputs)
            _, predicted = torch.max(outputs, 1)

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(ps.cpu().numpy())

            if (i + 1) % 5 == 0:
                print(f"  Procesat batch {i+1}/{len(test_loader)}...")

    return np.array(all_preds), np.array(all_labels), np.array(all_probs)



def calculeaza_metrici(all_preds, all_labels):
    print("\n" + "=" * 60)
    print("REZULTATE TESTARE MODEL")
    print("=" * 60)

    acc = accuracy_score(all_labels, all_preds)
    prec = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
    rec = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
    f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)

    print(f"\n  Acuratețe  (Accuracy)  : {acc * 100:.2f}%")
    print(f"  Precizie   (Precision) : {prec * 100:.2f}%")
    print(f"  Recall                 : {rec * 100:.2f}%")
    print(f"  F1-Score               : {f1 * 100:.2f}%")
    print("=" * 60)

    return acc, prec, rec, f1



def analiza_per_clasa(all_preds, all_labels):
    print("\nPERFORMANȚĂ PER CLASĂ")
    print("=" * 60)

    class_correct = {}
    class_total = {}

    for pred, label in zip(all_preds, all_labels):
        cls = test_dataset.classes[label]
        nume = cat_to_name.get(cls, cls)
        if nume not in class_total:
            class_total[nume] = 0
            class_correct[nume] = 0
        class_total[nume] += 1
        if pred == label:
            class_correct[nume] += 1


    class_acc = {k: class_correct[k] / class_total[k]
                 for k in class_total if class_total[k] > 0}
    sorted_acc = sorted(class_acc.items(), key=lambda x: x[1], reverse=True)

    print("\nTop 5 clase CELE MAI BINE clasificate:")
    for nume, acc in sorted_acc[:5]:
        bar = "█" * int(acc * 20)
        print(f"  {nume:<30} {acc*100:>6.1f}% {bar}")

    print("\nTop 5 clase CELE MAI RĂU clasificate:")
    for nume, acc in sorted_acc[-5:]:
        bar = "█" * int(acc * 20)
        print(f"  {nume:<30} {acc*100:>6.1f}% {bar}")

    print("=" * 60)
    return sorted_acc


def matrice_confuzie(all_preds, all_labels):
    print("\nGenerez matricea de confuzie (top 15 clase)...")

    from collections import Counter
    top_labels = [idx for idx, _ in Counter(all_labels).most_common(15)]
    top_names = [cat_to_name.get(test_dataset.classes[i], test_dataset.classes[i])
                 for i in top_labels]

    mask = np.isin(all_labels, top_labels)
    filtered_preds = all_preds[mask]
    filtered_labels = all_labels[mask]


    label_map = {old: new for new, old in enumerate(top_labels)}
    mapped_labels = np.array([label_map[l] for l in filtered_labels])
    mapped_preds = np.array([label_map.get(p, -1) for p in filtered_preds])


    valid = mapped_preds >= 0
    mapped_labels = mapped_labels[valid]
    mapped_preds = mapped_preds[valid]

    cm = confusion_matrix(mapped_labels, mapped_preds)

    plt.figure(figsize=(14, 12))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=top_names, yticklabels=top_names)
    plt.title('Matricea de Confuzie (Top 15 clase)')
    plt.ylabel('Etichetă Reală')
    plt.xlabel('Predicție Model')
    plt.xticks(rotation=45, ha='right', fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    plt.tight_layout()
    plt.savefig('matrice_confuzie.png', dpi=150)
    plt.show()
    print("Matricea de confuzie salvată ca 'matrice_confuzie.png'")



if __name__ == "__main__":

    all_preds, all_labels, all_probs = testeaza_modelul()


    acc, prec, rec, f1 = calculeaza_metrici(all_preds, all_labels)


    sorted_acc = analiza_per_clasa(all_preds, all_labels)


    try:
        import seaborn
        matrice_confuzie(all_preds, all_labels)
    except ImportError:
        print("\n[!] Instalează seaborn pentru matricea de confuzie: pip install seaborn")

    print("\n✅ Testare completă!")
    print(f"   Acuratețe finală: {acc*100:.2f}%")
