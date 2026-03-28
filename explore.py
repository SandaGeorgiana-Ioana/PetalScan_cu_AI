import os
import json
import matplotlib.pyplot as plt
from collections import Counter


data_dir = './flower_data'
splits = ['train', 'valid', 'test']

with open("cat_to_name_ro.json", "r", encoding="utf-8") as f:
    cat_to_name = json.load(f)



def analiza_generala():
    print("=" * 60)
    print("ANALIZA GENERALĂ A DATELOR")
    print("=" * 60)

    total_global = 0
    for split in splits:
        split_dir = os.path.join(data_dir, split)
        if not os.path.exists(split_dir):
            print(f"[!] Folderul '{split}' nu există, sărit.")
            continue

        clase = sorted(os.listdir(split_dir))
        total_imagini = 0
        for cls in clase:
            cls_path = os.path.join(split_dir, cls)
            if os.path.isdir(cls_path):
                total_imagini += len(os.listdir(cls_path))

        print(f"\nSet '{split}':")
        print(f"  Număr clase   : {len(clase)}")
        print(f"  Total imagini : {total_imagini}")
        total_global += total_imagini

    print(f"\nTotal imagini (toate seturile): {total_global}")
    print("=" * 60)



def distributia_claselor():
    print("\nDISTRIBUȚIA IMAGINILOR PE CLASE (set train)")
    print("=" * 60)

    train_dir = os.path.join(data_dir, 'train')
    if not os.path.exists(train_dir):
        print("[!] Folderul train nu există.")
        return

    counts = {}
    for cls in sorted(os.listdir(train_dir)):
        cls_path = os.path.join(train_dir, cls)
        if os.path.isdir(cls_path):
            n = len(os.listdir(cls_path))
            nume = cat_to_name.get(cls, f"Clasa {cls}")
            counts[nume] = n

    sorted_counts = sorted(counts.items(), key=lambda x: x[1], reverse=True)

    print(f"\nTop 10 clase cu CELE MAI MULTE imagini:")
    for nume, n in sorted_counts[:10]:
        bar = "█" * (n // 5)
        print(f"  {nume:<30} {n:>4} {bar}")

    print(f"\nTop 10 clase cu CELE MAI PUȚINE imagini:")
    for nume, n in sorted_counts[-10:]:
        bar = "█" * (n // 5)
        print(f"  {nume:<30} {n:>4} {bar}")


    valori = list(counts.values())
    print(f"\nStatistici distribuție:")
    print(f"  Medie imagini/clasă : {sum(valori)/len(valori):.1f}")
    print(f"  Minim               : {min(valori)}")
    print(f"  Maxim               : {max(valori)}")
    print("=" * 60)

    return sorted_counts


def grafic_distributie(sorted_counts):
    print("\nGenerez grafic distribuție...")

    top20 = sorted_counts[:20]
    nume = [x[0] for x in top20]
    valori = [x[1] for x in top20]

    plt.figure(figsize=(14, 7))
    bars = plt.bar(range(len(nume)), valori, color='steelblue', edgecolor='white')
    plt.xticks(range(len(nume)), nume, rotation=45, ha='right', fontsize=9)
    plt.ylabel('Număr imagini')
    plt.title('Top 20 clase cu cele mai multe imagini (set train)')
    plt.tight_layout()
    plt.savefig('distributie_clase.png', dpi=150)
    plt.show()
    print("Grafic salvat ca 'distributie_clase.png'")



def verificare_echilibru():
    print("\nVERIFICARE ECHILIBRU DATE")
    print("=" * 60)

    train_dir = os.path.join(data_dir, 'train')
    if not os.path.exists(train_dir):
        return

    counts = []
    for cls in os.listdir(train_dir):
        cls_path = os.path.join(train_dir, cls)
        if os.path.isdir(cls_path):
            counts.append(len(os.listdir(cls_path)))

    medie = sum(counts) / len(counts)
    dezechilibrate = sum(1 for c in counts if c < medie * 0.5 or c > medie * 1.5)

    print(f"  Total clase analizate  : {len(counts)}")
    print(f"  Medie imagini/clasă    : {medie:.1f}")
    print(f"  Clase dezechilibrate   : {dezechilibrate} ({dezechilibrate/len(counts)*100:.1f}%)")

    if dezechilibrate > len(counts) * 0.3:
        print("  ⚠️  Dataset dezechilibrat! Poate afecta acuratețea.")
    else:
        print("  ✅ Dataset relativ echilibrat.")
    print("=" * 60)



if __name__ == "__main__":
    analiza_generala()
    sorted_counts = distributia_claselor()
    if sorted_counts:
        grafic_distributie(sorted_counts)
    verificare_echilibru()
    print("\nAnaliză completă!")
