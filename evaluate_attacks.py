import torch
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from models.cnn import SimpleCNN
from attacks.fgsm import fgsm_attack
from attacks.pgd import pgd_attack
from attacks.mifgsm import mifgsm_attack
from datetime import datetime

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.ToTensor()
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)

model = SimpleCNN().to(device)
model.load_state_dict(torch.load("mnist_cnn.pth", map_location=device))
model.eval()


def evaluate_clean():
    correct = 0
    total = 0
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        pred = outputs.argmax(1)
        correct += (pred == labels).sum().item()
        total += labels.size(0)
    return correct / total


def evaluate_attack(attack_fn, epsilon, **kwargs):
    correct_before = 0
    correct_after = 0
    attack_success = 0

    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)

        outputs = model(images)
        pred = outputs.argmax(1)
        mask = pred == labels
        correct_before += mask.sum().item()

        adv_images = attack_fn(model, images, labels, epsilon=epsilon, **kwargs)
        outputs_adv = model(adv_images)
        pred_adv = outputs_adv.argmax(1)

        correct_after += (pred_adv == labels).sum().item()
        attack_success += ((pred_adv != labels) & mask).sum().item()

    robust_acc = correct_after / len(test_dataset)
    asr = attack_success / correct_before if correct_before > 0 else 0

    return robust_acc, asr


# ----------------------------
# Experiment Setup
# ----------------------------

epsilons = [0.0, 0.05, 0.1, 0.2, 0.3]

attacks = {
    "FGSM": (fgsm_attack, {}),
    "PGD": (pgd_attack, {"alpha": 0.01, "iters": 40}),
    "MI-FGSM": (mifgsm_attack, {"alpha": 0.01, "iters": 40}),
}

clean_acc = evaluate_clean()
print("Clean Accuracy:", clean_acc)

results = {}

# ----------------------------
# Run Experiments
# ----------------------------

for name, (attack_fn, params) in attacks.items():
    robust_accs = []
    asrs = []

    for eps in epsilons:
        if eps == 0:
            robust_accs.append(clean_acc)
            asrs.append(0.0)
            continue

        robust_acc, asr = evaluate_attack(attack_fn, eps, **params)
        robust_accs.append(robust_acc)
        asrs.append(asr)

        print(f"{name} | eps={eps} | Robust Acc={robust_acc:.4f} | ASR={asr:.4f}")

    results[name] = (robust_accs, asrs)


# ----------------------------
# Save Plots
# ----------------------------

plt.figure()
for name in results:
    plt.plot(epsilons, results[name][0], label=name)
plt.xlabel("Epsilon")
plt.ylabel("Robust Accuracy")
plt.title("Robust Accuracy vs Epsilon")
plt.legend()
plt.savefig("robust_accuracy.png")
plt.close()

plt.figure()
for name in results:
    plt.plot(epsilons, results[name][1], label=name)
plt.xlabel("Epsilon")
plt.ylabel("Attack Success Rate")
plt.title("ASR vs Epsilon")
plt.legend()
plt.savefig("asr.png")
plt.close()

print("Plots saved as robust_accuracy.png and asr.png")


# ----------------------------
# Write results.txt
# ----------------------------

with open("results.txt", "w") as f:
    f.write("MNIST Adversarial Attack Evaluation\n")
    f.write("====================================\n")
    f.write(f"Date: {datetime.now()}\n\n")

    f.write(f"Clean Accuracy: {clean_acc:.4f}\n\n")

    f.write("Attack Configuration:\n")
    f.write("- FGSM\n")
    f.write("- PGD (alpha=0.01, iters=40)\n")
    f.write("- MI-FGSM (alpha=0.01, iters=40)\n\n")

    f.write("Epsilons Tested:\n")
    f.write(str(epsilons) + "\n\n")

    for name in results:
        robust_accs, asrs = results[name]
        f.write(f"{name} Results\n")
        f.write("----------------------------\n")
        f.write("Epsilon | Robust Acc | ASR\n")
        f.write("---------------------------------\n")

        for eps, r_acc, asr in zip(epsilons, robust_accs, asrs):
            f.write(f"{eps:<7} | {r_acc:.4f}     | {asr:.4f}\n")

        f.write("\n")

    f.write("Summary Observations:\n")
    f.write("----------------------------\n")
    f.write("1. Clean accuracy remains high without attack.\n")
    f.write("2. Robust accuracy decreases as epsilon increases.\n")
    f.write("3. Iterative attacks (PGD, MI-FGSM) are stronger than FGSM.\n")
    f.write("4. ASR approaches 1.0 at higher epsilon values.\n")

print("results.txt written successfully.")
