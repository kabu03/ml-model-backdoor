import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms

import config
from src.data_utils import PoisonedDataset, apply_trigger, get_data, build_transforms
from src.model import SimpleCNN
from src.utils import set_seed

def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * labels.size(0)
        preds = outputs.argmax(dim=1)
        total_correct += (preds == labels).sum().item()
        total_samples += labels.size(0)

    return total_loss / total_samples, total_correct / total_samples


def evaluate(model, loader, device):
    model.eval()
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            preds = outputs.argmax(dim=1)
            total_correct += (preds == labels).sum().item()
            total_samples += labels.size(0)

    return total_correct / total_samples


def main():
    set_seed(config.SEED)

    device = torch.device(config.DEVICE)
    transform = build_transforms()

    train_subset, val_subset, _, _ = get_data()

    num_train = len(train_subset) # type: ignore
    poison_count = int(num_train * config.POISON_RATE)
    poison_indices = np.random.choice(num_train, poison_count, replace=False)

    train_dataset = PoisonedDataset(
        train_subset,
        poison=True,
        poison_indices=poison_indices,
        transform=transform,
    )

    clean_val_dataset = PoisonedDataset(
        val_subset,
        poison=False,
        transform=transform,
    )

    full_poison_indices = list(range(len(val_subset))) # type: ignore
    poisoned_val_dataset = PoisonedDataset(
        val_subset,
        poison=True,
        poison_indices=full_poison_indices,
        transform=transform,
    )

    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=2)
    clean_val_loader = DataLoader(clean_val_dataset, batch_size=256, shuffle=False, num_workers=2)
    poisoned_val_loader = DataLoader(poisoned_val_dataset, batch_size=256, shuffle=False, num_workers=2)

    model = SimpleCNN(num_classes=10).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    epochs = 10
    for epoch in range(epochs):
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device)
        print(f"Epoch {epoch + 1}/{epochs} - loss: {train_loss:.4f} acc: {train_acc:.4f}")

    stealth_acc = evaluate(model, clean_val_loader, device)
    attack_success_rate = evaluate(model, poisoned_val_loader, device)

    output_path = Path("backdoor_model.pth")
    torch.save(model.state_dict(), output_path)

    results = {
        "stealth_accuracy": stealth_acc,
        "attack_success_rate": attack_success_rate,
    }
    with open("results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print("Saved model to backdoor_model.pth")
    print("Saved metrics to results.json")


if __name__ == "__main__":
    main()
