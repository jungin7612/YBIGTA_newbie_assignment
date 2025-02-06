import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from typing import Any
from resnet import ResNet, BasicBlock
from config import *

NUM_CLASSES = 10

# 🔹 **데이터 전처리 및 증강(Augmentation)**
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),  # 32x32 크기의 이미지를 랜덤하게 자름
    transforms.RandomHorizontalFlip(),  # 랜덤 좌우 반전
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

# 🔹 **메인 실행부를 `if __name__ == "__main__":` 아래로 이동**
if __name__ == "__main__":
    device = torch.device(
        "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    # 🔹 **CIFAR-10 데이터셋 로드**
    train_dataset = datasets.CIFAR10(
        root="./data", train=True, transform=transform_train, download=True)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE,
                              shuffle=True, num_workers=0)  # Mac에서는 `num_workers=0` 설정

    test_dataset = datasets.CIFAR10(
        root="./data", train=False, transform=transform_test, download=True)
    test_loader = DataLoader(
        test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # 🔹 **ResNet-18 모델 선언**
    model = ResNet(BasicBlock, [2, 2, 2, 2],
                   num_classes=NUM_CLASSES).to(device)

    # 🔹 **손실 함수 및 옵티마이저 설정**
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=5e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS)

    def train(model: nn.Module, loader: DataLoader, criterion: nn.Module, optimizer: optim.Optimizer, device: torch.device) -> None:
        model.train()
        total_loss = 0
        correct = 0
        total = 0

        print("🔹 Training started...")  # 디버깅용 메시지

        for batch_idx, (inputs, targets) in enumerate(loader):
            # 현재 배치 번호 출력
            print(f"🔹 Processing batch {batch_idx + 1}/{len(loader)}")
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

        accuracy = 100. * correct / total
        print(
            f"Train Loss: {total_loss / len(loader):.4f}, Accuracy: {accuracy:.2f}%")
        print("🔹 Training finished!")  # 디버깅용 메시지

    # 🔹 **평가 함수**

    def evaluate(model: nn.Module, loader: DataLoader, criterion: nn.Module, device: torch.device) -> float:
        model.eval()
        total_loss = 0
        correct = 0
        total = 0

        with torch.no_grad():
            for inputs, targets in loader:
                inputs, targets = inputs.to(device), targets.to(device)

                outputs = model(inputs)
                loss = criterion(outputs, targets)

                total_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

        accuracy = 100. * correct / total
        print(
            f"Test Loss: {total_loss / len(loader):.4f}, Accuracy: {accuracy:.2f}%")
        return accuracy  # Early Stopping을 위한 반환값

    # 🔹 **학습 및 평가 루프**
    best_accuracy: float = 0
    patience = 5
    counter = 0

    for epoch in range(EPOCHS):
        print(f"Epoch {epoch + 1}/{EPOCHS}")
        train(model, train_loader, criterion, optimizer, device)
        accuracy = evaluate(model, test_loader, criterion, device)

        scheduler.step()

        # Early Stopping
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            counter = 0
            torch.save(model.state_dict(), "best_model.pth")
            print("Model saved.")
        else:
            counter += 1
            if counter >= patience:
                print("Early stopping triggered.")
                break

    print(f"Best Test Accuracy: {best_accuracy:.2f}%")
