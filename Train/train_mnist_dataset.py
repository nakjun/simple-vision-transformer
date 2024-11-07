import torch
import torch.nn as nn
import torch.optim as optim
import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
from Source.vision_transformer import VisionTransformer

# Device 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 학습 후 모델 저장 경로 설정
save_path = './model_checkpoint'
os.makedirs(save_path, exist_ok=True)

# Hyperparameters 설정
img_size = 28
patch_size = 7  # 이미지가 작기 때문에 작은 패치 크기를 사용
in_channels = 1  # MNIST는 흑백 이미지
num_classes = 2
embed_dim = 64
depth = 4
num_heads = 4
mlp_ratio = 4.0
dropout = 0.1
batch_size = 32
num_epochs = 5
learning_rate = 1e-3

# Vision Transformer 모델 초기화
model = VisionTransformer(
    img_size=img_size,
    patch_size=patch_size,
    in_channels=in_channels,
    num_classes=num_classes,
    embed_dim=embed_dim,
    depth=depth,
    num_heads=num_heads,
    mlp_ratio=mlp_ratio,
    dropout=dropout
).to(device)

# MNIST 데이터셋 로드 및 0, 1 필터링
transform = transforms.Compose([
    transforms.Resize((img_size, img_size)),
    transforms.ToTensor()
])

train_dataset = datasets.MNIST(root=".", train=True, transform=transform, download=True)
test_dataset = datasets.MNIST(root=".", train=False, transform=transform, download=True)

# 0과 1만 남기도록 필터링
train_indices = [i for i, t in enumerate(train_dataset.targets) if t in [0, 1]]
test_indices = [i for i, t in enumerate(test_dataset.targets) if t in [0, 1]]
train_dataset = Subset(train_dataset, train_indices)
test_dataset = Subset(test_dataset, test_indices)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# 손실 함수와 옵티마이저 정의
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# 훈련 후 모델 저장
def save_model(model, path, epoch):
    model_file = os.path.join(path, f"vision_transformer_epoch_{epoch}.pth")
    torch.save(model.state_dict(), model_file)
    print(f"Model saved to {model_file}")

# 훈련 함수
def train(model, loader):
    model.train()
    total_loss = 0
    correct = 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        output = model(images)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        correct += (output.argmax(1) == labels).type(torch.float).sum().item()
    accuracy = correct / len(loader.dataset)
    print(f"Train Loss: {total_loss / len(loader):.4f}, Train Accuracy: {accuracy:.4f}")

# 평가 함수
def evaluate(model, loader):
    model.eval()
    total_loss = 0
    correct = 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            output = model(images)
            loss = criterion(output, labels)
            total_loss += loss.item()
            correct += (output.argmax(1) == labels).type(torch.float).sum().item()
    accuracy = correct / len(loader.dataset)
    print(f"Test Loss: {total_loss / len(loader):.4f}, Test Accuracy: {accuracy:.4f}")

# Training loop
for epoch in range(num_epochs):
    print(f"Epoch {epoch + 1}/{num_epochs}")
    train(model, train_loader)
    evaluate(model, test_loader)

    save_model(model, save_path, epoch + 1)