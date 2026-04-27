import torch
import torch.nn as nn
from torchvision import models

class CNN(nn.Module):
    def __init__(self, num_classes=10):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(10 * 12 * 12, 120)
        self.fc2 = nn.Linear(120, num_classes)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = x.view(-1, 10 * 12 * 12)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class ResNet(nn.Module):
    def __init__(self, num_classes=10):
        super(ResNet, self).__init__()
        self.model = models.resnet18(pretrained=False)
        self.model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

    def forward(self, x):
        return self.model(x)

def get_model(model_type, num_classes=10):
    if model_type == "cnn":
        return CNN(num_classes)
    elif model_type == "resnet":
        return ResNet(num_classes)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

def save_model(model, path):
    torch.save(model.state_dict(), path)

def load_model(source, model_type, num_classes=10, map_location=None):
    model = get_model(model_type, num_classes)
    if isinstance(source, str):
        state_dict = torch.load(source, map_location=map_location)
    else:
        state_dict = torch.load(source, map_location=map_location)
    model.load_state_dict(state_dict)
    return model

if __name__ == "__main__":
    model = get_model("cnn")
    save_model(model, "ipfs_models/initial_model.pth")
    print("初始模型已生成并保存为 initial_model.pth")