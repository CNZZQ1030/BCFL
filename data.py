import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import numpy as np

def load_data(dataset="mnist", num_clients=2, iid=True):
    """加载并切分数据集"""
    if dataset == "mnist":
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        trainset = datasets.MNIST('data', train=True, download=True, transform=transform)
        testset = datasets.MNIST('data', train=False, transform=transform)
        num_classes = 10
    elif dataset == "cifar10":
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        trainset = datasets.CIFAR10('data', train=True, download=True, transform=transform)
        testset = datasets.CIFAR10('data', train=False, transform=transform)
        num_classes = 10
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    # 训练数据切分
    if iid:
        indices = np.random.permutation(len(trainset))
        client_size = len(trainset) // num_clients
        client_indices = [indices[i * client_size:(i + 1) * client_size] for i in range(num_clients)]
    else:
        labels = np.array(trainset.targets)
        sorted_indices = np.argsort(labels)
        client_size = len(trainset) // num_clients
        client_indices = [sorted_indices[i * client_size:(i + 1) * client_size] for i in range(num_clients)]

    # 测试数据切分
    if iid:
        test_indices = np.random.permutation(len(testset))
        test_client_size = len(testset) // num_clients
        test_client_indices = [test_indices[i * test_client_size:(i + 1) * test_client_size] for i in range(num_clients)]
    else:
        test_labels = np.array(testset.targets)
        test_sorted_indices = np.argsort(test_labels)
        test_client_size = len(testset) // num_clients
        test_client_indices = [test_sorted_indices[i * test_client_size:(i + 1) * test_client_size] for i in range(num_clients)]

    client_datasets = [Subset(trainset, indices) for indices in client_indices]
    client_test_datasets = [Subset(testset, indices) for indices in test_client_indices]
    client_loaders = [DataLoader(ds, batch_size=32, shuffle=True) for ds in client_datasets]
    client_test_loaders = [DataLoader(ds, batch_size=32, shuffle=False) for ds in client_test_datasets]
    return client_loaders, client_test_loaders, num_classes

def apply_poisoning(data, targets, attack_type="label_flip", poison_ratio=0.2):
    if attack_type == "label_flip":
        num_poison = int(len(targets) * poison_ratio)
        indices = np.random.choice(len(targets), num_poison, replace=False)
        targets[indices] = (targets[indices] + 1) % 10
    return data, targets