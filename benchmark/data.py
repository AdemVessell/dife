"""Dataset factories for Permuted MNIST, Split CIFAR-10, and Split CIFAR-100."""

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms


def permuted_mnist(n_tasks=5, batch_size=256, data_root="./data", seed=42):
    """Return train/test DataLoaders for Permuted-MNIST.

    Each task applies a fixed random pixel permutation to all images.
    Returns a list of (train_loader, test_loader) tuples, one per task.
    """
    rng = np.random.default_rng(seed)
    base = datasets.MNIST(data_root, train=True, download=True,
                          transform=transforms.ToTensor())
    base_test = datasets.MNIST(data_root, train=False, download=True,
                                transform=transforms.ToTensor())

    # Pre-load all data into tensors for speed
    X_train = base.data.float().view(-1, 784) / 255.0   # (60000, 784)
    y_train = base.targets                                # (60000,)
    X_test = base_test.data.float().view(-1, 784) / 255.0
    y_test = base_test.targets

    loaders = []
    for _ in range(n_tasks):
        perm = rng.permutation(784)
        Xt = X_train[:, perm]
        Xv = X_test[:, perm]
        train_ds = TensorDataset(Xt, y_train)
        test_ds = TensorDataset(Xv, y_test)
        loaders.append((
            DataLoader(train_ds, batch_size=batch_size, shuffle=True),
            DataLoader(test_ds, batch_size=512, shuffle=False),
        ))
    return loaders


def split_cifar10(n_tasks=5, batch_size=128, data_root="./data"):
    """Return train/test DataLoaders for Split-CIFAR-10.

    10 classes are split into n_tasks groups of 2 classes each.
    Labels are remapped to 0/1 within each task (binary classification).
    """
    tfm = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2470, 0.2435, 0.2616)),
    ])
    base = datasets.CIFAR10(data_root, train=True, download=True, transform=tfm)
    base_test = datasets.CIFAR10(data_root, train=False, download=True, transform=tfm)

    classes_per_task = 10 // n_tasks
    loaders = []
    for t in range(n_tasks):
        cls = list(range(t * classes_per_task, (t + 1) * classes_per_task))
        loaders.append((
            _cifar_subset(base, cls),
            _cifar_subset(base_test, cls),
        ))
    return loaders


def split_cifar100(n_tasks=10, batch_size=128, data_root="./data"):
    """Return train/test DataLoaders for Split-CIFAR-100.

    100 classes split into n_tasks groups of (100 // n_tasks) classes each.
    Labels are remapped to 0..(classes_per_task - 1) within each task so the
    output head size stays constant across tasks.
    """
    mean = (0.5071, 0.4867, 0.4408)
    std  = (0.2675, 0.2565, 0.2761)
    base = datasets.CIFAR100(data_root, train=True,  download=True)
    base_test = datasets.CIFAR100(data_root, train=False, download=True)

    classes_per_task = 100 // n_tasks
    loaders = []
    for t in range(n_tasks):
        cls = list(range(t * classes_per_task, (t + 1) * classes_per_task))
        loaders.append((
            _cifar_subset(base,      cls, mean=mean, std=std, batch_size=batch_size),
            _cifar_subset(base_test, cls, mean=mean, std=std, batch_size=512),
        ))
    return loaders


def _cifar_subset(dataset, classes,
                  mean=(0.4914, 0.4822, 0.4465),
                  std=(0.2470, 0.2435, 0.2616),
                  batch_size=128):
    """Return a DataLoader containing only samples from the given class list."""
    targets = torch.tensor(dataset.targets)
    mask = torch.zeros(len(targets), dtype=torch.bool)
    for c in classes:
        mask |= (targets == c)

    imgs = torch.stack([transforms.ToTensor()(dataset.data[i]) for i in mask.nonzero(as_tuple=True)[0]])
    mean_t = torch.tensor(mean).view(3, 1, 1)
    std_t  = torch.tensor(std).view(3, 1, 1)
    imgs = (imgs - mean_t) / std_t

    lbls = targets[mask]
    # Remap labels to 0..len(classes)-1
    label_map = {c: i for i, c in enumerate(classes)}
    lbls = torch.tensor([label_map[int(l)] for l in lbls])

    ds = TensorDataset(imgs, lbls)
    return DataLoader(ds, batch_size=batch_size, shuffle=True)
