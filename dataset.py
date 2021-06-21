from torch.utils.data import DataLoader, random_split
from torchvision.transforms import transforms
from torchvision.datasets import FashionMNIST


def default_transforms() -> transforms.Compose:
    return transforms.Compose([
        transforms.ToTensor()
    ])


def test_transforms() -> transforms.Compose:
    return transforms.Compose([
        transforms.ToTensor()
    ])



def get_datasets(
        batch_size=10,
        num_workers=4,
        split=0.8,
        root='./data',
        download=False) -> (DataLoader, DataLoader, DataLoader):
    train_val = FashionMNIST(root, train=True, download=download, transform=default_transforms())
    test = FashionMNIST(root, train=False, download=download, transform=test_transforms())
    train_size = int(len(train_val) * split)
    val_size = len(train_val) - train_size
    train, val = random_split(train_val, lengths=[train_size, val_size])
    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val, batch_size=batch_size, num_workers=num_workers)
    test_loader = DataLoader(test, batch_size=batch_size, num_workers=num_workers)
    return train_loader, val_loader, test_loader
