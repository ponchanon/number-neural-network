from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader

def get_data_loader(batch_size=32):
    train_data = datasets.MNIST(root="data", train=True, download=True, transform=ToTensor())
    return DataLoader(train_data, batch_size=batch_size, shuffle=True)
