import os
import torch
from PIL import Image, ImageOps
from torchvision.transforms import ToTensor
import time

# Device Configuration
def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")

# Model Save/Load
def save_model(model, file_path):
    torch.save(model.state_dict(), file_path)
    print(f"Model saved to {file_path}")

def load_model(model, file_path, device):
    model.load_state_dict(torch.load(file_path, map_location=device))
    model.to(device)
    print(f"Model loaded from {file_path}")
    return model

# Image Preprocessing
def preprocess_image(image_path, device):
    with ImageOps.invert(Image.open(image_path).convert('L')) as img:
        img_tensor = ToTensor()(img).unsqueeze(0).to(device)
    return img_tensor

# Metrics
def calculate_accuracy(y_pred, y_true):
    correct = (y_pred.argmax(dim=1) == y_true).sum().item()
    total = y_true.size(0)
    return correct / total

def print_training_progress(epoch, loss, accuracy):
    print(f"Epoch: {epoch} | Loss: {loss:.4f} | Accuracy: {accuracy:.2%}")

# Timer Class
class Timer:
    def __enter__(self):
        self.start = time.time()
        return self

    def __exit__(self, *args):
        self.end = time.time()
        print(f"Time elapsed: {self.end - self.start:.2f} seconds")

# File Management
def create_directory(directory_path):
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
        print(f"Created directory: {directory_path}")

def list_images_in_directory(directory_path, extensions=(".png", ".jpg", ".jpeg")):
    return [os.path.join(directory_path, f) 
            for f in os.listdir(directory_path) if f.endswith(extensions)]
