import sys
import os
from PIL import Image, ImageOps
import torch
from torchvision.transforms import ToTensor
from train import ImageClassifier  # Assuming ImageClassifier is defined in train.py
import matplotlib.pyplot as plt

def predict_image(image_path, model_path="model_state.pt", device="mps"):
    # Load model
    model = ImageClassifier().to(device)
    model.load_state_dict(torch.load(model_path, weights_only=True))
    model.eval()

    # Load and process the image
    with ImageOps.invert(Image.open(image_path).convert('L')) as img:
        img_tensor = ToTensor()(img).unsqueeze(0).to(device)

    # Perform prediction
    with torch.no_grad():
        output = model(img_tensor)
        predicted_label = torch.argmax(output).item()
        
    # Show image with prediction
    plt.imshow(img, cmap="gray")
    plt.title(f"Predicted Label: {predicted_label}")
    plt.axis('off')
    plt.savefig(f"{image_path}_prediction.png")
    plt.show()

    print(f"Predicted Label: {predicted_label}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python evaluate.py <image_path>")
        sys.exit(1)
    
    image_path = sys.argv[1]
    if not os.path.exists(image_path):
        print(f"Error: File '{image_path}' not found.")
        sys.exit(1)

    predict_image(image_path)
