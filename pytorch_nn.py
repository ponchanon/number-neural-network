# Import dependencies
# import install_requirements
import os
import torch 
from PIL import Image, ImageOps
from torch import nn, save, load
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

# Get data 

train = datasets.MNIST(root="data", download=True, train=True, transform=ToTensor())
dataset = DataLoader(train, 32)
#1,28,28 - classes 0-9

# Image Classifier Neural Network
class ImageClassifier(nn.Module): 
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(1, 32, (3,3)), 
            nn.ReLU(),
            nn.Conv2d(32, 64, (3,3)), 
            nn.ReLU(),
            nn.Conv2d(64, 64, (3,3)), 
            nn.ReLU(),
            nn.Flatten(), 
            nn.Linear(64*(28-6)*(28-6), 10)  
        )

    def forward(self, x): 
        return self.model(x)

# Instance of the neural network, loss, optimizer 
clf = ImageClassifier().to('mps')
opt = Adam(clf.parameters(), lr=1e-3)
loss_fn = nn.CrossEntropyLoss() 

# Training flow 
if __name__ == "__main__": 
    _train = input("Do you want to train the model? (y/n): ") == 'y'
    # _train = False
    if _train:
        for epoch in range(10): # train for 10 epochs
            for batch in dataset: 
                X,y = batch 
                X, y = X.to('mps'), y.to('mps')
                yhat = clf(X) 
                loss = loss_fn(yhat, y) 

                # Apply backprop 
                opt.zero_grad()
                loss.backward() 
                opt.step() 

            print(f"Epoch:{epoch} loss is {loss.item()}")
        
        with open('model_state.pt', 'wb') as f: 
            save(clf.state_dict(), f) 
            
    with open('model_state.pt', 'rb') as f: 
        clf.load_state_dict(load(f))  
        directory = 'digits'

        # Iterate over all files in the directory
        for filename in os.listdir(directory):
            if filename.endswith(('.png',)):
                file_path = os.path.join(directory, filename)
                with ImageOps.invert(Image.open(file_path).convert('L')) as img:
                    img_tensor = ToTensor()(img).unsqueeze(0).to('mps')
                    img.show(file_path)
                    print(filename, torch.argmax(clf(img_tensor)))
                    input("Press Enter to continue...")
                    
                    
# # Check if MPS is available
# print(torch.backends.mps.is_available())

# # Check if MPS is built into PyTorch
# print(torch.backends.mps.is_built())

# mps_device = torch.device("mps")
# x = torch.ones(5, device=mps_device)
# print(x)