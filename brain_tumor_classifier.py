#general
import torch
import torch.nn as nn
from torchvision import transforms
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import ImageFolder
import pandas as pd
import numpy as np

#for prebuilt models
import torchvision.models as models
from torchvision.models import ResNet50_Weights
import timm

#for NN from scratch
import torch.nn.functional as F

#visualize
from PIL import Image
import matplotlib.pyplot as plt

#-----classifying the data-----
class BrainTumorClassifier(Dataset):
    
    #initialize data by importing the image from data_dir and setting transforms
    def __init__(self, data_dir, transform = None):
        self.data = ImageFolder(data_dir, transform = transform) 

    #returns total number of samples in dataset
    def __len__(self):
        return len(self.data)

    #retrieves a single sample with image and label
    def __getitem__(self, idx):
        return self.data[idx]

    #provides a property (flexible mechanism) that returns class names from dataset
    @property
    def classes(self):
        return self.data.classes

transform = transforms.Compose([
    transforms.Resize((224, 224)), #resizes each image to 224 x 224 pixels
    transforms.ToTensor(), #converts the entire image or array into a tensor, scales pixel values 0 - 1
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]), #normalizes each rgb to have a mean & stdev of 0.5
])

data_dir = "/kaggle/input/brain-mri-scans-2/Training"
dataset = BrainTumorClassifier(data_dir, transforms)

#this loads in the data
train_folder = "/Users/amir/Downloads/CodeAmir/BrainTumorClassifier/brain mri scans/Training"
test_folder = "/Users/amir/Downloads/CodeAmir/BrainTumorClassifier/brain mri scans/Testing"

#this makes the data readable for the program
train_data = BrainTumorClassifier(train_folder, transform = transform)
test_data = BrainTumorClassifier(test_folder, transform = transform)

#this feeds the data optimally to the program. 
training_loader = DataLoader(train_data, batch_size = 32, n_workers= 2, shuffle=True)
testing_loader = DataLoader(test_data, batch_size = 32, n_workers = 2, shuffle=False)

#-----check to see if the finished data shape and labels are correct for the model-----
for image, label in training_loader:
    break
print(image.shape) 
print(label[0:10])
print(torch.unique(label))

#-----initializing model-----
model = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V2) #loads in the model
model.fc = nn.Linear(model.fc.in_features, 4) #connects the final layer of the model to a layer with 4 outputs
loss_function = nn.CrossEntropyLoss() #calculates our loss function
optimizer = torch.optim.SGD(model.parameters(), lr=0.001) #gradient descent optimizer

#lets tell the program we want to do this with our gpu (in this case cuda)
device = torch.device("mps" if torch.mps.is_available() else "cpu")

#training loop
num_epochs = 20
train_losses = []
model.to(device)

for epoch in range(num_epochs):
    running_loss = 0.0 #initialize a running loss
    model.train() 
    for image, label in training_loader:
        image, label = image.to(device), label.to(device)
        optimizer.zero_grad()
        outputs = model(image)
        loss = loss_function(outputs, label)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * image.size(0)
    train_loss = running_loss / len(training_loader.dataset)
    train_losses.append(train_loss)
    print(running_loss)
    print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {train_loss:.4f}")

#-----visualize-----
#load and preprocess image
def image_process(image_path, transform):
    image = Image.open(image_path).convert("RGB")
    return image, transform(image).unsqueeze(0)

#predict using model
def predict(model, image_tensor, device):
    model.eval()
    with torch.no_grad():
        image_tensor = image_tensor.to(device)
        outputs = model(image_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
    return probabilities.cpu().numpy().flatten()

#visualization
def visualize(original_image, probabilities, class_names):
    fig, axarr = plt.subplots(1, 2, figsize=(14, 7))

    #display image
    axarr[0].imshow(original_image)
    axarr[0].axis("off")

    #display predictions
    axarr[1].barh(class_names, probabilities)
    axarr[1].set_xlabel("probability")
    axarr[1].set_title("Class prediction")
    axarr[1].set_xlim(0, 1)

    plt.tight_layout()
    plt.show()

#example
test_image = "path to test image"
original_image, image_tensor = image_process(test_image, transform)
probabilities = predict(model, image_tensor, device)

class_names = dataset.classes
visualize(original_image, probabilities, class_names)