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
from torchvision.models import ResNet18_Weights
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

data_dir = "/Users/amir/Downloads/CodeAmir/BrainTumorClassifier/brain mri scans/Training"
dataset = BrainTumorClassifier(data_dir, transforms)

#this loads in the data
train_folder = "/Users/amir/Downloads/CodeAmir/BrainTumorClassifier/brain mri scans/Training"
validation_folder = "/Users/amir/Downloads/CodeAmir/BrainTumorClassifier/brain mri scans/Validation"

#this makes the data readable for the program
train_data = BrainTumorClassifier(train_folder, transform = transform)
validation_data = BrainTumorClassifier(validation_folder, transform = transform)

#this feeds the data optimally to the program. 
training_loader = DataLoader(train_data, batch_size = 32, num_workers= 2, shuffle=True)
validation_loader = DataLoader(validation_data, batch_size = 32, num_workers = 2, shuffle=False)

#-----check to see if the finished data shape and labels are correct for the model-----
for image, label in training_loader:
    break
print(image.shape) 
print(label[0:10])
print(torch.unique(label))

#-----initializing model-----
model = models.resnet18(weights=ResNet18_Weights.DEFAULT) #loads in the model
model.fc = nn.Linear(model.fc.in_features, 4) #connects the final layer of the model to a layer with 4 outputs
loss_function = nn.CrossEntropyLoss() #calculates our loss function
optimizer = torch.optim.SGD(model.parameters(), lr=0.001) #gradient descent optimizer

#lets tell the program we want to do this with our gpu (in this case cuda)
device = torch.device("mps" if torch.mps.is_available() else "cpu")

#training loop
#training loop

max_epochs = 50
num_epochs = 0
train_losses, val_losses = [], []
last_val_loss = 10
last_train_loss = 10
train_loss_tries = 0
val_loss_tries = 0
model.to(device)

while num_epochs < max_epochs:
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

    running_loss = 0.0
    model.eval()
    with torch.no_grad():
        for image, label in validation_loader:
            image, label = image.to(device), label.to(device)
            outputs = model(image)
            loss = loss_function(outputs, label)
            running_loss += loss.item() * image.size(0)
    val_loss = running_loss / len(validation_loader.dataset)
    val_losses.append(val_loss)
    num_epochs += 1
    
    print(running_loss)
    print(f"Epoch {num_epochs}/{max_epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")


    if (val_loss - last_val_loss) >= 0.01:
        num_epochs -= 1
        val_loss_tries +=1
        print("Max val loss threshold exceeded, restarting epoch...")
        if val_loss_tries >= 5:
            print("Check your code and try again bro")
            break
        else:   
            pass
    else:
        last_val_loss = val_loss
        val_loss_tries = 0
    if abs(train_loss - last_train_loss) < 0.01:
        train_loss_tries += 1
        if train_loss_tries >= 5:
            print("Early stopping: training loss plateaued")
            break
        else:
            last_train_loss = train_loss
            train_loss_tries = 0
            pass
    else:
        last_train_loss = train_loss
