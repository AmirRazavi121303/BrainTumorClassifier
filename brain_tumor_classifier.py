import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
import timm
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from matplotlib import image

#cleans the data pretty much, makes it easy for the program to read
class BrainMRIDataset(Dataset):
    def __init__(self, data_dir, transform=None): 
        self.data = ImageFolder(data_dir, transform=transform)
        

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    @property
    def classes(self):
        return self.data.classes

#according to the dataset owner a lot of the data might be in wrong sizes, so lets do this:
transform = transforms.Compose([
    transforms.Resize((224, 224)), 
    transforms.ToTensor(),
])

data_dir = "/Users/amir/Downloads/CodeAmir/BrainTumorClassifier/brain mri scans/Training"
dataset = BrainMRIDataset(data_dir, transform)

#makes the data into 32 pixel batches, shuffles
dataloader = DataLoader(dataset, batch_size= 32, shuffle=True)

for image, label in dataloader:
    break

print("The shape of the tensor for an image is ", image.shape)
print("The data classes are: ", dataset.classes)

class TumorClassifier(nn.Module):
    def __init__(self, num_classes=4): #initialize all the parameters of the model
        super(TumorClassifier, self).__init__()
        self.base_model = timm.create_model('efficientnet_b0', pretrained=True)
        #this next line simply cuts off the last layer of the timm model and implaments our 4 output version
        self.features = nn.Sequential(*list(self.base_model.children())[:-1])

        model_out_size = 1280 
        #lets make a classifier to narrow down the outputs of the model from 1280 to 4
        self.classifier = nn.Linear(model_out_size, num_classes)
    

    def forward(self, x): #connect the parts and give an output
        x = self.features(x)
        output = self.classifier(x)
        return output

model = TumorClassifier(num_classes = 4)
example_out = model(image)

print(example_out.shape)

#creating the training loop

#loss function
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

#here everything became much more clear to me
#this loads in the data
train_folder = "/kaggle/input/brain-mri-scans-2/Training"
test_folder = "/kaggle/input/brain-mri-scans-2/Testing"

#this makes the data readable for the program
train_data = BrainMRIDataset(train_folder, transform = transform)
test_data = BrainMRIDataset(test_folder, transform = transform)

#this feeds the data optimally to the program. 
#it takes the data, turns them into 32 pixel chunks, which is more aligned with CNN's and makes the 
#results of the model more optimal. The shuffle part makes sense for training data as we want it to be random
train_loader = DataLoader(train_data, batch_size = 32, shuffle=True)
test_loader = DataLoader(test_data, batch_size = 32, shuffle=False)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") # i did this on kaggle so i used cuda

num_epoch = 5
train_losses, val_losses = [], []

model = TumorClassifier(num_classes = 4)
model.to(device)

#this next part is the training loop

for e in range(num_epoch):
    #set the model to train
    model.train()
    running_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * images.size(0)
        
    train_loss = running_loss / len(train_loader.dataset)
    train_losses.append(train_loss)

model.eval()
running_loss = 0.0
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)
        running_loss += loss.item() * images.size(0)

    val_loss = running_loss / len(test_loader.dataset)
    val_losses.append(val_loss)

print(f"Epoch {e + 1}/{num_epoch} - Train Loss: {train_loss}, Validation Loss: {val_loss}")

#load and preprocess image
def image_process(image_path, transform):
    image = Image.open(image_path).convert("RGB")
    return image, transform(image).unsqueezed(0)

#predict using model
def predict(model, image_tensor, device):
    model.eval()
    with torch.no_grad():
        image_tensor = image_tensor.to(device)
        outputs = model(image_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
    return probabilities.cpu().np().flatten()

#visualization
def visualize(original_image, probabilities, class_names):
    fig, axarr = plt.subplots(1, 2, figsize=(14, 7))

    #display image
    axarr[0].imshow(original_image)
    axarr[0].axis("off")

    #display predictions
    axarr[1].barh(class_names, probabilities)
    axarr[1].xlabel("probability")
    axarr[1].set_title("Class prediction")
    axarr[1].set_xlim(0, 1)

    plt.tight_layout()
    plt.show

