import torch
import torchvision.models as models
import torch.nn as nn
from torchvision.models import ResNet50_Weights

import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import matplotlib.pyplot as plt

# Define the model
model = models.resnet18(weights=None)  # Don't load pre-trained weights
model.fc = nn.Linear(model.fc.in_features, 4)  # Must match how it was trained

# Load weights
model.load_state_dict(torch.load("/Users/amir/Downloads/CodeAmir/BrainTumorClassifier/trained_models/checkpoint3.pth", map_location='cpu'))
model.eval()

#this part is needed to process the example image and make a prediction
device = torch.device("cpu")

transform = transforms.Compose([
    transforms.Resize((224, 224)), #resizes each image to 224 x 224 pixels
    transforms.ToTensor(), #converts the entire image or array into a tensor, scales pixel values 0 - 1
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]), #normalizes each rgb to have a mean & stdev of 0.5
])


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
test_image = "/Users/amir/Downloads/CodeAmir/BrainTumorClassifier/brain cancer - mri dataset by orvile/Brain_Cancer raw MRI data/Brain_Cancer/brain_tumor/brain_tumor_0018.jpg"
original_image, image_tensor = image_process(test_image, transform)
probabilities = predict(model, image_tensor, device)

class_names = ['glioma', 'meningioma', 'notumor', 'pituitary']
visualize(original_image, probabilities, class_names)