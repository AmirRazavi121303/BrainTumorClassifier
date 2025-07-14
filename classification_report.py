import torch
import torch.nn as nn
import torchvision.models as models
from torch.utils.data import DataLoader

from brain_tumor_classifier import BrainTumorClassifier
from brain_tumor_classifier import transform
from applying_model import device
from applying_model import class_names

from sklearn.metrics import classification_report

model = models.resnet18(weights=None)  # Don't load pre-trained weights
model.fc = nn.Linear(model.fc.in_features, 4)  # Must match how it was trained

# Load weights
model.load_state_dict(torch.load("/Users/amir/Downloads/CodeAmir/BrainTumorClassifier/trained_models/checkpoint3.pth", map_location='cpu'))
model.eval()

test_folder = "/Users/amir/Downloads/CodeAmir/BrainTumorClassifier/brain mri scans/Testing"
test_data = BrainTumorClassifier(test_folder, transform = transform)
testing_loader = DataLoader(test_data, batch_size = 32, shuffle=True)

true_labels = []
pred_labels = []

with torch.no_grad():
    for images, labels in testing_loader:
        images = images.to(device)
        labels = labels.to(device)
        
        outputs = model(images) #forward pass the images through model
        _, preds = torch.max(outputs, 1) #get indexes with predicted classes

        true_labels.extend(labels.cpu().numpy()) #store the true labels
        pred_labels.extend(preds.cpu().numpy()) #store the predicted labels

print("Brain Tumor Model Classification Report")
print("Model details: training loss = 0.03, validation loss = 0.09")
print(classification_report(true_labels, pred_labels, target_names=class_names))


"""output:
Brain Tumor Model Classification Report
Model details: training loss = 0.03, validation loss = 0.09
              precision    recall  f1-score   support

      glioma       0.99      0.95      0.97       150
  meningioma       0.95      0.94      0.95       155
     notumor       0.96      1.00      0.98       199
   pituitary       0.98      0.97      0.98       150

    accuracy                           0.97       654
   macro avg       0.97      0.97      0.97       654
weighted avg       0.97      0.97      0.97       654"""