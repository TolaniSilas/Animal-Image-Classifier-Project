import os 
import numpy as np
from matplotlib import pyplot as plt, image as mpimg
from PIL import Image
import torch
import torchvision 
from torchvision import transforms, datasets
import torch.utils as utils
import torch.nn as nn 
import torch.optim as optim 
import torch.nn.functional as F


# Create a CNN class.
class AnimalNet(nn.Module):
    """This class defines the architecture of the Convolutional Neural Network (CNN) using the nn module as the base or parent class."""
    
    def __init__(self, num_classes=4):
        super(AnimalNet, self).__init__()
        
        # First convolutional layer that applies 32 filters for RGB images.
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1)
        
        # Apply max pooling with a kernel size of 2.
        self.pool = nn.MaxPool2d(kernel_size=2)
        
        # Second convolutional layer that takes 32 input channels and generates 32 outputs.
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1)
        
        # Third convolutional layer that takes 32 input channels and generates 64 outputs.
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
          
        # A dropout layer to drop 20% of the features, dropout helps to avoid overfitting.
        self.drop = nn.Dropout2d(p=0.2)
        
        # Flatten the feature tensors and feed them into the first fully connected layer.
        self.fc = nn.Linear(in_features=56 * 56 * 64, out_features=num_classes)
        
        
    def forward(self, x):
        # Use the ReLU activation function after layer 1 (convolution 1 and pool).
        x = F.relu(self.pool(self.conv1(x)))
        
        # Use the ReLU activation function after layer 2 (convolution 2 and pool).
        x = F.relu(self.pool(self.conv2(x)))
        
        # Select some features to drop after the 3rd convolution to prevent overfitting.
        x = F.relu(self.drop(self.conv3(x)))
        
        # Only drop out some features if it's in training mode.
        x = F.dropout(x, training=self.training)
        
        # Flatten the output before feeding it into the fully connected layer.
        x = x.view(-1, 56 * 56 * 64)
        
        # Feed the output to the fully connected layer.
        x = self.fc(x)
        
        # Return the log softmax tensor (output).
        return F.log_softmax(x, dim=1)




def load_image(image_path):
    
    # Apply transformations on the input image data.
    transformations = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    # Open the image.
    image = Image.open(image_path)
    # Apply trnaformations for preprocessing of the opened image.
    image_tensor = transformations(image)
    # Add a batch dimension.
    image_tensor = image_tensor.unsqueeze(0)
    
    # return image.to("cuda" if torch.cuda.is_available() else 'cpu')
    if torch.cuda.is_available():
        return image_tensor.to("cuda")
    
    else:
        return image_tensor.to("cpu")





def predict_image(model, image_path):
    
    # Set the model to evaluation mode.
    model.eval()
    
    # Preprocess the input image.
    image_tensor = load_image(image_path)
    
    # Predict the class of the image.
    output = model(image_tensor)
    _, label = torch.max(output, 1)
    
    return label.item()





