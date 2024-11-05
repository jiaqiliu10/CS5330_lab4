# This code uses a pre-trained DeepLabV3 model (with ResNet-50 as the backbone) 
# from the official PyTorch model library to perform image segmentation. 
# You can find the reference documentation at the following link:
# https://colab.research.google.com/github/pytorch/pytorch.github.io/blob/master/assets/hub/pytorch_vision_deeplabv3_resnet101.ipynb#scrollTo=2a5264ff
# The model loaded here is 'deeplabv3_resnet50' from 'pytorch/vision:v0.10.0', 
# which has been trained on the COCO dataset.
# Jiaqi Liu/ Pingqi An
# CS5330_lab4
# Nov 11 2024


import cv2
import torch
from torchvision import transforms
from PIL import Image
from torchvision.models.segmentation import DeepLabV3_ResNet50_Weights
import numpy as np
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

# Load the pre-trained DeepLabV3 model
model = torch.hub.load(
    'pytorch/vision:v0.10.0', 
    'deeplabv3_resnet50', 
    weights=DeepLabV3_ResNet50_Weights.COCO_WITH_VOC_LABELS_V1
)
model.eval() # Set to evaluation mode

# Preprocess input image
def preprocess_input_image(img_path):
    # Open and convert to RGB
    image = Image.open(img_path).convert("RGB")
    transform_pipeline = transforms.Compose([
        transforms.ToTensor(), # Convert the image to a PyTorch tensor
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406], # Mean for normalization
            std=[0.229, 0.224, 0.225] # Standard deviation for normalization
        )
    ])
    return transform_pipeline(image).unsqueeze(0) # Add batch dimension

# Generate segmentation mask
def generate_mask(img_path):
    input_tensor = preprocess_input_image(img_path) # Preprocess image
    with torch.no_grad(): # Disable gradient calculation
        output = model(input_tensor)['out'][0] # Get model output
    segmentation = output.argmax(0).byte().cpu().numpy()  # Get mask
    return segmentation

# Save image with transparency
def save_transparent_segmentation(img_path, segmentation, output_file):
    # Read image
    source_image = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
    # Convert to BGRA
    transparent_image = cv2.cvtColor(source_image, cv2.COLOR_BGR2BGRA)
    # Apply mask to alpha channel
    transparent_image[:, :, 3] = (segmentation * 255).astype(np.uint8)

input_path = "person_1.jpg"  # Input image path
mask = generate_mask(input_path) # Create mask
# Save result
save_transparent_segmentation(input_path, mask, "segmented_person.png")
