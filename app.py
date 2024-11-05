# Jiaqi Liu/ Pingqi An
# CS5330_lab4
# Nov 11 2024

import cv2
import numpy as np
import torch
import gradio as gr
from torchvision import transforms
from PIL import Image

# Load pre-trained DeepLabV3 model
model = torch.hub.load('pytorch/vision:v0.10.0', 'deeplabv3_resnet50', pretrained=True)
model.eval()

# Image segmentation function
def segment_person(image_path):
    input_image = Image.open(image_path).convert("RGB")  # Open image and convert to RGB
    preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    input_tensor = preprocess(input_image).unsqueeze(0)  # Preprocess and add batch dimension
    
    # Get segmentation mask
    with torch.no_grad():
        output = model(input_tensor)['out'][0]
    mask = output.argmax(0).byte().cpu().numpy()  # Extract mask from model output
    
    # Apply mask to create image with transparent background
    rgba_image = np.array(input_image)
    # Set transparency for person class
    alpha_channel = np.where(mask == 15, 255, 0).astype(np.uint8)  
    rgba_image = cv2.cvtColor(rgba_image, cv2.COLOR_RGB2RGBA)
    rgba_image[:, :, 3] = alpha_channel  # Set alpha channel
    
    segmented_path = "segmented_person.png"
    cv2.imwrite(segmented_path, rgba_image)  # Save segmented image with transparency
    return segmented_path

# Split stereo image into left and right views
def split_stereo_image(sbs_image_path):
    image = cv2.imread(sbs_image_path)  # Read side-by-side stereo image
    if image is None:
        print(f"Error: Cannot read stereo image at path {sbs_image_path}")
        return None, None
    h, w, _ = image.shape
    left_image = image[:, :w // 2]  # Left view
    right_image = image[:, w // 2:]  # Right view
    return left_image, right_image

# Insert segmented person with depth adjustment
def insert_person_with_depth(
    left_image, right_image, person_path, depth_level, 
    x, y, scale=1.0
):
    # Read segmented person image with alpha channel
    person = cv2.imread(person_path, cv2.IMREAD_UNCHANGED)
    
    # Check if images loaded successfully
    if left_image is None or right_image is None:
        print("Error: Left or right image is missing.")
        return left_image, right_image
    if person is None:
        print(f"Error: Cannot read person image at path {person_path}")
        return left_image, right_image

    # Set disparity based on depth level
    if depth_level == "close":
        disparity = 30  # Larger offset for close depth
    elif depth_level == "medium":
        disparity = 15  # Medium offset
    elif depth_level == "far":
        disparity = 5  # Smaller offset for far depth
    else:
        disparity = 15  # Default to medium if unspecified

    # Resize person image based on scale
    h_person, w_person = person.shape[:2]
    person = cv2.resize(person, (int(w_person * scale), int(h_person * scale)))
    h_person, w_person = person.shape[:2]

    # Calculate shifted x position for right view
    x_right = x + disparity

    # Ensure person fits within left and right images
    h_left, w_left, _ = left_image.shape
    h_right, w_right, _ = right_image.shape
    x = min(x, w_left - w_person)
    y = min(y, h_left - h_person)
    x_right = min(x_right, w_right - w_person)

    # Overlay person onto left and right images
    alpha_mask = person[:, :, 3] / 255.0  # Extract alpha mask for blending
    for c in range(3):  # Blend RGB channels
        left_image[y:y+h_person, x:x+w_person, c] = (
            (1 - alpha_mask) * left_image[y:y+h_person, x:x+w_person, c] +
            alpha_mask * person[:, :, c]
        )
    for c in range(3):  # Apply to right image with shifted position
        right_image[y:y+h_person, x_right:x_right+w_person, c] = (
            (1 - alpha_mask) * right_image[y:y+h_person, x_right:x_right+w_person, c] +
            alpha_mask * person[:, :, c]
        )
    return left_image, right_image

# Create red-cyan anaglyph image
def create_anaglyph(left_image, right_image):
    anaglyph = np.zeros_like(left_image)
    anaglyph[:, :, 0] = left_image[:, :, 0]  # Red channel from left view
    anaglyph[:, :, 1] = right_image[:, :, 1]  # Green channel from right view
    anaglyph[:, :, 2] = right_image[:, :, 2]  # Blue channel from right view
    return anaglyph

# Generate anaglyph function for Gradio interface
def generate_anaglyph(person_image, stereo_image, depth, x, y, scale):
    # Segment person and create image with transparency
    segmented_person_path = segment_person(person_image)
    
    # Split stereo image into left and right views
    left_image, right_image = split_stereo_image(stereo_image)
    if left_image is None or right_image is None:
        return None

    # Insert person into stereo image with depth adjustment
    left_image_with_person, right_image_with_person = insert_person_with_depth(
        left_image, right_image, segmented_person_path, depth, x, y, scale
    )

    # Create anaglyph image from modified left and right views
    anaglyph_image = create_anaglyph(left_image_with_person, right_image_with_person)

    return anaglyph_image

# Gradio interface setup
iface = gr.Interface(
    fn=generate_anaglyph,
    inputs=[
        gr.Image(type="filepath", label="Upload Person Image (no need for alpha channel)"),
        gr.Image(type="filepath", label="Upload Stereo Side-by-Side Image"),
        gr.Radio(["close", "medium", "far"], label="Depth Level", value="medium"),
        gr.Slider(0, 1000, value=50, step=10, label="Horizontal Position (x)"),
        gr.Slider(0, 1000, value=50, step=10, label="Vertical Position (y)"),
        gr.Slider(0.5, 2.0, value=1.0, step=0.1, label="Scale")  # Scale adjustment slider
    ],
    outputs=gr.Image(label="Anaglyph Output"),
    title="3D Anaglyph Image Composer",
    description=(
        "Upload a person image and a stereo side-by-side image to create a 3D "
        "anaglyph image with adjustable depth perception."
    ))

iface.launch(share=True)

