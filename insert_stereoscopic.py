# Jiaqi Liu/ Pingqi An
# CS5330_lab4
# Nov 11 2024

import cv2
import numpy as np

# Split the side-by-side stereo image into left and right views
def split_stereo_image(sbs_image_path):
    image = cv2.imread(sbs_image_path)  # Read the stereo image
    if image is None:
        print(f"Error: Cannot read stereo image at path {sbs_image_path}")
        return None, None
    h, w, _ = image.shape
    left_image = image[:, :w // 2]  # Extract the left half of the image
    right_image = image[:, w // 2:]  # Extract the right half of the image
    return left_image, right_image

# Insert the segmented person into both left and right images with depth adjustment
def insert_person_with_depth(
    left_image, right_image, person_path, depth_level, 
    output_left_path, output_right_path, x=50, y=50
):  
    # Read the person image with alpha channel
    person = cv2.imread(person_path, cv2.IMREAD_UNCHANGED) 

    # Check if images are loaded correctly
    if left_image is None or right_image is None:
        print("Error: Left or right image is missing.")
        return
    if person is None:
        print(f"Error: Cannot read person image at path {person_path}")
        return

    # Set disparity and scaling based on the depth level
    if depth_level == "close":
        disparity = 30  # Large offset for closer appearance
        scale = 2.0  # Slightly enlarge
    elif depth_level == "medium":
        disparity = 15  # Medium offset
        scale = 1.0  # Original size
    elif depth_level == "far":
        disparity = 5  # Small offset for farther appearance
        scale = 0.8  # Slightly shrink
    else:
        disparity = 15  # Default to medium offset if unspecified
        scale = 1.0

    # Resize the person image based on the scale factor
    h, w = person.shape[:2]
    person = cv2.resize(person, (int(w * scale), int(h * scale)))
    h, w = person.shape[:2]

    # Check if the insertion position is within image bounds
    if y + h > left_image.shape[0]:
        h = left_image.shape[0] - y  # Adjust height to fit
        person = person[:h, :, :]
    if x + w > left_image.shape[1]:
        w = left_image.shape[1] - x  # Adjust width to fit
        person = person[:, :w, :]

    # Calculate the shifted x position for the right image
    x_right = x + disparity

    # Insert the person into the left image
    left_image[y:y+h, x:x+w, :] = np.where(
        person[:, :, 3:] == 0,
        left_image[y:y+h, x:x+w, :],
        person[:, :, :3]
    )

    # Insert the person into the right image with horizontal offset
    # Check bounds for the right image
    if x_right + w <= right_image.shape[1] and y + h <= right_image.shape[0]:
        right_image[y:y+h, x_right:x_right+w, :] = np.where(
            person[:, :, 3:] == 0,
            right_image[y:y+h, x_right:x_right+w, :],
            person[:, :, :3]
        )
    else:
        print("Warning: Person image is out of bounds in the right image.")

    # Save the resulting images
    cv2.imwrite(output_left_path, left_image)
    cv2.imwrite(output_right_path, right_image)
    print(f"Images saved as {output_left_path} and {output_right_path}")

# Example usage
sbs_image_path = "sbs_neu.jpg"  # Path to the side-by-side stereo image
person_path = "segmented_person.png"  # Path to the segmented person image
depth_level = "close"  # Set depth level (close, medium, far)

# Split the stereo image into left and right views
left_image, right_image = split_stereo_image(sbs_image_path)

# Insert the person image into the stereo pair
insert_person_with_depth(
    left_image, right_image, person_path, depth_level, 
    "output_left.png", "output_right.png", 
    x=700, y=350
)
