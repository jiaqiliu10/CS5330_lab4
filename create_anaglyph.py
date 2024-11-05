# Jiaqi Liu/ Pingqi An
# CS5330_lab4
# Nov 11 2024

import cv2

# Create an Anaglyph image
def create_anaglyph(left_image_path, right_image_path, output_anaglyph_path):
    # Read the left and right images
    left_image = cv2.imread(left_image_path)
    right_image = cv2.imread(right_image_path)

    # Check if both images were loaded successfully
    if left_image is None or right_image is None:
        print("Error: Cannot read one or both of the images.")
        return

    # Extract the red channel from the left image
    left_red = left_image[:, :, 2]

    # Extract the green and blue channels from the right image
    right_green_blue = right_image[:, :, :2]

    anaglyph_image = cv2.merge((
        right_green_blue[:, :, 0],  # Green channel from the right image
        right_green_blue[:, :, 1],  # Blue channel from the right image
        left_red  # Red channel from the left image
    ))
    # Save the Anaglyph image
    cv2.imwrite(output_anaglyph_path, anaglyph_image)
    print(f"Anaglyph image saved as {output_anaglyph_path}")

# Example usage
create_anaglyph("output_left.png", "output_right.png", "anaglyph_output.png")

