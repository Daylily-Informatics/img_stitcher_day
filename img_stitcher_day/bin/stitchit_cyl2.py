import cv2
import numpy as np
import sys

def unwrap_cylinder(image, cylinder_radius):
    height, width = image.shape[:2]
    
    # Calculate the expected height of the unwrapped image
    # This is an approximation and might need adjustment
    expected_height = height

    unwrapped = np.zeros((expected_height, width, 3), dtype=np.uint8)

    for x in range(width):
        for y in range(height):
            # Normalize the y-coordinate
            normalized_y = (y / height) - 0.5

            # Calculate the angle theta
            theta = normalized_y * np.pi

            # Calculate the new y-coordinate on the unwrapped image
            new_y = int(((theta + np.pi) / (2 * np.pi)) * expected_height)

            if 0 <= new_y < expected_height:
                unwrapped[new_y, x] = image[y, x]

    return unwrapped

# Load the image
image_path = 'path_to_your_cylinder_image.png'  # replace with your image path
image = cv2.imread(sys.argv[1])
out_png = sys.argv[2]

# Cylinder unwrapping
cylinder_radius = 25  # Radius for a 100mm diameter cylinder
unwrapped_image = unwrap_cylinder(image, cylinder_radius)

# Save the unwrapped image
cv2.imwrite(out_png, unwrapped_image)
