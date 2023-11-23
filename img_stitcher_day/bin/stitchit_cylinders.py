import cv2
import numpy as np
import sys

def unwrap_cylinder(image, cylinder_radius):
    height, width = image.shape[:2]
    unwrapped_height = int(2 * np.pi * cylinder_radius)
    unwrapped = np.zeros((unwrapped_height, width, 3), dtype=np.uint8)

    for x in range(width):
        for y in range(height):
            # Calculate the normalized height
            normalized_y = (y / height) - 0.5

            # Calculate the angle theta and the new y-coordinate
            theta = normalized_y * np.pi
            new_y = int((theta + np.pi) / (2 * np.pi) * unwrapped_height)

            # Map the pixel to the unwrapped image
            if 0 <= new_y < unwrapped_height:
                unwrapped[new_y, x] = image[y, x]

    return unwrapped

# Load the image
image_path = sys.argv[1]
image = cv2.imread(image_path)

# Cylinder unwrapping (adjust the radius as per your cylinder)
cylinder_radius = 100  # example radius, adjust as needed
unwrapped_image = unwrap_cylinder(image, cylinder_radius)

# Save the unwrapped image
cv2.imwrite('unwrapped_cylinder.png', unwrapped_image)
