import sys
import cv2
import numpy as np

def load_images(image_paths):
    return [cv2.imread(path) for path in image_paths]

def stitch_images(images):
    # Initialize OpenCV's stitcher class and stitch images
    stitcher = cv2.Stitcher_create()
    status, stitched = stitcher.stitch(images)

    if status == cv2.Stitcher_OK:
        return stitched
    else:
        print("Error during stitching:", status)
        return None

def cylindrical_unwrap(image):
    # This function would implement the cylindrical unwrapping
    # The implementation details would depend on the specific requirements and the geometry of the test tube
    # This is a placeholder for the actual unwrapping algorithm
    return image

# List of image paths in PNG format
out_stitched_png_path_and_prefix = sys.argv[1]
image_paths = [sys.argv[2], sys.argv[3]]
images = load_images(image_paths)

# Stitch images
stitched = stitch_images(images)
if stitched is not None:
    # Apply cylindrical unwrapping
    unwrapped = cylindrical_unwrap(stitched)

    # Save or display the result
    cv2.imwrite(out_stitched_png_path_and_prefix + "_stitched_unwrapped.png", unwrapped)
    cv2.imwrite(out_stitched_png_path_and_prefix + "_stitched.png", stitched)
    #cv2.imshow("Stitched and Unwrapped Image", unwrapped)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
