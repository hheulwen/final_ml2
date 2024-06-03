import cv2
import numpy as np
from staff_detection import preprocess, get_staff_lines
from image_operations import save_image

def staff_removal(image_path, staff_dist):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    img = cv2.bitwise_not(img)
    th2 = cv2.adaptiveThreshold(
        img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 15, -2)

    vertical = th2.copy()
    rows, cols = vertical.shape

    verticalsize = int(staff_dist / 2)
    verticalStructure = cv2.getStructuringElement(
        cv2.MORPH_RECT, (1, verticalsize))
    vertical = cv2.erode(vertical, verticalStructure)
    vertical = cv2.dilate(vertical, verticalStructure)

    vertical_inverted = cv2.bitwise_not(vertical)
    smooth = vertical_inverted.copy()
    smooth = cv2.blur(smooth, (4, 4))
    (rows, cols) = np.where(img == 0)
    vertical_inverted[rows, cols] = smooth[rows, cols]

    return vertical_inverted

# Example usage
image_path = r'C:/Users/Chou/CODE/ml2/optical-music-recognition-master/optical-music-recognition-master/code/preprocessing/fuzzy-wuzzy.png'
output_path = r'C:/Users/Chou/CODE/ml2/optical-music-recognition-master/optical-music-recognition-master/code/preprocessing/processed.png'

# Load the image in grayscale
img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
if img is None:
    print(f"Error: Unable to read image at {image_path}")
else:
    height, width = img.shape
    staff_lines_thicknesses, staff_lines = get_staff_lines(width, height, img)

    # Preprocess to remove staff lines
    processed_img = preprocess(img)

    # Further process to remove remaining staff artifacts
    final_img = staff_removal(image_path, max(staff_lines_thicknesses) if staff_lines_thicknesses else 10)

    # Save the final processed image
    if save_image(output_path, final_img):
        print("Staff removal completed successfully.")
    else:
        print("Staff removal failed.")
