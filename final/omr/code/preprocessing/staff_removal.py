import cv2
import numpy as np


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