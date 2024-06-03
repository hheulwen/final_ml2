import numpy as np
import cv2

# creates contours around notes and returns the coordinates of rectangle bounding
# said objects
def make_bounding_boxes(image):
    raw_image = image

    bilateral_filtered_image = cv2.bilateralFilter(raw_image, 5, 175, 175)
    edge_detected_image = cv2.Canny(bilateral_filtered_image, 75, 200)

    # Use cv2.findContours and handle both return cases
    contours, _ = cv2.findContours(edge_detected_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    bounding_boxes = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        boxed_img = raw_image[y:y+h, x:x+w]
        if np.size(boxed_img) > 0:
            bounding_boxes.append([x, y, w, h])

    # Convert bounding_boxes to numpy array
    bounding_boxes = np.array(bounding_boxes)

    # Group rectangles, if necessary
    if len(bounding_boxes) > 0:
        bounding_boxes, _ = cv2.groupRectangles(bounding_boxes.tolist(), 1, 0.9)

    return np.array(bounding_boxes).astype(int)