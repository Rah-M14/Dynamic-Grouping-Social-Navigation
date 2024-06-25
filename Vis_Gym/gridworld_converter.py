import cv2
import numpy as np
import matplotlib.pyplot as plt

def grid_convert(imageName, scale=1):
    # Load the Image
    image = cv2.imread(imageName, cv2.IMREAD_COLOR)

    # Convert the Image to Grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply Binary Thresholding
    _, thresh = cv2.threshold(gray, 2, 255, cv2.THRESH_BINARY_INV)

    # Detect Contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Create an Empty Gridworld Matrix with scaled dimensions
    height, width = gray.shape
    scaled_height = height // scale
    scaled_width = width // scale
    gridworld = np.zeros((scaled_height, scaled_width), dtype=np.uint8)

    # Draw contours on the gridworld matrix
    for cnt in contours:
        scaled_cnt = cnt // scale
        cv2.drawContours(gridworld, [scaled_cnt], -1, (255), thickness=cv2.FILLED)

    # Apply dilation to thicken the borders and close holes
    kernel = np.ones((2, 2), np.uint8)
    dilated_gridworld = cv2.dilate(gridworld, kernel, iterations=1)

    # Find contours in the dilated image
    contours, _ = cv2.findContours(dilated_gridworld, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filter contours by area and remove small and large ones
    min_area = 400 // (scale ** 2)  # scale the area thresholds
    max_area = 20000 // (scale ** 2)
    filtered_contours = [cnt for cnt in contours if min_area < cv2.contourArea(cnt) < max_area]

    # Create a cleaned gridworld matrix
    cleaned_gridworld = np.zeros((scaled_height, scaled_width), dtype=np.uint8)
    cv2.drawContours(cleaned_gridworld, filtered_contours, -1, (255), thickness=cv2.FILLED)

    return cleaned_gridworld
