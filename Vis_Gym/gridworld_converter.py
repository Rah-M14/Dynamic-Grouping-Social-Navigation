import cv2
import numpy as np
import matplotlib.pyplot as plt

# def display_image(title, image, cmap=None):
#     plt.figure(figsize=(6, 6))
#     plt.title(title)
#     plt.imshow(image, cmap=cmap)
#     plt.axis('off')
#     plt.show()

def grid_convert(imageName):
    
    # Load the Image
    image = cv2.imread(imageName, cv2.IMREAD_COLOR)
    # display_image('Original Image', cv2.cvtColor(image, cv2.COLOR_BGR2RGB))


    # Convert the Image to Grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # display_image('Grayscale Image', gray, cmap='gray')


    #Apply Binary Thresholding
    # Pixels with values above 2 are set to 0 (black), and pixels with values below 2 are set to 255 (white), effectively inverting the image 
    _, thresh = cv2.threshold(gray, 2, 255, cv2.THRESH_BINARY_INV)
    # display_image('Thresholded Image', thresh, cmap='gray')

    #Detect Contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    #Create an Empty Gridworld Matrix
    height, width = gray.shape
    gridworld = np.zeros((height, width), dtype=np.uint8)

    # Draw contours on the gridworld matrix
    cv2.drawContours(gridworld, contours, -1, (255), thickness=cv2.FILLED)

    # Apply dilation to thicken the borders and close holes
    kernel = np.ones((5,5), np.uint8)
    dilated_gridworld = cv2.dilate(gridworld, kernel, iterations=1)
    # display_image('Dilated Gridworld Matrix', dilated_gridworld, cmap='gray')

    # Find contours in the dilated image
    contours, _ = cv2.findContours(dilated_gridworld, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filter contours by area and remove small and large ones
    min_area = 300   # minimum area threshold
    max_area = 20000  # maximum area threshold
    filtered_contours = [cnt for cnt in contours if min_area < cv2.contourArea(cnt) < max_area]

    # Create a cleaned gridworld matrix
    cleaned_gridworld = np.zeros((height, width), dtype=np.uint8)
    cv2.drawContours(cleaned_gridworld, filtered_contours, -1, (255), thickness=cv2.FILLED)
    # display_image('Cleaned Gridworld Matrix', cleaned_gridworld, cmap='gray')
    
    return cleaned_gridworld





