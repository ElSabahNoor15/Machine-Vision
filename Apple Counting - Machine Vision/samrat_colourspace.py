import cv2
import matplotlib.pyplot as plt
import numpy as np
from skimage.exposure import rescale_intensity
 
def identify_red_apples(image_path):
   
    apple = cv2.imread(image_path)
 
    # Convert the image to HSV color space
    hsv_image = cv2.cvtColor(apple, cv2.COLOR_BGR2HSV)
    plt.imshow(hsv_image)
    # Define the lower and upper bounds for red color in HSV
    lower_red = np.array([0, 90, 90])
    upper_red = np.array([20, 255, 255])
 
    # Create a mask for red pixels
    red_mask = cv2.inRange(hsv_image, lower_red, upper_red)
 
    # Find contours in the mask
    contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
 
    # Draw the contours on the original image and count the apples
    count = 0
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 50:  # Adjust the threshold based on your image
            cv2.drawContours(apple, [contour], -1, (0, 255, 0), 2)
            count += 1
 
    # Display the total number of red apples
    cv2.putText(apple, f"Total red apples: {count}", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    print(f"Total red apples detected: {count}")
 
    plt.imshow(cv2.cvtColor(apple, cv2.COLOR_BGR2RGB))
    plt.show()