import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

def laplacian(input_img):
    lap = cv.Laplacian(input_img,cv.CV_64F)
    lap = np.uint8(np.absolute(lap))
    return lap

def canny(input_img):
    result = cv.Canny(input_img,100,200)
    return result

def sobel(input_img):
    sobelx = cv.Sobel(input_img, cv.CV_64F,1,0,)
    sobely = cv.Sobel(input_img, cv.CV_64F,0,1)
    combined_sobel = cv.bitwise_or(sobelx, sobely)
    return combined_sobel

img = cv.imread('apples.png',0)
output = canny(img)

# Display images
plt.subplot(121),plt.imshow(img,cmap = 'gray')
plt.title('Original Image'), plt.xticks([]), plt.yticks([])

plt.subplot(122),plt.imshow(output,cmap = 'gray')
plt.title('Edge Detection'), plt.xticks([]), plt.yticks([])

plt.show()





