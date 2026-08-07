import numpy as np
import cv2


image = cv2.imread('exercises/my_solutions/cat.jpg')

newImage = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)  # Convert the image from BGR to HSV color space

cv2.imwrite('exercises/my_solutions/ex1/1_6_5_test.png', newImage)  # Save the converted image