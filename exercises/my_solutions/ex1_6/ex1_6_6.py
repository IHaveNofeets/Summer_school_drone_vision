import numpy as np
import cv2


image = cv2.imread('exercises/my_solutions/cat.jpg')

lightImage = cv2.cvtColor(image, cv2.COLOR_BGR2HLS)  # Convert the image from BGR to HLS color space

maxLightnessPos = (0, 0)

for i in range(lightImage.shape[0]):
    for j in range(lightImage.shape[1]):
        if lightImage[i, j, 1] > lightImage[maxLightnessPos[0], maxLightnessPos[1], 1]:
            maxLightnessPos = (i, j)

cv2.circle(image, (maxLightnessPos[1], maxLightnessPos[0]), 10, (0, 0, 255), 3)  # Draw a red circle at the position of maximum lightness

cv2.imwrite('exercises/my_solutions/ex1/1_6_6_test.png', image)  # Save the image with the circle