import numpy as np
import cv2

image = cv2.imread('exercises/my_solutions/cat.jpg')

print(image.shape)  # Print the shape of the image (height, width, channels)

newHeight = image.shape[0] / 2  # Half the original height

newImage = np.zeros((int(newHeight), image.shape[1], image.shape[2]), np.uint8)

for i in range(int(newHeight)):
    newImage[i] = image[i]

cv2.imwrite('exercises/my_solutions/ex1/1_6_4_test.png', newImage)  # Save the resized image