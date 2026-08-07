import numpy as np
import cv2
import matplotlib.pyplot as plt

image = cv2.imread('exercises/my_solutions/ex1/flower.jpg')
annotated_image = cv2.imread('exercises/my_solutions/ex1/flower-petals-annotated.jpg')

avg = (0, 0, 0)
avgCount = 0

for i in range(annotated_image.shape[0]):
    for j in range(annotated_image.shape[1]):
        if (annotated_image[i, j, 0] < 5 and annotated_image[i, j, 1] < 5 and annotated_image[i, j, 2] > 220):  # Check if the pixel is red
            #print(f"Red pixel found at: ({j}, {i}), image color is {image[i, j]}")  # Print the coordinates of the red pixel
            avg = (avg[0] + int(image[i, j, 0]), avg[1] + int(image[i, j, 1]), avg[2] + int(image[i, j, 2]))
            avgCount += 1

avg = (int(avg[0] / avgCount), int(avg[1] / avgCount), int(avg[2] / avgCount))  # Calculate the average color of the red pixels
print(f"Average color: {avg}")
