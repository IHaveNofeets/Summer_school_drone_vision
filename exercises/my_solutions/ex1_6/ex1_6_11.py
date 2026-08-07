import numpy as np
import cv2
import matplotlib.pyplot as plt

image = cv2.imread('exercises/my_solutions/ex1/flower.jpg')

reference_color = (178, 180, 187)

# Calculate the Euclidean distance between each pixel and the reference color
distances = np.sqrt(np.sum((image - reference_color) ** 2, axis=2))
#cv2.imshow('Distances', distances.astype(np.uint8))
cv2.imwrite('exercises/my_solutions/ex1/1_6_11_distances.png', distances)  # Save the distances image
print (f"Distances shape: {distances.shape}")