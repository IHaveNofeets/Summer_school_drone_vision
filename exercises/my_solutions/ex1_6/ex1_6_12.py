import numpy as np
import cv2
import matplotlib.pyplot as plt

image = cv2.imread('exercises/my_solutions/ex1/1_6_11_distances.png')

print(f"Image shape: {image.shape}")  # Print the shape of the image (height, width, channels)

# max value of each channel is 255
print(f"Max value in each channel: {np.max(image, axis=(0, 1))}")


plt.hist(image[:,:,0].ravel(), bins=30, edgecolor='black')

# Tilføj titler og labels
plt.title("Histogram")
plt.xlabel("Values")
plt.ylabel("Count")

# Vis diagrammet
plt.show()