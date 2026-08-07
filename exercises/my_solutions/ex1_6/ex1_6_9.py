import numpy as np
import cv2
import matplotlib.pyplot as plt

image = cv2.imread('exercises/my_solutions/ex1/flower.jpg')

hls_image = cv2.cvtColor(image, cv2.COLOR_BGR2HLS)

# Extract the L (lightness) channel
l_channel = hls_image[:, :, 1]

# Create the histogram
plt.hist(l_channel.ravel(), bins=30, edgecolor='black')

# Tilføj titler og labels
plt.title("Lightness Channel Histogram")
plt.xlabel("Values")
plt.ylabel("Count")

# Vis diagrammet
plt.show()