import numpy as np
import matplotlib.pyplot as plt
from BasicEdgeDetection import BasicEdgeDetection
import cv2

image = cv2.imread("images/bad_light_1.jpg")
gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

basic_edge_detection = BasicEdgeDetection(image, contrast_based_smoothing=True)

gradient = basic_edge_detection.sobelEdgeDetection()

plt.subplot(1, 2, 1)
plt.imshow(gradient, cmap="gray")
plt.subplot(1, 2, 2)
plt.imshow(np.zeros_like(gradient), cmap="gray")

plt.axis("off")
plt.show()
