import numpy as np
import matplotlib.pyplot as plt
from HalftoningAlgorithm import HalfToningImage
import cv2

image = cv2.imread("images/flower.png")
gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
print(gray_image.shape)

halftone = HalfToningImage(image)

dithered_image = halftone.order_dither(4)

plt.subplot(1, 2, 1)
plt.imshow(gray_image, cmap="gray")
plt.axis("off")
plt.subplot(1, 2, 2)
plt.imshow(dithered_image, cmap="gray")
plt.axis("off")
plt.show()
