import numpy as np
import matplotlib.pyplot as plt
from BasicEdgeDetection import BasicEdgeDetection
import cv2

image = cv2.imread("images/bad_light_1.jpg")
gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

basic_edge_detection = BasicEdgeDetection(image, contrast_based_smoothing=True)

gradient, direction = basic_edge_detection.kirschEdgeDetectionWithDirection()

plt.subplot(1, 2, 1)
plt.imshow(gradient, cmap="gray")
plt.subplot(1, 2, 2)
plt.imshow(np.zeros_like(gradient), cmap="gray")
print(direction)
subset_directions = direction[::15, ::15]
for i in range(subset_directions.shape[0]):
	for j in range(subset_directions.shape[1]):
		plt.text(j * 15, i * 15, subset_directions[i, j], color='red', fontsize=10, fontweight='bold', ha='center', # NOQA
				 va='center')  # NOQA

plt.axis("off")
plt.show()
