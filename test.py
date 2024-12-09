import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mplt
# from BasicEdgeDetection import BasicEdgeDetection
from Filtering import Filtering
from ColorManipulator import ColorManipulator
# import cv2


# image = cv2.imread("images/bad_light_1.jpg")
# gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

# basic_edge_detection = BasicEdgeDetection(image, contrast_based_smoothing=True)

# gradient = basic_edge_detection.sobelEdgeDetection()

# plt.subplot(1, 2, 1)
# plt.imshow(gradient, cmap="gray")
# plt.subplot(1, 2, 2)
# plt.imshow(np.zeros_like(gradient), cmap="gray")

# plt.axis("off")
# plt.show()

# image = mplt.imread("images/bad_light_1.jpg")
image = mplt.imread("images/woody.png")
color = ColorManipulator(image=image)
out_image = color.apply_color_filter(1, 3, 1)

# filtering = Filtering(image=image)
# low_img, high_img = filtering.applyLowPassFilter(), filtering.applyHighPassFilter()
# plt.subplot(1, 2, 1)
# plt.imshow(low_img, cmap="gray")
# plt.subplot(1, 2, 2)
# plt.imshow(high_img, cmap="gray")

# image_flib = image[:, ::-1]
# image_flib = image[::-1, :]
# image_flib = image[::-1, ::-1]    
# image_flib = image[:, ::2]



plt.subplot(1, 2, 1)
plt.axis("off")
plt.imshow(image)
plt.subplot(1, 2, 2)
plt.imshow(out_image)

plt.axis("off")
plt.show()