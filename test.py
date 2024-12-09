import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mplt
# from BasicEdgeDetection import BasicEdgeDetection
from Filtering import Filtering
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

image = mplt.imread("images/bad_light_1.jpg")
filtering = Filtering(image=image)
low_img, high_img = filtering.applyLowPassFilter(), filtering.applyHighPassFilter()
plt.subplot(1, 2, 1)
plt.imshow(low_img, cmap="gray")
plt.subplot(1, 2, 2)
plt.imshow(high_img, cmap="gray")

# image_flib = image[:, ::-1]
# image_flib = image[::-1, :]
# image_flib = image[::-1, ::-1]    
# image_flib = image[:, ::2]



# plt.subplot(1, 2, 1)
# plt.axis("off")
# plt.imshow(image)
# plt.subplot(1, 2, 2)
# plt.imshow(image_flib)

# plt.axis("off")
# plt.show()