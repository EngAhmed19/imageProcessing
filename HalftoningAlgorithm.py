import cv2
import numpy as np
import matplotlib.pyplot as plt
from helperFunctions import convertImageToGray


class HalfToningImage:
	def __init__(self, image: np.ndarray):
		self.image = image
		self.gray_image=convertImageToGray(image)

	def simpleHalftoning(self) -> np.ndarray:
		half_toning_image: np.ndarray = np.zeros(self.gray_image.shape)

		row, column = self.gray_image.shape

		for i in range(row):
			for j in range(column):
				if self.gray_image[i, j] > 127:
					half_toning_image[i, j] = 255
				else:
					half_toning_image[i, j] = 0
		return half_toning_image

	def errorDiffusionHalfToning(self) -> np.ndarray:
		gray_image: np.ndarray = convertImageToGray(self.image).astype(np.float32)

		half_toning_image: np.ndarray = np.zeros_like(gray_image)

		row, column = gray_image.shape

		for i in range(row):
			for j in range(column):
				pixel_old = gray_image[i, j]
				if gray_image[i, j] > 127:
					half_toning_image[i, j] = 255
				else:
					half_toning_image[i, j] = 0
				error = pixel_old - half_toning_image[i, j]

				# check for the Right neighbor
				if j + 1 < gray_image.shape[1]:
					gray_image[i, j + 1] += error * (7 / 16)

				# check for the bottom-left neighbor
				if i + 1 < gray_image.shape[0] and j - 1 >= 0:
					gray_image[i + 1, j - 1] += error * (3 / 16)

				# check for the bottom neighbor
				if i + 1 < gray_image.shape[0]:
					gray_image[i + 1, j] += error * (5 / 16)

				# check for the bottom-right neighbor
				if i + 1 < gray_image.shape[0] and j + 1 < gray_image.shape[1]:
					gray_image[i + 1, j + 1] += error * (1 / 16)

		return half_toning_image
