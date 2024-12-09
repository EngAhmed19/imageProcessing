import numpy as np
from helperFunctions import convertImageToGray, custDynamicThreshold, ThresholdStrategy


class HalfToningImage:
	"""
	This is a halftoning class that apply halftoning algorithms on the image

	:parameter:
		:param image: The input image
		:type image: np.ndarray

	:raises ValueError: The image must be specified. please provide an image
	"""

	def __init__(self, image: np.ndarray):
		if image is None or not isinstance(image, np.ndarray):
			raise ValueError("The image must be specified. please provide an image")
		else:
			self.image = image
			self.gray_image = convertImageToGray(image)

	def simpleHalftoning(self, threshold_strategy: ThresholdStrategy = ThresholdStrategy.MEAN) -> np.ndarray:
		"""
		Apply simple halftoning algorithm on the image.

		:returns: The result image after applying simple halftoning algorithm.
		:rtype: np.ndarray.
		"""
		half_toning_image: np.ndarray = np.zeros(self.gray_image.shape)

		row, column = self.gray_image.shape

		t: float = custDynamicThreshold(self.gray_image, threshold_strategy)

		for i in range(row):
			for j in range(column):
				if self.gray_image[i, j] > t:
					half_toning_image[i, j] = 255
				else:
					half_toning_image[i, j] = 0
		return half_toning_image

	def errorDiffusionHalfToning(self, threshold_strategy: ThresholdStrategy = ThresholdStrategy.MEAN) -> np.ndarray:
		"""
		Applying error diffusion algorithm on the image.
		:return: The image after applying error diffusion halftoning algorithm.
		:rtype: np.ndarray
		"""
		gray_image: np.ndarray = convertImageToGray(self.image).astype(np.float32)

		half_toning_image: np.ndarray = np.zeros_like(gray_image)

		row, column = gray_image.shape

		t: float = custDynamicThreshold(self.gray_image, threshold_strategy)

		for i in range(row):
			for j in range(column):
				pixel_old = gray_image[i, j]
				if gray_image[i, j] > t:
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

	def order_dither(self, matrix_size: int = 4) -> np.ndarray:
		if matrix_size == 2:
			bayer_matrix: np.ndarray = np.array([[0, 2], [3, 1]]) * 1 / 4
		elif matrix_size == 4:
			bayer_matrix = np.array([
				[0, 8, 2, 10],
				[12, 4, 14, 6],
				[3, 11, 1, 9],
				[15, 7, 13, 5]
			]) * 1 / 16
		elif matrix_size == 6:
			bayer_matrix = np.array([
				[0, 32, 8, 0, 32, 8],
				[48, 16, 40, 48, 16, 40],
				[12, 44, 4, 12, 44, 4],
				[0, 32, 8, 0, 32, 8],
				[48, 16, 40, 48, 16, 40],
				[12, 44, 4, 12, 44, 4],
			]) * 1 / 36
		elif matrix_size == 8:
			bayer_matrix = np.array([
				[0, 32, 8, 40, 2, 34, 10, 42],
				[48, 16, 56, 24, 50, 18, 58, 26],
				[12, 44, 4, 36, 14, 46, 6, 38],
				[60, 28, 52, 20, 62, 30, 54, 22],
				[3, 35, 11, 43, 1, 33, 9, 41],
				[51, 19, 59, 27, 49, 17, 57, 25],
				[15, 47, 7, 39, 13, 45, 5, 37],
				[63, 31, 55, 23, 61, 29, 53, 21],
			]) * 1 / 64
		else:
			raise ValueError("Matrix size must be 2 or 4 or 6 or 8")

		normalized_image: np.ndarray = self.gray_image / 255

		width, height = normalized_image.shape

		tiled_matrix = np.tile(bayer_matrix, (width // matrix_size + 1, height // matrix_size + 1))
		tiled_matrix = tiled_matrix[:width, :height]

		dithered_image: np.ndarray = np.zeros_like(normalized_image)
		dithered_image = (normalized_image > tiled_matrix).astype(np.uint8) * 255

		return dithered_image
