import numpy as np
from helperFunctions import convertImageToGray, convolution


class BasicEdgeDetection:
	def __init__(self, image: np.ndarray):
		self.image = image
		self.gray_image = convertImageToGray(image)

	def sobelEdgeDetection(self) -> np.ndarray:
		sobel_x: np.ndarray = np.array([[-1, 0, 1],
										[-2, 0, 2],
										[-1, 0, -1]])
		sobel_y: np.ndarray = np.array([[-1, -2, -1],
										[0, 0, 0],
										[1, 2, 1]])

		gx: np.ndarray = convolution(self.gray_image, sobel_x)
		gy: np.ndarray = convolution(self.gray_image, sobel_y)

		gradient_magnitude: np.ndarray = np.sqrt(gx ** 2 + gy ** 2)
		gradient_magnitude = np.clip(gradient_magnitude, 0, 255)

		edge_detection_image: np.ndarray = np.zeros_like(gradient_magnitude)

		T: int = gradient_magnitude.mean()

		for i in range(gradient_magnitude.shape[0]):
			for j in range(gradient_magnitude.shape[1]):
				if gradient_magnitude[i, j] > T:
					edge_detection_image[i, j] = 255
				else:
					edge_detection_image[i, j] = 0

		return edge_detection_image

	def perwittEdgeDetection(self):
		perwitt_x: np.ndarray = np.array([[-1, 0, 1],
										  [-1, 0, 1],
										  [-1, 0, 1]])
		perwitt_y: np.ndarray = np.array([[-1, -1, -1],
										  [0, 0, 0],
										  [1, 1, 1]])

		gx: np.ndarray = convolution(self.gray_image, perwitt_x)
		gy: np.ndarray = convolution(self.gray_image, perwitt_y)

		gradient_magnitude: np.ndarray = np.sqrt(gx ** 2 + gy ** 2)
		gradient_magnitude = np.clip(gradient_magnitude, 0, 255)

		edge_detection_image: np.ndarray = np.zeros_like(gradient_magnitude)

		T: int = gradient_magnitude.mean()

		for i in range(gradient_magnitude.shape[0]):
			for j in range(gradient_magnitude.shape[1]):
				if gradient_magnitude[i, j] > T:
					edge_detection_image[i, j] = 255
				else:
					edge_detection_image[i, j] = 0

		return edge_detection_image
