import numpy as np
from helperFunctions import convertImageToGray, convolution


class BasicEdgeDetection:
	"""
	This class have basic methods to apply edge detection in image.

	:argument:
		:arg image:The input image.
	"""

	def __init__(self, image: np.ndarray):
		self.image = image
		self.gray_image = convertImageToGray(image)

	def _calculateEdgeDetection(self, mask1: np.ndarray, mask2: np.ndarray) -> np.ndarray:
		"""
		This function calucalate the gradiant on both directions x and y and then take the square root of its square
		and then applying the threshold and return the image with detected edges.

		that is the two masks that can be used in this function:

		>>> sobel_x: np.ndarray = np.array([[-1, 0, 1],[-2, 0, 2], [-1, 0, -1]])
		>>> perwitt_x: np.ndarray = np.array([[-1, 0, 1],[-1, 0, 1],[-1, 0, 1]])
		>>> sobel_y: np.ndarray = np.array([[-1, -2, -1],[0, 0, 0],[1, 2, 1]])
		>>> perwitt_y: np.ndarray = np.array([[-1, -1, -1],[0, 0, 0],[1, 1, 1]])
		:param mask1: The first mask to convolve it with the image
		:param mask2: The second mask to convolve it with the image

		:returns: The Image with an edges detected after applying the masks on it
		"""
		gx: np.ndarray = convolution(self.gray_image, mask1)
		gy: np.ndarray = convolution(self.gray_image, mask2)

		gradient_magnitude: np.ndarray = np.sqrt(gx ** 2 + gy ** 2)
		gradient_magnitude = np.clip(gradient_magnitude, 0, 255)

		edge_detection_image: np.ndarray = np.zeros_like(gradient_magnitude)

		t: int = gradient_magnitude.mean()

		for i in range(gradient_magnitude.shape[0]):
			for j in range(gradient_magnitude.shape[1]):
				if gradient_magnitude[i, j] > t:
					edge_detection_image[i, j] = 255
				else:
					edge_detection_image[i, j] = 0

		return edge_detection_image

	def sobelEdgeDetection(self) -> np.ndarray:
		"""
		This function calculate the sobel edge detection algorithm.
		The sobel algorithm uses two masks:

		>>> sobel_x_: np.ndarray = np.array([[-1, 0, 1],[-2, 0, 2], [-1, 0, -1]])
		>>> sobel_y_: np.ndarray = np.array([[-1, -2, -1],[0, 0, 0],[1, 2, 1]])
		:return: The image with edges detected using sobel edge detection algorithm.
		"""
		sobel_x: np.ndarray = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, -1]])
		sobel_y: np.ndarray = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])

		edge_detection_image_result: np.ndarray = self._calculateEdgeDetection(sobel_x, sobel_y)

		return edge_detection_image_result

	def perwittEdgeDetection(self):
		"""
		This function calculate the perwitt edge detection algorithm.
		The perwitt algorithm uses two masks:

		>>> perwitt_x_: np.ndarray = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
		>>> perwitt_y_: np.ndarray = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]])
		:return: The image with edges detected using perwitt edge detection algorithm.
		"""
		perwitt_x: np.ndarray = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
		perwitt_y: np.ndarray = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]])

		edge_detection_image_result: np.ndarray = self._calculateEdgeDetection(perwitt_x, perwitt_y)

		return edge_detection_image_result
