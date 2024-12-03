import numpy as np
from helperFunctions import convertImageToGray, convolution, custImageToGray


class BasicEdgeDetection:
	def __init__(self, image: np.ndarray, contrast_based_smoothing:bool = False):
		self.image = image
		# self.gray_image = convertImageToGray(image)
		self.gray_image = custImageToGray(image)
		self.contrast_based_smoothing = contrast_based_smoothing

		if self.contrast_based_smoothing:
			self.gray_image =  self._contrastSmoothing(self.gray_image)

	def _contrastSmoothing(self, image:np.ndarray)->np.ndarray:
		smoothing_kernel = np.ones((3,3)) / 9
		return convolution(image, smoothing_kernel)

	def _calculateEdgeDetection(self, mask1: np.ndarray, mask2: np.ndarray):
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
		sobel_x: np.ndarray = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, -1]])
		sobel_y: np.ndarray = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])

		edge_detection_image_result: np.ndarray = self._calculateEdgeDetection(sobel_x, sobel_y)

		return edge_detection_image_result

	def perwittEdgeDetection(self):
		perwitt_x: np.ndarray = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
		perwitt_y: np.ndarray = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]])

		edge_detection_image_result: np.ndarray = self._calculateEdgeDetection(perwitt_x, perwitt_y)

		return edge_detection_image_result
