import numpy as np
from helperFunctions import convolution, custImageToGray, custDynamicThreshold, ThresholdStrategy
import cv2


class BasicEdgeDetection:
	"""
	This class have basic methods to apply edge detection in image.

	:parameter:
		:param image:The input image.
		:type image:np.ndarray
		:param contrast_based_smoothing: Applying a contrast based smoothing to smooth the image or not
		:type contrast_based_smoothing: bool

	:raises ValueError: The image must be specified. please provide an image
	"""

	def __init__(self, image: np.ndarray, contrast_based_smoothing: bool = False):
		self.image = image
		if image is None:
			raise ValueError("The image must be specified. please provide an image")
		else:
			# self.gray_image = convertImageToGray(image)
			self.gray_image = custImageToGray(image)
			self.contrast_based_smoothing = contrast_based_smoothing

			if self.contrast_based_smoothing:
				self.gray_image = self._contrastSmoothing(self.gray_image)

	def _contrastSmoothing(self, image: np.ndarray) -> np.ndarray:  # NOQA
		smoothing_kernel = np.ones((3, 3)) / 9
		return convolution(image, smoothing_kernel)

	def _calculateEdgeDetection(self, mask1: np.ndarray, mask2: np.ndarray,
								threshold_strategy: ThresholdStrategy) -> np.ndarray:  # NOQA
		"""
		This function calucalate the gradiant on both directions x and y and then take the square root of its square
		and then applying the threshold and return the image with detected edges.

		that is the two masks that can be used in this function:

		>>> sobel_x: np.ndarray = np.array([[-1, 0, 1],[-2, 0, 2], [-1, 0, -1]])
		>>> perwitt_x: np.ndarray = np.array([[-1, 0, 1],[-1, 0, 1],[-1, 0, 1]])
		>>> sobel_y: np.ndarray = np.array([[-1, -2, -1],[0, 0, 0],[1, 2, 1]])
		>>> perwitt_y: np.ndarray = np.array([[-1, -1, -1],[0, 0, 0],[1, 1, 1]])

		:parameter:
			:param mask1: The first mask to convolve it with the image
			:type mask1:np.ndarray
			:param mask2: The second mask to convolve it with the image
			:type mask2:np.ndarray

		:returns: The Image with an edges detected after applying the masks on it
		"""
		gx: np.ndarray = convolution(self.gray_image, mask1)
		gy: np.ndarray = convolution(self.gray_image, mask2)

		gradient_magnitude: np.ndarray = np.sqrt(gx ** 2 + gy ** 2)

		gradient_magnitude = np.uint8(
			255 * (gradient_magnitude / np.max(gradient_magnitude)))  # Normalize the result between 0-255

		edge_detection_image: np.ndarray = np.zeros_like(gradient_magnitude)

		t: int = custDynamicThreshold(gradient_magnitude, threshold_strategy)

		for i in range(gradient_magnitude.shape[0]):
			for j in range(gradient_magnitude.shape[1]):
				if gradient_magnitude[i, j] > t:
					edge_detection_image[i, j] = 255
				else:
					edge_detection_image[i, j] = 0

		return edge_detection_image

	def sobelEdgeDetection(self, threshold_strategy: ThresholdStrategy = ThresholdStrategy.MEAN) -> np.ndarray:
		"""
		This function calculate the sobel edge detection algorithm.
		The sobel algorithm uses two masks:

		>>> sobel_x_: np.ndarray = np.array([[-1, 0, 1],[-2, 0, 2], [-1, 0, -1]])
		>>> sobel_y_: np.ndarray = np.array([[-1, -2, -1],[0, 0, 0],[1, 2, 1]])

		It applies a threshold strategy to chosse the best threshold for the image like (mean,median,...,etc.).

		:parameter:
			:param threshold_strategy: The threshold strategy that will be applied to the image.
			:type threshold_strategy:ThresholdStrategy

		:return: The image with edges detected using sobel edge detection algorithm.
		:rtype: np.ndarray
		"""
		sobel_x: np.ndarray = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, -1]])
		sobel_y: np.ndarray = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])

		edge_detection_image_result: np.ndarray = self._calculateEdgeDetection(sobel_x, sobel_y, threshold_strategy)

		return edge_detection_image_result

	def perwittEdgeDetection(self, threshold_strategy: ThresholdStrategy = ThresholdStrategy.MEAN) -> np.ndarray:
		"""
		This function calculate the perwitt edge detection algorithm.
		The perwitt algorithm uses two masks:

		>>> perwitt_x_: np.ndarray = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
		>>> perwitt_y_: np.ndarray = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]])

		It applies a threshold strategy to chosse the best threshold for the image like (mean,median,...,etc.)

		:parameter:
			:param threshold_strategy: The threshold strategy that will be applied to the image.
			:type threshold_strategy:ThresholdStrategy

		:return: The image with edges detected using perwitt edge detection algorithm.
		:rtype: np.ndarray
		"""
		perwitt_x: np.ndarray = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
		perwitt_y: np.ndarray = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]])

		edge_detection_image_result: np.ndarray = self._calculateEdgeDetection(perwitt_x, perwitt_y, threshold_strategy)

		return edge_detection_image_result

	def kirschEdgeDetectionWithDirection(self) -> tuple[np.ndarray, np.ndarray]:
		"""
		Applies Kirsch edge detection to a grayscale image, determining edge magnitudes and their directions.

		The function uses predefined Kirsch masks for different compass directions (N, NW, W, SW, S, SE, E, NE)
		to compute the gradient magnitude and the corresponding edge directions for each pixel in the image.
		Then it applies a threshold strategy to choose the best threshold.
		The output is normalized and thresholded for better visualization.

		Example usage:
			>>> gradient_, directions =self.kirschEdgeDetectionWithDirection()
			>>> print(gradient_.shape)  # Shape of the edge magnitude image
			>>> print(directions[100, 100])  # Direction of the edge at pixel (100, 100)

		:returns:
			tuple containing:
				- The first element is a 2D numpy array representing the gradient magnitude image, normalized to 0-255.
				-The second element is a 2D numpy array of strings, where each string represents the direction of the edge
				at the corresponding pixel (e.g., "N", "NW").
		"""
		kirsch_masks: dict[str:np.ndarray] = {  # That is all mask for all directions
			"N": np.array([[5, 5, 5], [-3, 0, -3], [-3, -3, -3]]),
			"NW": np.array([[5, 5, -3], [5, 0, -3], [-3, -3, -3]]),
			"W": np.array([[5, -3, -3], [5, 0, -3], [5, -3, -3]]),
			"SW": np.array(([[-3, -3, -3], [5, 0, -3], [5, 5, - 3]])),
			"S": np.array([[-3, -3, -3], [-3, 0, 3], [5, 5, 5]]),
			"SE": np.array([[-3, -3, -3], [-3, 0, 5], [-3, 5, 5]]),
			"E": np.array([[-3, -3, 5], [-3, 0, 5], [-3, -3, 5]]),
			"NE": np.array([[-3, 5, 5], [-3, 0, 5], [-3, -3, -3]])
		}
		h, w = self.gray_image.shape
		gradient_magnitude: np.ndarray = np.zeros_like(
			self.gray_image)  # The resulting image after applying kirsch edge detection
		edge_direction: np.ndarray = np.full((h, w), "")  # The direction for each pixel

		for direction, mask in kirsch_masks.items():
			response: np.ndarray = cv2.filter2D(self.gray_image, -1, mask)  # apply convolution for each mask
			indices_mask = response > gradient_magnitude  # indices for the pixels where this mask has the highest response
			gradient_magnitude[indices_mask] = response[indices_mask]  # update the max magnitude
			edge_direction[indices_mask] = direction  # update with the max direction

		gradient_magnitude = np.uint8(
			255 * (gradient_magnitude / np.max(gradient_magnitude)))  # Normalize the result between 0-255

		t = custDynamicThreshold(gradient_magnitude,
								 ThresholdStrategy.MEAN)  # applying the threshold strategy to apply on the image for better result # NOQA

		for i in range(gradient_magnitude.shape[0]):
			for j in range(gradient_magnitude.shape[1]):
				if gradient_magnitude[i, j] > t:
					gradient_magnitude[i, j] = 255
				else:
					gradient_magnitude[i, j] = 0

		return gradient_magnitude, edge_direction
