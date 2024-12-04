import numpy as np
from helperFunctions import custImageToGray, convolution, custGenericFilter, ThresholdStrategy, custDynamicThreshold


class AdvancedEdgeDetection:
	def __init__(self, image: np.ndarray):
		if not image.any():
			raise ValueError("The image must be specified. please provide an image")
		else:
			self.image = image
			self.gray_image = custImageToGray(image)

	# self.gray_image = convertImageToGray(image)

	def _homogeneityFunction(self, neighborhood: np.ndarray) -> float:  # NOQA
		"""
		Function to calculate homogeneity for a given neighborhood.
		The homogeneity operator computes the maximum absolute difference
		between the center pixel and the surrounding pixels.
		"""
		center_pixel = neighborhood[len(neighborhood) // 2, len(neighborhood) // 2]
		diff = np.abs(neighborhood - center_pixel)
		return np.max(diff)

	def homogeneityOperator(self, area_size: int = 3, threshold: int = None,
							strategy: ThresholdStrategy = ThresholdStrategy.MEAN_PLUS_STD) -> np.ndarray:  # NOQA
		"""
		Applies the Homogeneity Operator for edge detection.

		Args:
			area_size (int): The size of the neighborhood (odd number).
			threshold (int): The threshold for binary edge map, a number between 0 and 1.

		Returns:
			np.ndarray: The thresholded edge map.
			:param threshold:
			:param area_size:
			:param strategy:

		"""
		filtered_image = custGenericFilter(self.gray_image, function=self._homogeneityFunction, kernel_size=area_size,
										   padding=True)  # NOQA
		if threshold:
			threshold = threshold
		else:
			# strategy is Enum
			threshold = custDynamicThreshold(image=filtered_image, strategy=strategy)

		print(f"threshold {threshold}")
		# Convert to 0 or 255 (binary image)
		edge_map = (filtered_image > threshold).astype(np.uint8) * 255
		# print(f"Filtered image in the class  {filtered_image.shape}")
		# print(f"Edge map in the class  {edge_map.shape}")
		return edge_map

	def _differenceFunction(self, neighborhood: np.ndarray) -> float:  # NOQA
		"""
		Applies the Difference Operator for edge detection.
		Used with custGenericFilter,
		It calculates differences between specific pixel pairs and returns the maximum value.
		It works with 3×3 only
		"""
		return np.max(
			[
				np.abs(neighborhood[0, 0] - neighborhood[2, 2]),  # Top-left ➖ Bottom-right
				np.abs(neighborhood[0, 2] - neighborhood[2, 0]),  # Top-right ➖ Bottom-left
				np.abs(neighborhood[0, 1] - neighborhood[2, 1]),  # Top-center ➖ Bottom-center
				np.abs(neighborhood[1, 0] - neighborhood[1, 2]),  # Left-center ➖ Right-center
			]
		)

	def differenceOperator(self, threshold: int = None,
						   strategy: ThresholdStrategy = ThresholdStrategy.MEAN_PLUS_STD) -> np.ndarray:  # NOQA
		"""

		"""
		area_size: int = 3  # it works with 3×3
		filtered_image = custGenericFilter(self.gray_image, function=self._differenceFunction, kernel_size=area_size,
										   padding=True)  # NOQA
		if threshold:
			threshold = threshold
		else:
			# strategy is Enum
			threshold = custDynamicThreshold(image=filtered_image, strategy=strategy)

		print(f"threshold {threshold}")
		# Convert to 0 or 255 (binary image)
		edge_map = (filtered_image > threshold).astype(np.uint8) * 255
		# print(f"Filtered image in the class  {filtered_image.shape}")
		# print(f"Edge map in the class  {edge_map.shape}")
		return edge_map

	def contrastBaseSmoothing(self, smoothing_factor: float = (1 / 9)):
		cpy_img = self.gray_image.copy()
		smoothing_kernel = np.ones((3, 3)) / smoothing_factor
		return convolution(cpy_img, smoothing_kernel)

	def _gaussian_kernel(self, size: int, sigma: float) -> np.ndarray:  # NOQA
		"""Generate a Gaussian kernel."""
		kernel = np.fromfunction(
			lambda x, y: (1 / (2 * np.pi * sigma ** 2)) * np.exp(
				- ((x - (size - 1) / 2) ** 2 + (y - (size - 1) / 2) ** 2) / (2 * sigma ** 2)),
			(size, size)
		)
		return kernel / np.sum(kernel)

	def _guassian_blure(self, sigma: float, kernel_size=7) -> np.ndarray:
		"""Apply Gaussian blur to an image."""
		cpy_img = self.gray_image.copy()
		# must be odd
		kernel = self._gaussian_kernel(kernel_size, sigma)
		return convolution(cpy_img, kernel)

	def differenceOfGaussians(self, sigma1: float, sigma2: float, kernel_size: int = 7) -> np.ndarray:
		"""Apply the Difference of Gaussian edge detection."""
		blurred1 = self._guassian_blure(sigma1, kernel_size=kernel_size)
		blurred2 = self._guassian_blure(sigma2, kernel_size=kernel_size)
		DoG = blurred1 - blurred2
		threshold = custDynamicThreshold(image=DoG, strategy=ThresholdStrategy.MEAN_PLUS_STD)
		return np.where(DoG > threshold, 255, 0).astype(np.uint8)

	def _varianceFunction(self, neighborhood: np.ndarray) -> float:  # NOQA
		mean = np.mean(neighborhood)
		std = np.std(neighborhood)
		return (mean + std) ** 2 / (len(neighborhood))

	def varianceEdgeDetector(self, kernel_size: int = 3, strategy: ThresholdStrategy = ThresholdStrategy.MEAN_PLUS_STD):

		"""
		Applies variance-based edge detection to the grayscale image using a local variance filter.

		This method uses a sliding window approach to compute the variance within each local region of the image.
		High variance regions indicate edges, as the pixel values change more drastically in these areas.
		The function applies the `_varianceFunction` to each local region using the specified kernel size.

		:param strategy:
		:param kernel_size:
				The size of the square kernel to use for calculating local variance. Default is 3.
				A larger kernel size considers a bigger neighborhood, which can smooth out finer details
				but might miss sharp edges. A smaller kernel focuses on finer, more localized edges.
		:type kernel_size: int

		:return: A binary image where edges are highlighted, with higher variance regions indicating edges.
		:rtype: np.ndarray

		:note:
			- The function relies on a helper method `custGenericFilter` that applies the variance filter to the image.
			- The `padding=True` argument ensures that the image edges are handled correctly by padding the image during
			the convolution process.

		"""
		filtered_image = custGenericFilter(self.gray_image, function=self._varianceFunction, kernel_size=kernel_size,
										   padding=True)  # NOQA
		threshold = custDynamicThreshold(image=filtered_image, strategy=strategy)
		return (filtered_image > threshold).astype(np.uint8) * 255

	def _rangeFunction(self, neighborhood: np.ndarray) -> float:  # NOQA
		return np.max(neighborhood) - np.min(neighborhood)

	def rangeEdgeDetector(self, kernel_size: int = 3, strategy: ThresholdStrategy = ThresholdStrategy.MEAN_PLUS_STD):
		filtered_image = custGenericFilter(self.gray_image, function=self._rangeFunction, kernel_size=kernel_size,
										   padding=True)  # NOQA
		threshold = custDynamicThreshold(image=filtered_image, strategy=strategy)
		return (filtered_image > threshold).astype(np.uint8) * 255
