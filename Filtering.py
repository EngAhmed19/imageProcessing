from typing import Any

import numpy as np
from numpy import floating

from helperFunctions import convolution, custImageToGray, custGenericFilter, convertImageToGray


class Filtering:
	def __init__(self, image: np.ndarray):
		if image is None or not isinstance(image, np.ndarray):
			raise ValueError("The image must be specified. please provide an image, and must be a valid numpy array")
		else:
			self.image = image
			self.gray_image = convertImageToGray(self.image)

			self.low_pass_filter = np.array(
				[
					[0, -1, 0],
					[-1, 5, -1],
					[0, -1, 0]
				]
			)

			self.high_pass_filter = (1 / 6) * np.array(
				[
					[0, 1, 0],
					[1, 2, 1],
					[0, 1, 0]
				]
			)

	def applyLowPassFilter(self, low_pass_filter: np.ndarray = None):
		cpy_img = self.gray_image.copy()
		if low_pass_filter is None:
			filter_ = self.low_pass_filter.copy()
		else:
			filter_ = low_pass_filter
		result = convolution(cpy_img, filter_)
		result = np.uint8(255 * (result / np.max(result)))  # Normalize the image between 0-255
		return result

	def applyHighPassFilter(self, high_pass_filter: np.ndarray = None):
		cpy_img = self.gray_image.copy()
		if high_pass_filter is None:
			filter_ = self.high_pass_filter.copy()
		else:
			filter_ = high_pass_filter
		result = convolution(cpy_img, filter_)
		result = np.uint8(255 * (result / np.max(result)))  # Normalize the image between 0-255
		return result

	def _medianFunction(self, neighborhood: np.ndarray) -> floating[Any]:  # NOQA
		return np.median(neighborhood)

	def applyMedianFilter(self, kernel_size: int = 5):
		"""
		Applies Median Filter
		The median filter is considered a non-linear filter
		and does not fit into the typical categories of low-pass or high-pass filters that are usually associated with
		linear filters.
		But it behaves similarly to low-pass filter.
		Linear filters is where we apply convolution operation, a weighted sum calculated by a sliding window.
		"""
		filtered_image = custGenericFilter(self.gray_image, function=self._medianFunction, kernel_size=kernel_size,
										   padding=True)  # NOQA
		filtered_image = np.uint8(255 * (filtered_image / np.max(filtered_image)))  # Normalize the image between 0-255
		return filtered_image
