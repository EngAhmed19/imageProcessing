import numpy as np
from helperFunctions import convertImageToGray
from copy import copy


class ImageOperation:
	def __init__(self, image: np.ndarray):
		if image is None or not isinstance(image, np.ndarray):
			raise ValueError("The image must be specified. please provide an image")
		else:
			self.image = image
			self.gray_image = convertImageToGray(self.image)
			self.cpy_image = copy(self.gray_image)

	def addImage(self, another_image: np.ndarray) -> np.ndarray:
		"""
		Add 2 images together.
		:return: the result of addition of 2 images.
		:rtype: np.ndarray
		"""
		added_image: np.ndarray = self.image + another_image
		added_image = np.clip(added_image, 0, 255)
		return added_image

	def subtractImage(self, another_image: np.ndarray) -> np.ndarray:
		"""
		subtract 2 images.
		:return: the result of subtraction of 2 images.
		:rtype: np.ndarray
		"""
		subtracted_image: np.ndarray = self.image - another_image
		subtracted_image = np.clip(subtracted_image, 0, 255)
		return subtracted_image

	def invertImage(self) -> np.ndarray:
		"""
		Invert The image.
		:return: the result of inverting an image.
		:rtype: np.ndarray
		"""
		inverted_image: np.ndarray = 255 - self.image
		return inverted_image
