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

	def addImage(self) -> np.ndarray:
		added_image: np.ndarray = self.gray_image + self.cpy_image
		added_image = np.clip(added_image, 0, 255)
		return added_image

	def subtractImage(self) -> np.ndarray:
		subtracted_image: np.ndarray = self.gray_image - self.cpy_image
		subtracted_image = np.clip(subtracted_image, 0, 255)
		return subtracted_image

	def invertImage(self) -> np.ndarray:
		inverted_image: np.ndarray = 255 - self.gray_image
		return inverted_image
