import cv2
import numpy as np


class Mask:
	def __init__(self, image: np.ndarray):
		self.image = image

	def apply_mask(self, shape_function, opacity=0.5):
		image = self.image
		mask = shape_function()
		masked_image = cv2.bitwise_and(image, image, mask=mask)
		result_image = cv2.addWeighted(image, 1 - opacity, masked_image, opacity, 0)
		return result_image

	def circle_function(self, center=None, radius=None):  # NOQA
		height, width = self.image.shape[:2]
		if center is None:
			center = (width // 2, height // 2)
		if radius is None:
			radius = min(width, height) // 2

		mask = np.zeros((height, width), dtype=np.uint8)
		mask = cv2.circle(mask, center, radius, 255, -1)  # NOQA
		return mask

	def heart_function(self):  # NOQA
		def heart_curve(t):  # NOQA
			x = 16 * (np.sin(t) ** 3)  # NOQA
			y = 13 * np.cos(t) - 5 * np.cos(2 * t) - 2 * np.cos(3 * t) - np.cos(4 * t)  # NOQA
			return x, y

		height, width = self.image.shape[:2]
		t = np.linspace(0, 2 * np.pi, 1000)
		x, y = heart_curve(t)

		x = ((x - np.min(x)) / (np.max(x) - np.min(x)) * (width - 1)).astype(np.int32)
		y = ((y - np.min(y)) / (np.max(y) - np.min(y)) * (height - 1)).astype(np.int32)

		mask = np.zeros((height, width), dtype=np.uint8)
		points = np.array([list(zip(x, y))], dtype=np.int32)
		cv2.fillPoly(mask, points, 255)  # NOQA
		return mask[::-1, :]

	def tri_function(self):  # NOQA
		height, width = self.image.shape[:2]
		mask = np.zeros((height, width), dtype=np.uint8)
		points = np.array([
			[width // 2, height // 4],
			[width // 4, 3 * height // 4],
			[3 * width // 4, 3 * height // 4]
		])
		cv2.fillPoly(mask, [points], 255)  # NOQA
		return mask
