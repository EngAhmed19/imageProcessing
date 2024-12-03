import numpy as np
from helperFunctions import calculateSumOfHist, convertImageToGray


class Histogram:
	def __init__(self, image: np.ndarray):
		self.image = image
		self.gray_image = convertImageToGray(image)
		self.flat_image = self.gray_image.flatten()

	def getHistogram(self, bins: int = 256) -> np.ndarray:
		histogram: np.ndarray = np.zeros(bins)

		for pixel in self.gray_image.ravel():
			histogram[pixel] += 1

		return histogram

	def histogramEqualization(self) -> (np.ndarray, np.ndarray):
		histogram: np.ndarray = self.getHistogram(256)

		sum_of_hist: np.ndarray = calculateSumOfHist(histogram)
		row, col = self.gray_image.shape

		equalized_image = np.zeros_like(self.gray_image)
		area: int = row * col
		dm = 256

		for i in range(row):
			for j in range(col):
				k = self.gray_image[i][j]
				equalized_image[i, j] = (dm / area) * sum_of_hist[k]

		return self.gray_image, equalized_image
