import numpy as np
import cv2

np.set_printoptions(suppress=True)


def convertImageToGray(image: np.ndarray) -> np.ndarray:
	if len(image.shape) == 3:
		return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
	return image


def calculateSumOfHist(histogram: np.ndarray) -> np.ndarray:
	sum_of_hist = [0] * len(histogram)
	sum_i = 0
	for i in range(len(histogram)):
		sum_i += histogram[i]
		sum_of_hist[i] = sum_i
	return np.array(sum_of_hist)


def calculateTargetSize(image: np.ndarray, kernel: int, padding: int, stride: int) -> (int, int):
	w, h = image.shape

	out_shape: tuple[int, int] = (
		int(((w - kernel + (2 * padding) / stride) + 1)), int(((h - kernel + (2 * padding) / stride) + 1)))
	return out_shape


def convolution(image: np.ndarray, filter_mask: np.ndarray) -> np.ndarray:
	k = filter_mask.shape[0]

	target_size: tuple[int, int] = calculateTargetSize(image, kernel=k, padding=0, stride=1)

	convolved_image: np.ndarray = np.zeros(shape=(target_size[0], target_size[1]))

	for i in range(target_size[0]):
		for j in range(target_size[1]):
			matrix: np.ndarray = image[i:i + k, j:j + k]
			convolved_image[i, j] = np.sum(np.multiply(matrix, filter_mask))

	return convolved_image
