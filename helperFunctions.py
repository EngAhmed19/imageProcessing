import numpy as np
import cv2

np.set_printoptions(suppress=True)


def convertImageToGray(image: np.ndarray) -> np.ndarray:
	if len(image.shape) == 3:
		return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
	return image


def custImageToGray(image: np.ndarray) -> np.ndarray:
	"""
	Luminosity Method : Uses weighted average based on human perception of color brightness. python return np.dot(img,
	[0.2989, 0.5870, 0.1140]) we can use this but some images may have a fourth channel, so we slice only three
	channels,but using `img[..., :3]` or ` img[:, :, :3]` is taking care of this possibility Args: image (np.ndarry):
	numpy array of image pixels.

	:parameter:
		:param image: The input image

	:return: np.ndarry: numpy array represent the gray image pixels.

	"""
	# this is a binary image
	if len(image.shape) < 3:
		return image
	return np.dot(image[:, :, :3], [0.2989, 0.5870, 0.1140])


def calculateSumOfHist(histogram: np.ndarray) -> np.ndarray:
	sum_of_hist = [0] * len(histogram)
	sum_i = 0
	for i in range(len(histogram)):
		sum_i += histogram[i]
		sum_of_hist[i] = sum_i
	return np.array(sum_of_hist)


def calculateTargetSize(image: np.ndarray, kernel: int, padding: int, stride: int) -> tuple[int, int]:
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


def custGenericFilter(image: np.ndarray, function, kernel_size: int = 3, padding: bool = True, **Kwargs) -> np.ndarray:
	"""
	Custom implementation of generic filter, it applies some operation on an image using a kernel.
	It's some sort of sliding window.

	:parameter:
		:param kernel_size: The size of the window (must be an odd number).
		:param function: The custom operation (function) to apply on each window.
		:param image: The input image (grayscale).
		:param padding: To pad the image or not to pad it.
		:keyword **kwargs: Additional padding options (e.g., pad_value, pad_mode).

	:returns:np.ndarray: The resulting image after applying the custom operation.

	:raises ValueError :Kernel size must be an odd number.
	"""
	if kernel_size % 2 == 0:
		raise ValueError("Kernel size must be an odd number.")

	# 1 in case of 3×3 kernels
	# 2 in case of 5×5 kernels
	# it made so the first pixel in the image would be in the center of the kernel
	pad_size = kernel_size // 2

	if padding:
		# kwargs is a dictionary
		# get(use the value of this key, else use this default value)
		pad_value = Kwargs.get('pad_value', 0)
		pad_mode = Kwargs.get('pad_mode', 'constant')  # constant-> is the default mode in NumPy
		padded_image = np.pad(image, pad_width=pad_size, mode=pad_mode, constant_values=pad_value)

	else:
		padded_image = image

	output_image = np.zeros_like(padded_image, dtype=np.float32)

	for row in range(pad_size, padded_image.shape[0] - pad_size):  # start looping from the first row before padding
		for col in range(pad_size, padded_image.shape[1] - pad_size):  # start looping from the first column before padding
			neighborhood = padded_image[(row - pad_size): (row + pad_size + 1), (col - pad_size): (col + pad_size + 1)]

			output_image[row, col] = function(neighborhood)

	# If padding was applied, return the result without the padded edges
	if padding:
		return output_image[pad_size: -pad_size, pad_size: -pad_size]
	else:
		return output_image
