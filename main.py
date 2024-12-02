import cv2
import matplotlib.pyplot as plt
import numpy as np


def error_diffusion_halftoning(image):
	# Convert image to grayscale if it is not
	if len(image.shape) == 3:
		image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

	# Normalize image to 0-1
	normalized_image = image / 255.0

	# Create an empty image to store the halftoned result
	halftone_image = np.zeros_like(normalized_image)

	# Loop through each pixel
	for y in range(normalized_image.shape[0]):
		for x in range(normalized_image.shape[1]):
			old_pixel = normalized_image[y, x]
			new_pixel = 1.0 if old_pixel >= 0.5 else 0.0
			halftone_image[y, x] = new_pixel
			quant_error = old_pixel - new_pixel

			# Distribute the quantization error to neighbors with explicit checks
			# Right neighbor
			if x + 1 < normalized_image.shape[1]:
				normalized_image[y, x + 1] += quant_error * 7 / 16

			# Bottom-left neighbor
			if x - 1 >= 0 and y + 1 < normalized_image.shape[0]:
				normalized_image[y + 1, x - 1] += quant_error * 3 / 16

			# Bottom neighbor
			if y + 1 < normalized_image.shape[0]:
				normalized_image[y + 1, x] += quant_error * 5 / 16

			# Bottom-right neighbor
			if x + 1 < normalized_image.shape[1] and y + 1 < normalized_image.shape[0]:
				normalized_image[y + 1, x + 1] += quant_error * 1 / 16

	# Convert back to 0-255
	halftone_image = (halftone_image * 255).astype(np.uint8)
	return halftone_image


# Load an image
image = cv2.imread("images/nature.jpg", cv2.IMREAD_GRAYSCALE)

# Apply error diffusion halftoning
halftoned_image = error_diffusion_halftoning(image)

# Display the result
plt.imshow(halftoned_image, cmap="gray")
plt.show()
