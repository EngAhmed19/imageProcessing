import gradio as gr
import numpy as np
import matplotlib.pyplot as plt

from HalftoningAlgorithm import HalfToningImage, convertImageToGray
from BasicEdgeDetection import BasicEdgeDetection
from AdvancedEdgeDetection import AdvancedEdgeDetection, ThresholdStrategy
from HistogramEqualization import Histogram


def plot_histogram(image: np.ndarray, label: str) -> plt.Figure:
	"""
	This function show the histogram of the image in the gradio interface.
	:param image: The input image
	:param label: The label that show on the figure.
	:return: A figure with the histogram of the image
	"""
	fig, ax = plt.subplots()
	ax.hist(image.ravel(), bins=256, range=(0, 255), alpha=0.7)
	ax.set_title(label)
	ax.set_xlabel("Pixel Intensity")
	ax.set_ylabel("Frequency")
	return fig


def simpleHalfToningAlgorithm(image: np.ndarray, choice: str) -> tuple[np.ndarray, np.ndarray]:
	"""
	Apply a halftoning algorithm to an image based on the selected choice. This function supports two types of halftoning
	algorithms:

	- Simple Halftoning
	- Error Diffusion Halftoning

	The function converts the input image to grayscale before applying the halftoning algorithm.
	The result is normalized to the range [0, 1].

	:param image:The input image as a NumPy array. Must be a valid image array.
	:type image: np.ndarray

	:param choice:
		The choice of halftoning algorithm to apply. Options are:
		- "Simple Halftoning"
		- "Error Diffusion Halftoning"
	:type choice: str

	:return:
		A tuple containing:
		- The grayscale version of the input image as a NumPy array.
		- The halftoned image result as a NumPy array (normalized to [0, 1]).
	:rtype: tuple[np.ndarray, np.ndarray]

	Example usage::

		>>> gray_image_, halftoned_image = simpleHalfToningAlgorithm(image, "Simple Halftoning")


	"""
	result, gray_image_fn = None, None  # NOQA
	if image is not None:
		gray_image_fn = convertImageToGray(image)

		result: np.ndarray = np.zeros(image.shape)
		half_toning_image = HalfToningImage(image)
		if choice == "Simple Halftoning":
			result = half_toning_image.simpleHalftoning()

			result /= 255

		elif choice == "Error Diffusion Halftoning":
			result = half_toning_image.errorDiffusionHalfToning()
			result /= 255

	return gray_image_fn, result


def histogramEqualization(image: np.ndarray) -> tuple[np.ndarray, np.ndarray, plt.Figure, plt.Figure]:
	"""
	Apply the histogram equalization to an image.

	Then it plot the histogram for both the gray scale image and the equalized image.
	:param image:The input array image as numpy array.
	:type image:np.ndarray


	:returns:
		A tuple containing the following:
		- The gray image as numpy array.
		- The equalized image as numpy array.
		- The plot of the histogram for the image as a figure.
		- The plot of the histogram for the equalized image as a figure.
	:rtype: tuple[np.ndarray, np.ndarray, plt.Figure, plt.Figure]
	"""
	gray_image_gradio, equalized_image_result = None, None
	gray_hist_plot, equalized_hist_plot = None, None  # NOQA
	if image is not None:
		equalized_image = Histogram(image)
		gray_image_gradio, equalized_image_result = equalized_image.histogramEqualization()

		gray_hist_plot: plt.Figure = plot_histogram(gray_image_gradio, "Histogram of gray image")
		equalized_hist_plot: plt.Figure = plot_histogram(equalized_image_result, "Histogram of equalized image")

	return gray_image_gradio, equalized_image_result, gray_hist_plot, equalized_hist_plot

# Callback for Basic Edge Detection
def apply_basic_edge_detection(image, method, contrast_smoothing):
    basic_detector = BasicEdgeDetection(image, contrast_based_smoothing=contrast_smoothing)
    if method == "Sobel":
        return basic_detector.sobelEdgeDetection()
    elif method == "Perwitt":
        return basic_detector.perwittEdgeDetection()

# Callback for Advanced Edge Detection
def apply_advanced_edge_detection(image, method, kernel_size, threshold_strategy:ThresholdStrategy, sigma1, sigma2):
    advanced_detector = AdvancedEdgeDetection(image)
    strategy = threshold_strategy

    if method == "Homogeneity":
        return advanced_detector.homogeneityOperator(area_size=kernel_size, strategy=strategy)
    elif method == "Difference":
        return advanced_detector.differenceOperator(strategy=strategy)
    elif method == "Variance":
        return advanced_detector.varianceEdgeDetector(kernel_size=kernel_size, strategy=strategy)
    elif method == "Range":
        return advanced_detector.rangeEdgeDetector(kernel_size=kernel_size, strategy=strategy)
    elif method == "Difference of Gaussians":
        return advanced_detector.differenceOfGaussians(sigma1=sigma1, sigma2=sigma2)

with gr.Blocks() as demo:
	with gr.Tab("Halftoning Algorithms"):
		with gr.Row():
			with gr.Column():
				input_image = gr.Image(type="numpy", label="Upload Image")
				radio_choose = gr.Radio(["Simple Halftoning", "Error Diffusion Halftoning"],
										label="Choose The Algorithm",
										value="Simple Halftoning")  # NOQA
				with gr.Row():
					halftone_button = gr.Button("Apply Halftoning")
					clear_button = gr.Button("Clear")

			with gr.Column():
				gray_image = gr.Image(type="numpy", label="Gray Image")
				output_image = gr.Image(type="numpy", label="Halftoned Image")

			halftone_button.click(
				fn=simpleHalfToningAlgorithm,
				inputs=[input_image, radio_choose],
				outputs=[gray_image, output_image]
			)
			clear_button.click(
				fn=lambda: (None, None, None, "Simple Halftoning"),
				inputs=[],
				outputs=[input_image, output_image, gray_image, radio_choose]
			)
	with gr.Tab("Histogram Equalization"):
		with gr.Row():
			with gr.Column():
				input_image = gr.Image(type="numpy", label="Upload Image")
				with gr.Row():
					halftone_button = gr.Button("Apply Histogram Equalization")
					clear_button = gr.Button("Clear")

			with gr.Column():
				gray_image = gr.Image(type="numpy", label="Gray Image")
				output_image = gr.Image(type="numpy", label="Equalized Image")

			with gr.Column():
				gray_hist = gr.Plot(label="Gray Image Histogram")
				equalized_hist = gr.Plot(label="Equalized Image Histogram")

			halftone_button.click(
				fn=histogramEqualization,
				inputs=[input_image],
				outputs=[gray_image, output_image, gray_hist, equalized_hist]
			)
			clear_button.click(
				fn=lambda: (None, None, None, None, None),
				inputs=[],
				outputs=[input_image, output_image, gray_image, gray_hist, equalized_hist]
			)
	


if __name__ == '__main__':
	demo.launch(debug=True, share=True)
