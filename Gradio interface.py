import gradio as gr
import numpy as np
import matplotlib.pyplot as plt

from HalftoningAlgorithm import HalfToningImage
from BasicEdgeDetection import BasicEdgeDetection
from AdvancedEdgeDetection import AdvancedEdgeDetection
from HistogramEqualization import Histogram
from Filtering import Filtering
from ImageOperation import ImageOperation

from helperFunctions import convertImageToGray, ThresholdStrategy


def select_threshold_strategy(threshold_strategy_str: str) -> ThresholdStrategy:
	threshold_strategy: ThresholdStrategy | None = None
	if threshold_strategy_str == "Mean":
		threshold_strategy = ThresholdStrategy.MEAN
	elif threshold_strategy_str == "Median":
		threshold_strategy = ThresholdStrategy.MEDIAN
	elif threshold_strategy_str == "Standerd deviation":
		threshold_strategy = ThresholdStrategy.STD
	elif threshold_strategy_str == "Mean+Std":
		threshold_strategy = ThresholdStrategy.MEAN_PLUS_STD
	elif threshold_strategy_str == "Mean-Std":
		threshold_strategy = ThresholdStrategy.MEAN_MINUS_STD
	elif threshold_strategy_str == "Median+Std":
		threshold_strategy = ThresholdStrategy.MEDIAN_PLUS_STD

	return threshold_strategy


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


def simpleHalfToningAlgorithm(image: np.ndarray, choice: str, threshold_strategy_str: str) -> tuple[
	np.ndarray, np.ndarray]:
	"""
	Apply a halftoning algorithm to an image based on the selected choice. This function supports two types of halftoning
	algorithms:

	- Simple Halftoning
	- Error Diffusion Halftoning

	The function converts the input image to grayscale before applying the halftoning algorithm.
	The result is normalized to the range [0, 1].

	:parameter:
		:param image:The input image as a NumPy array. Must be a valid image array.
		:type image: np.ndarray
		:param choice:
			The choice of halftoning algorithm to apply. Options are:
			- "Simple Halftoning"
			- "Error Diffusion Halftoning"
		:type choice: str
		:param threshold_strategy_str: which threshold strategy will be applied to the image.
		:type threshold_strategy_str:str

	:return:
		A tuple containing:
		- The grayscale version of the input image as a NumPy array.
		- The halftoned image result as a NumPy array (normalized to [0, 1]).
	:rtype: tuple[np.ndarray, np.ndarray]

	Example usage::

		>>> gray_image_, halftoned_image = simpleHalfToningAlgorithm(image, "Simple Halftoning")


	"""
	threshold_strategy = select_threshold_strategy(threshold_strategy_str)
	result, gray_image_fn = None, None  # NOQA
	if image is not None:
		gray_image_fn = convertImageToGray(image)

		result: np.ndarray = np.zeros(image.shape)
		half_toning_image = HalfToningImage(image)
		if choice == "Simple Halftoning":
			result = half_toning_image.simpleHalftoning(threshold_strategy)

			result /= 255

		elif choice == "Error Diffusion Halftoning":
			result = half_toning_image.errorDiffusionHalfToning(threshold_strategy)
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


def customDirectionToString(directions: np.ndarray) -> str:
	return "\n".join([" ".join(row) for row in directions])


# Callback for Basic Edge Detection
def apply_basic_edge_detection(image: np.ndarray, method: str, contrast_smoothing: bool,
							   threshold_strategy_str: str) -> tuple[  # NOQA
	np.ndarray, np.ndarray | None, str | np.ndarray]:  # NOQA
	"""
	Apply deffirent basic algorithm for edge detection including:
		1- Sobel Algorithm.

		2- Perwitt Algorithm.

		3- Kirsch Algorithm.

	After that choosing the threshold strategy you want to apply on the image.And choose to apply contrast based smoothing
	or not.


	:parameter:
		:param image: The input image.
		:type image: np.ndarray
		:param method: Choice from deffirent algorithms like (sobel,perwitt,...,etc.)
		:type method:str
		:param contrast_smoothing: Whether to apply a contrast based smoothing or not.
		:type contrast_smoothing: bool
		:param threshold_strategy_str: which threshold strategy will be applied to the image.
		:type threshold_strategy_str: str

	:return: A tuple containing 3 outputs:
			1- Original image in gray scale
			2- Image after applying edge detection algorithm (sobel,perwitt,...,etc.) with the threshold strategy
			(mean,median,...,etc.).
			3- The direction of each pixel in the image after applying edge detection algorithm.
	:rtype: tuple[np.ndarray, np.ndarray | None, str | np.ndarray]
	"""

	gray_image_basic_edge, edge, direction_text = None, None, ""  # NOQA
	if image is not None:
		threshold_strategy: ThresholdStrategy = select_threshold_strategy(threshold_strategy_str)
		gray_image_basic_edge: np.ndarray = convertImageToGray(image)
		basic_detector: BasicEdgeDetection = BasicEdgeDetection(image, contrast_based_smoothing=contrast_smoothing)
		if method == "Sobel":
			result = basic_detector.sobelEdgeDetection(threshold_strategy)
			return gray_image_basic_edge, result, ""
		elif method == "Perwitt":
			result = basic_detector.perwittEdgeDetection(threshold_strategy)
			return gray_image_basic_edge, result, ""
		elif method == "Kirsch":
			edge, direction = basic_detector.kirschEdgeDetectionWithDirection(threshold_strategy)
			direction_text: str = customDirectionToString(direction)
			return gray_image_basic_edge, edge, direction_text
	return gray_image_basic_edge, edge, direction_text


# Callback for Advanced Edge Detection
def apply_advanced_edge_detection(image, method, kernel_size: int, threshold_strategy_str: str, sigma1,
								  sigma2) -> tuple[np.ndarray, np.ndarray]:  # NOQA
	"""
	Apply deffirent Advanced algorithm for edge detection including:
		1- Homogeneity Algorithm.

		2- Difference Algorithm.

		3- Variance Algorithm.

		4- Range Algorithm.

		5- Difference of Gaussians Algorithm.

		6- Contrast based smoothing Algorithm.

	After that choosing the threshold strategy you want to apply on the image.

	:parameter:
		:param image: The input image.
		:type image: np.ndarray
		:param method: The choice between deffirent algorithms (Homogeneity,Range,...,etc.).
		:type method: str
		:param kernel_size: The size of the square kernel to use for calculating local variance. Default is 3.
		:type kernel_size: int
		:param threshold_strategy_str: which threshold strategy will be applied to the image.
		:type threshold_strategy_str: str
		:param sigma1: The first sigma to apply.
		:type sigma1: float
		:param sigma2: The second sigma to apply.
		:type sigma2:float

	:returns: A tuple containing 2 outputs:
			1- Original image in gray scale
			2- Image after applying advanced edge detection algorithm (Homogeneity,Range,...,etc.) with the threshold strategy
			(mean,median,...,etc.).
	:rtype: tuple[np.ndarray, np.ndarray].
	"""
	if image is not None:
		threshold_strategy: ThresholdStrategy = select_threshold_strategy(threshold_strategy_str)
		advanced_detector = AdvancedEdgeDetection(image)
		gray_image_Advance_edge: np.ndarray = convertImageToGray(image)
		if method == "Homogeneity":
			return gray_image_Advance_edge, advanced_detector.homogeneityOperator(area_size=kernel_size,
																				  strategy=threshold_strategy)  # NOQA
		elif method == "Difference":
			return gray_image_Advance_edge, advanced_detector.differenceOperator(strategy=threshold_strategy)
		elif method == "Variance":
			return gray_image_Advance_edge, advanced_detector.varianceEdgeDetector(kernel_size=kernel_size,
																				   strategy=threshold_strategy)  # NOQA
		elif method == "Range":
			return gray_image_Advance_edge, advanced_detector.rangeEdgeDetector(kernel_size=kernel_size,
																				strategy=threshold_strategy)  # NOQA
		elif method == "Difference of Gaussians":
			return gray_image_Advance_edge, advanced_detector.differenceOfGaussians(sigma1=sigma1, sigma2=sigma2,
																					kernel_size=kernel_size,  # NOQA
																					threshold_strategy=threshold_strategy)  # NOQA
		elif method == "Contrast based smoothing":
			return gray_image_Advance_edge, advanced_detector.contrastBaseSmoothing(threshold_strategy,
																					smoothing_factor=1 / 9)  # NOQA


def applyFiltering(image: np.ndarray, method: str, kernel_size: int = 5) -> tuple[np.ndarray, np.ndarray]:
	if image is not None:
		gray_image_filter: np.ndarray = convertImageToGray(image)
		filtering = Filtering(image)
		if method == "High Pass Filter":
			return gray_image_filter, filtering.applyHighPassFilter()
		elif method == "Low Pass Filter":
			return gray_image_filter, filtering.applyLowPassFilter()
		elif method == "Median Filter":
			return gray_image_filter, filtering.applyMedianFilter(kernel_size=kernel_size)


def applyImageOperation(image: np.ndarray, choice: str) -> tuple[np.ndarray, np.ndarray]:
	if image is not None:
		gray_image_operation: np.ndarray = convertImageToGray(image)
		operator = ImageOperation(image)

		if choice == "Add":
			return gray_image_operation, operator.addImage()
		elif choice == "Subtract":
			return gray_image_operation, operator.subtractImage()
		elif choice == "Invert":
			return gray_image_operation, operator.invertImage()


with gr.Blocks() as demo:
	with gr.Tab("Halftoning Algorithms"):
		with gr.Row():
			with gr.Column():
				input_image = gr.Image(type="numpy", label="Upload Image")
				radio_choose = gr.Radio(["Simple Halftoning", "Error Diffusion Halftoning"],
										label="Choose The Algorithm",
										value="Simple Halftoning")  # NOQA
				radio_threshold_strategy = gr.Radio(
					["Mean", "Median", "Standerd deviation", "Mean+Std", "Mean-Std", "Median+Std"],
					label="Choose The Threshold strategy",  # NOQA
					value="Mean")  # NOQA
				with gr.Row():
					halftone_button = gr.Button("Apply Halftoning")
					clear_button = gr.Button("Clear")

			with gr.Column():
				gray_image = gr.Image(type="numpy", label="Gray Image")
				output_image_halftone = gr.Image(type="numpy", label="Halftoned Image")

			halftone_button.click(
				fn=simpleHalfToningAlgorithm,
				inputs=[input_image, radio_choose, radio_threshold_strategy],
				outputs=[gray_image, output_image_halftone]
			)
			clear_button.click(
				fn=lambda: (None, None, None, "Simple Halftoning", "Mean"),
				inputs=[],
				outputs=[input_image, output_image_halftone, gray_image, radio_choose, radio_threshold_strategy]
			)
	with gr.Tab("Histogram Equalization"):
		with gr.Row():
			with gr.Column():
				input_image = gr.Image(type="numpy", label="Upload Image")
				with gr.Row():
					histogram_button = gr.Button("Apply Histogram Equalization")
					clear_button = gr.Button("Clear")

			with gr.Column():
				gray_image = gr.Image(type="numpy", label="Gray Image")
				output_image_equalization = gr.Image(type="numpy", label="Equalized Image")

			with gr.Column():
				gray_hist = gr.Plot(label="Gray Image Histogram")
				equalized_hist = gr.Plot(label="Equalized Image Histogram")

			histogram_button.click(
				fn=histogramEqualization,
				inputs=[input_image],
				outputs=[gray_image, output_image_equalization, gray_hist, equalized_hist]
			)
			clear_button.click(
				fn=lambda: (None, None, None, None, None),
				inputs=[],
				outputs=[input_image, output_image_equalization, gray_image, gray_hist, equalized_hist]
			)

	with gr.Tab("Basic Edge Detection Algorithms"):
		with gr.Row():
			with gr.Column():
				input_image = gr.Image(type="numpy", label="Upload Image")
				radio_choose_basic_edge = gr.Radio(["Sobel", "Perwitt", "Kirsch"],
												   label="Choose The Algorithm",  # NOQA
												   value="Sobel")  # NOQA
				radio_threshold_strategy = gr.Radio(
					["Mean", "Median", "Standerd deviation", "Mean+Std", "Mean-Std", "Median+Std"],
					label="Choose The Threshold strategy",  # NOQA
					value="Mean")  # NOQA
				apply_contrast = gr.Checkbox(value=False, label="Apply contrast based smoothing to an image")
				with gr.Row():
					basic_edge_button = gr.Button("Apply Edge Detection")
					clear_button = gr.Button("Clear")

			with gr.Column():
				gray_image = gr.Image(type="numpy", label="Gray Image")
				output_image_edge = gr.Image(type="numpy", label="Edge Detection Image")
				direction_output = gr.Textbox(label="Edge Directions", lines=10)

			basic_edge_button.click(
				fn=apply_basic_edge_detection,
				inputs=[input_image, radio_choose_basic_edge, apply_contrast, radio_threshold_strategy],
				outputs=[gray_image, output_image_edge, direction_output]
			)
			clear_button.click(
				fn=lambda: (None, None, None, "Sobel", False, None, "Mean"),
				inputs=[],
				outputs=[input_image, output_image_edge, gray_image, radio_choose_basic_edge, apply_contrast,
						 direction_output, radio_threshold_strategy]  # NOQA
			)

	with gr.Tab("Advanced Edge Detection Algorithms"):
		with gr.Row():
			with gr.Column():
				input_image = gr.Image(type="numpy", label="Upload Image")
				radio_choose_advanced_edge = gr.Radio(
					["Homogeneity", "Difference", "Variance", "Range", "Difference of Gaussians",
					 "Contrast based smoothing"],  # NOQA
					label="Choose The Algorithm",  # NOQA
					value="Homogeneity")  # NOQA
				kernel_size_gradio = gr.Slider(minimum=1, maximum=9, value=3, step=2, label="Kernel Size")
				sigma1_gradio = gr.Number(label="Enter sigma1")
				sigma2_gradio = gr.Number(label="Enter sigma2")
				radio_threshold_strategy = gr.Radio(
					["Mean", "Median", "Standerd deviation", "Mean+Std", "Mean-Std", "Median+Std"],
					label="Choose The Threshold strategy",  # NOQA
					value="Mean")  # NOQA
				with gr.Row():
					edge_detection_button = gr.Button("Apply Edge Detection")
					clear_button = gr.Button("Clear")

			with gr.Column():
				gray_image = gr.Image(type="numpy", label="Gray Image")
				output_image_edge_advance = gr.Image(type="numpy", label="Edge Detection Image")

			edge_detection_button.click(
				fn=apply_advanced_edge_detection,
				inputs=[input_image, radio_choose_advanced_edge, kernel_size_gradio, radio_threshold_strategy,
						sigma1_gradio, sigma2_gradio],  # NOQA
				outputs=[gray_image, output_image_edge_advance]
			)
			clear_button.click(
				fn=lambda: (None, None, None, "Homogeneity", "Mean", 3),
				inputs=[],
				outputs=[input_image, output_image_edge_advance, gray_image, radio_choose_basic_edge,
						 radio_threshold_strategy, kernel_size_gradio]  # NOQA
			)
	with gr.Tab("Filtering Algorithms"):
		with gr.Row():
			with gr.Column():
				input_image = gr.Image(type="numpy", label="Upload Image")
				radio_choose = gr.Radio(["High Pass Filter", "Low Pass Filter", "Median Filter"],
										label="Choose The Type Of Filter",
										value="High Pass Filter")  # NOQA
				kernel_size_gradio = gr.Slider(minimum=1, maximum=9, value=5, step=2, label="Kernel Size")
				with gr.Row():
					filtering_button = gr.Button("Apply Filter")
					clear_button = gr.Button("Clear")

			with gr.Column():
				gray_image = gr.Image(type="numpy", label="Gray Image")
				output_image_filtering = gr.Image(type="numpy", label="Filtered Image")

			filtering_button.click(
				fn=applyFiltering,
				inputs=[input_image, radio_choose, kernel_size_gradio],
				outputs=[gray_image, output_image_filtering]
			)
			clear_button.click(
				fn=lambda: (None, None, None, "High Pass Filter", 5),
				inputs=[],
				outputs=[input_image, output_image_filtering, gray_image, radio_choose, kernel_size_gradio]
			)
	with gr.Tab("Image Operation Algorithms"):
		with gr.Row():
			with gr.Column():
				input_image = gr.Image(type="numpy", label="Upload Image")
				radio_choose = gr.Radio(["Add", "Subtract", "Invert"],
										label="Choose The Algorithm",
										value="Add")  # NOQA
				with gr.Row():
					halftone_button = gr.Button("Apply Operation")
					clear_button = gr.Button("Clear")

			with gr.Column():
				gray_image = gr.Image(type="numpy", label="Gray Image")
				output_image_operation = gr.Image(type="numpy", label="Output Image")

			halftone_button.click(
				fn=applyImageOperation,
				inputs=[input_image, radio_choose],
				outputs=[gray_image, output_image_operation]
			)
			clear_button.click(
				fn=lambda: (None, None, None, "Add"),
				inputs=[],
				outputs=[input_image, output_image_operation, gray_image, radio_choose]
			)

if __name__ == '__main__':
	demo.launch(debug=True, share=True)
