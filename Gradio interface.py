import gradio as gr
import numpy as np
import matplotlib.pyplot as plt

from HalftoningAlgorithm import HalfToningImage
from BasicEdgeDetection import BasicEdgeDetection
from AdvancedEdgeDetection import AdvancedEdgeDetection
from HistogramEqualization import Histogram
from Filtering import Filtering
from ImageOperation import ImageOperation
from HistogramBasedSegmentation import HistogramBasedSegmentation, NoiseReductionStrategy
from ImageFlipper import ImageFlipper
from ColorManipulator import ColorManipulator
from Mask import Mask

from helperFunctions import convertImageToGray, ThresholdStrategy, resize


def select_noise_reduction_strategy(noise_reduction_strategy_str: str) -> NoiseReductionStrategy:
	noise_reduction_strategy: NoiseReductionStrategy = NoiseReductionStrategy.GuassianSmoothing
	if noise_reduction_strategy_str == "Guassian Smoothing":
		noise_reduction_strategy = NoiseReductionStrategy.GuassianSmoothing
	elif noise_reduction_strategy_str == "Median Filtering":
		noise_reduction_strategy = NoiseReductionStrategy.MedianFiltering

	return noise_reduction_strategy


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


def simpleHalfToningAlgorithm(image: np.ndarray, choice: str, threshold_strategy_str: str,
							  bayer_matrix_size: int = 2) -> tuple[np.ndarray, np.ndarray]:  # NOQA
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
		:param bayer_matrix_size: The bayer matrix size that is used for orderd ditherd algorithm
		:type bayer_matrix_size: int

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

		elif choice == "Ordered Dithering":
			result = half_toning_image.order_dither(bayer_matrix_size)
	return gray_image_fn, result


def update_halftoning_algorithm_control(choice: str) -> gr.update:
	if choice == "Ordered Dithering":
		return gr.update(visible=True), gr.update(visible=False)
	return gr.update(visible=False), gr.update(visible=True)


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
	"""
	Apply deffirent filtering algorithms including:
		1- High Pass Filter.

		2- Low Pass Filter.

		3- Median Filter.

	:parameter:
		:param image: The input image.
		:type image: np.ndarray
		:param method: choose between deffirent filters like (High pass filter,Low Pass Filter,...,etc.).
		:type method: str
		:param kernel_size: he size of the square kernel to use for calculating local variance. Default is 5.
		:type kernel_size: int.

	:returns: A tuple containing 2 outputs:
			1- Original image in gray scale
			2- Image after applying filter algorithm on it (High pass filter,Low Pass Filter,...,etc.).
	:rtype: tuple[np.ndarray, np.ndarray]
	"""
	if image is not None:
		gray_image_filter: np.ndarray = convertImageToGray(image)
		filtering = Filtering(image)
		if method == "High Pass Filter":
			return gray_image_filter, filtering.applyHighPassFilter()
		elif method == "Low Pass Filter":
			return gray_image_filter, filtering.applyLowPassFilter()
		elif method == "Median Filter":
			return gray_image_filter, filtering.applyMedianFilter(kernel_size=kernel_size)


def applyImageOperation(image1: np.ndarray, image2: np.ndarray, choice: str, radio_choose_flipping_str: str,
						red_factor_float: float, new_width: int, new_height: int,  # NOQA
						green_factor: float, blue_factor: float, mask_choice: str) -> np.ndarray:  # NOQA
	"""
	Apply deffirent Operation on Image including:
		1- Addition

		2- Subtraction.

		3- Inversion.


	:parameter:
		:param image1: The first input image.
		:type image1: np.ndarray
		:param image2: The second input image.
		:type image2: np.ndarray
		:param new_width: The second input image.
		:type new_width: int
		:param new_height: The second input image.
		:type new_height: int
		:param choice: choose between deffirent operations like (Addition,Subtraction,...,etc.).
		:type choice: str
		:param radio_choose_flipping_str: The flipping option.
		:type radio_choose_flipping_str: str
		:param red_factor_float: The red factor.
		:type red_factor_float: float
		:param blue_factor: The blue factor.
		:type blue_factor: float
		:param green_factor: The green factor.
		:type green_factor: float
		:param mask_choice: The choice of the mask
		:type mask_choice: str

	:returns: Image after applying operation on it (Addition,Subtraction,...,etc.).
	:rtype: np.ndarray
	"""
	resized_image1 = resize(image1, new_width, new_height)
	if image2 is not None:
		resized_image2 = resize(image2, new_width, new_height)
	if image1 is not None:
		operator1 = ImageOperation(resized_image1)
		operator2 = ImageOperation(image1)

		if choice == "Add":
			return operator1.addImage(resized_image2)  # NOQA
		elif choice == "Subtract":
			return operator1.subtractImage(resized_image2)  # NOQA
		elif choice == "Invert":
			return operator2.invertImage()
		elif choice == "Flipping":
			flipper = ImageFlipper(image1)
			return flipper.flip(radio_choose_flipping_str)
		elif choice == "Color Manipulation":
			color_manipulator = ColorManipulator(image1)
			return color_manipulator.apply_color_filter(red_factor_float, green_factor, blue_factor)  # NOQA
		elif choice == "Mask":
			masker = Mask(image1)
			if mask_choice == "Circle":
				return masker.apply_mask("circle")
			if mask_choice == "Heart":
				return masker.apply_mask("heart")
			if mask_choice == "Triangle":
				return masker.apply_mask("triangle")


def update_image_operation_control(choice: str) -> gr.update:
	if choice in ["Add", "Subtract"]:
		return gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(
			visible=False), gr.update(visible=False), gr.update(visible=True), gr.update(visible=True), gr.update(
			visible=True)
	elif choice == "Flipping":
		return gr.update(visible=True), gr.update(visible=False), gr.update(visible=False), gr.update(
			visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(
			visible=False)
	elif choice == "Color Manipulation":
		return gr.update(visible=False), gr.update(visible=True), gr.update(visible=True), gr.update(
			visible=True), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(
			visible=False)
	elif choice == "Mask":
		return gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(
			visible=False), gr.update(visible=True), gr.update(visible=False), gr.update(visible=False), gr.update(
			visible=False)
	return gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(
		visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(
		visible=False)


def applyHistogramBasedSegmentation(image: np.ndarray, choice: str, noise_reduction_strategy_str: str,
									lower_threshold: float, upper_threshold: float, min_peaks: int = 10,  # NOQA
									sigma: int = 3,
									kernel_size: int = 5,
									active_contrast_enhancement: bool = False) -> tuple[np.ndarray, np.ndarray]:  # NOQA
	noise_reduction_strategy = select_noise_reduction_strategy(noise_reduction_strategy_str)
	if image is not None:
		gray_image_segmentation: np.ndarray = convertImageToGray(image)
		segmentor = HistogramBasedSegmentation(image, noise_reduction_strategy, sigma, kernel_size)
		_ = segmentor.preprocess(True, active_contrast_enhancement)
		if choice == "Manual histogram segmentation":
			return gray_image_segmentation, segmentor.manual_histogram_segmentation(lower_threshold, upper_threshold)
		elif choice == "Peak histogram segmentation":
			return gray_image_segmentation, segmentor.peak_histogram_segmentation(min_peaks)
		elif choice == "Valley histogram segmentation":
			return gray_image_segmentation, segmentor.valley_histogram_segmentation(min_peaks)
		elif choice == "Adaptive histogram segmentation":
			return gray_image_segmentation, segmentor.adaptive_histogram_segmentation(min_peaks)


def update_advanced_edge_controls(choice: str) -> gr.update:
	if choice in ["Variance", "Range", "Homogeneity"]:
		return gr.update(visible=True), gr.update(visible=False), gr.update(visible=False)
	elif choice in ["Difference of Gaussians"]:
		return gr.update(visible=True), gr.update(visible=True), gr.update(visible=True)
	return gr.update(visible=False), gr.update(visible=False), gr.update(visible=False)


def update_histogram_based_segmentation_control(choice: str) -> gr.update:
	if choice == "Manual histogram segmentation":
		return gr.update(visible=True), gr.update(visible=True), gr.update(visible=False)
	return gr.update(visible=False), gr.update(visible=False), gr.update(visible=True)


with gr.Blocks() as demo:
	with gr.Tab("Halftoning Algorithms"):
		with gr.Row():
			with gr.Column():
				input_image = gr.Image(type="numpy", label="Upload Image")
				radio_choose = gr.Radio(["Simple Halftoning", "Error Diffusion Halftoning", "Ordered Dithering"],
										label="Choose The Algorithm",
										value="Simple Halftoning")  # NOQA
				bayer_matrix_size = gr.Slider(minimum=2, maximum=8, value=4, step=2, label="Bayer Matrix Size",
											  visible=False)  # NOQA
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
				inputs=[input_image, radio_choose, radio_threshold_strategy, bayer_matrix_size],
				outputs=[gray_image, output_image_halftone]
			)
			clear_button.click(
				fn=lambda: (None, None, None, "Simple Halftoning", "Mean", 4),
				inputs=[],
				outputs=[input_image, output_image_halftone, gray_image, radio_choose, radio_threshold_strategy,
						 bayer_matrix_size]  # NOQA
			)
			radio_choose.change(
				fn=update_halftoning_algorithm_control,
				inputs=[radio_choose],
				outputs=[bayer_matrix_size, radio_threshold_strategy]
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
				kernel_size_gradio = gr.Slider(minimum=3, maximum=9, value=3, step=2, label="Kernel Size",
											   visible=True)  # NOQA
				sigma1_gradio = gr.Number(label="Enter sigma1", visible=False)
				sigma2_gradio = gr.Number(label="Enter sigma2", visible=False)
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
			radio_choose_advanced_edge.change(
				fn=update_advanced_edge_controls,
				inputs=[radio_choose_advanced_edge],
				outputs=[kernel_size_gradio, sigma1_gradio, sigma2_gradio]
			)
	with gr.Tab("Filtering Algorithms"):
		with gr.Row():
			with gr.Column():
				input_image = gr.Image(type="numpy", label="Upload Image")
				radio_choose = gr.Radio(["High Pass Filter", "Low Pass Filter", "Median Filter"],
										label="Choose The Type Of Filter",
										value="High Pass Filter")  # NOQA
				kernel_size_gradio = gr.Slider(minimum=3, maximum=9, value=5, step=2, label="Kernel Size")
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
				input_image1 = gr.Image(type="numpy", label="Upload Image1")
				input_image2 = gr.Image(type="numpy", label="Upload Image2", visible=True)
				radio_choose = gr.Radio(["Add", "Subtract", "Invert", "Flipping", "Color Manipulation", "Mask"],
										label="Choose The Algorithm",
										value="Add")  # NOQA
				radio_choose_flipping = gr.Radio(["horizontal", "vertical", "horizontal-vertical"],
												 label="Choose The Flipping techniqe",  # NOQA
												 value="horizontal", visible=False)  # NOQA
				red_factor = gr.Slider(label="Red Factor", minimum=0, maximum=10, value=0.5, step=0.1, visible=False)
				green_factor = gr.Slider(label="Green Factor", minimum=0, maximum=10, value=0.5, step=0.1,
										 visible=False)  # NOQA
				blue_factor = gr.Slider(label="Blue Factor", minimum=0, maximum=10, value=0.5, step=0.1, visible=False)
				radio_choose_Mask = gr.Radio(["Circle", "Heart", "Triangle"],
											 label="Choose The Mask",  # NOQA
											 value="Circle", visible=False)  # NOQA
				new_width = gr.Number(label="Enter new width", minimum=1, visible=True, value=400)
				new_height = gr.Number(label="Enter new height", minimum=1, visible=True, value=400)
				with gr.Row():
					halftone_button = gr.Button("Apply Operation")
					clear_button = gr.Button("Clear")

			with gr.Column():
				output_image_operation = gr.Image(type="numpy", label="Output Image")

			halftone_button.click(
				fn=applyImageOperation,
				inputs=[input_image1, input_image2, radio_choose, radio_choose_flipping, red_factor, new_width,
						new_height, green_factor, blue_factor, radio_choose_Mask],  # NOQA
				outputs=[output_image_operation]
			)
			clear_button.click(
				fn=lambda: (None, None, "Add", "horizontal", 1, 1, 1, "Circle", None),
				inputs=[],
				outputs=[input_image1, output_image_operation, radio_choose, radio_choose_flipping,
						 red_factor, green_factor, blue_factor, radio_choose_Mask, input_image2]  # NOQA
			)
			radio_choose.change(
				fn=update_image_operation_control,
				inputs=[radio_choose],
				outputs=[radio_choose_flipping, red_factor, green_factor, blue_factor, radio_choose_Mask, input_image2,
						 new_width, new_height]  # NOQA
			)
	with gr.Tab("Histogram Based Segmentation Algorithms"):
		with gr.Row():
			with gr.Column():
				input_image = gr.Image(type="numpy", label="Upload Image")
				radio_choose_histogram_segmentation = gr.Radio(
					["Manual histogram segmentation", "Peak histogram segmentation", "Valley histogram segmentation",
					 "Adaptive histogram segmentation"],  # NOQA
					label="Choose The Algorithm",  # NOQA
					value="Manual histogram segmentation")  # NOQA
				kernel_size_gradio = gr.Slider(minimum=3, maximum=9, value=5, step=2, label="Kernel Size")
				sigma_gradio = gr.Number(label="Enter sigma")
				threshold1 = gr.Number(label="Enter Lower Threshold", visible=True)
				threshold2 = gr.Number(label="Enter Upper Threshold", visible=True)
				min_peaks = gr.Number(label="Choose minimum peak", visible=False, value=10, minimum=1)
				radio_noise_reduction_strategy = gr.Radio(
					["Guassian Smoothing"],
					label="Choose The Noise Reduction Strategy",  # NOQA
					value="Guassian Smoothing")  # NOQA
				apply_contrast_enhancement = gr.Checkbox(value=False, label="Apply contrast enhancement to an image")
				with gr.Row():
					edge_detection_button = gr.Button("Apply Histogram Based Segmentation")
					clear_button = gr.Button("Clear")

			with gr.Column():
				gray_image = gr.Image(type="numpy", label="Gray Image")  # NOQA
				output_image_histogram_segmented = gr.Image(type="numpy", label="Histogram Based Segmentation Image")

			edge_detection_button.click(
				fn=applyHistogramBasedSegmentation,
				inputs=[input_image, radio_choose_histogram_segmentation, radio_noise_reduction_strategy, threshold1,
						threshold2, min_peaks, sigma_gradio, kernel_size_gradio, apply_contrast_enhancement],  # NOQA
				outputs=[gray_image, output_image_histogram_segmented]
			)
			clear_button.click(
				fn=lambda: (
					None, None, None, "Manual histogram segmentation", "Guassian Smoothing", 5, 2, 0, 0,
					False, 10),
				inputs=[],
				outputs=[input_image, output_image_histogram_segmented, gray_image, radio_choose_histogram_segmentation,
						 radio_noise_reduction_strategy, kernel_size_gradio, sigma_gradio  # NOQA
					, threshold1, threshold2, apply_contrast_enhancement, min_peaks]  # NOQA
			)
			radio_choose_histogram_segmentation.change(
				fn=update_histogram_based_segmentation_control,
				inputs=[radio_choose_histogram_segmentation],
				outputs=[threshold1, threshold2, min_peaks]
			)

if __name__ == '__main__':
	demo.launch(debug=True, share=True)
