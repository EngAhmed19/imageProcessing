import gradio as gr
import numpy as np
import matplotlib.pyplot as plt

from HalftoningAlgorithm import HalfToningImage, convertImageToGray
from HistogramEqualization import Histogram


def plot_histogram(image: np.ndarray, label: str):
	fig, ax = plt.subplots()
	ax.hist(image.ravel(), bins=256, range=(0, 255), alpha=0.7)
	ax.set_title(label)
	ax.set_xlabel("Pixel Intensity")
	ax.set_ylabel("Frequency")
	return fig


def simpleHalfToningAlgorithm(image: np.ndarray, choice: str) -> tuple[np.ndarray, np.ndarray]:
	result, gray_image_fn = None, None # NOQA
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
	gray_image_gradio, equalized_image_result = None, None
	gray_hist_plot, equalized_hist_plot = None, None # NOQA
	if image is not None:
		equalized_image = Histogram(image)
		gray_image_gradio, equalized_image_result = equalized_image.histogramEqualization()

		gray_hist_plot: plt.Figure = plot_histogram(gray_image_gradio, "Histogram of gray image")
		equalized_hist_plot: plt.Figure = plot_histogram(equalized_image_result, "Histogram of equalized image")

	return gray_image_gradio, equalized_image_result, gray_hist_plot, equalized_hist_plot


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
