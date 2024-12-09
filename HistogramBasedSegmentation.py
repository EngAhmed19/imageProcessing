import numpy as np
from helperFunctions import custImageToGray
from HistogramEqualization import Histogram
from Filtering import Filtering
# we can use _guassian_blure or contrastBaseSmoothing for preprocessing
from AdvancedEdgeDetection import AdvancedEdgeDetection
from enum import Enum
from scipy.ndimage import label


# from scipy.signal import find_peaks


class NoiseReductionStrategy(Enum):
	GuassianSmoothing = "Guassian Smoothing"
	MedianFiltering = "Median Filtering"


class HistogramBasedSegmentation:
	"""
	A class for performing image segmentation based on histogram analysis.

	This class includes functionality for noise reduction, contrast enhancement,
	and various histogram-based segmentation techniques, such as manual thresholding,
	peak-based segmentation, and adaptive segmentation.
	"""

	def __init__(self, image: np.ndarray,
				 noise_reduction_strategy: NoiseReductionStrategy = NoiseReductionStrategy.MedianFiltering,  # NOQA
				 sigma: float = 2, kernel_size: int = 5):  # NOQA
		if image is None or not isinstance(image, np.ndarray):
			raise ValueError("The image must be specified. please provide an image, and must be a valid numpy array")
		else:
			self.image = image
			self.gray_image = custImageToGray(image)
			self.noise_reduction_strategy = noise_reduction_strategy
			self.sigma = sigma
			self.kernel_size = kernel_size

	def noiseRedution(self, image: np.ndarray = None):
		"""
		Apply noise reduction to the image using the specified strategy.
		
		:parameter:
			:image: The image to apply noise reduction on, defaults to the original image.
		:return: The image after noise reduction.
		"""
		if image is None:
			image = self.image
		if self.noise_reduction_strategy == NoiseReductionStrategy.MedianFiltering:
			filter_ = Filtering(image)
			return filter_.applyMedianFilter(kernel_size=self.kernel_size)
		elif self.noise_reduction_strategy == NoiseReductionStrategy.GuassianSmoothing:
			filter_ = AdvancedEdgeDetection(image)
			return filter_.guassian_blure(sigma=self.sigma)

	def contrast_enhancement(self, image: np.ndarray = None):
		"""
		Enhance the contrast of the image using histogram equalization.
		:parameter:
			:param image: The image to enhance contrast for, defaults to the original image.
		:returns: The contrast-enhanced image.
		"""
		if image is None:
			image = self.image
		# return the equalized image
		return Histogram(image).histogramEqualization()[1]

	def preprocess(self, active_noiseReduction=False, active_contrast_enhancement=False):
		"""
		Preprocess the image by applying noise reduction and/or contrast enhancement.

		:parameter:
			:param active_noiseReduction: Whether to apply noise reduction, defaults to False.
			:param active_contrast_enhancement: Whether to apply contrast enhancement, defaults to False.
		:returns: The preprocessed image.
		"""
		print(f"Image preprocess (1): gray image is copied")
		if active_noiseReduction:
			self.gray_image = self.noiseRedution(image=self.gray_image)
			self.gray_image = np.uint8((self.gray_image * 255) / 255)
			print(f"Image preprocess (2): Noise Reduction is applied")
		if active_contrast_enhancement:
			self.gray_image = self.contrast_enhancement(image=self.gray_image)
			print(f"Image preprocess (3): Contrast Enhancement is applied")
		return self.gray_image

	def manual_histogram_segmentation(self, lower_threshold: float, upper_threshold: float,
									  region_grouping: bool = False):  # NOQA
		"""
		Segment the image manually using specified thresholds.

		:parameter:
			:param lower_threshold: The lower intensity threshold for segmentation.
			:param upper_threshold: The upper intensity threshold for segmentation.
			:param region_grouping: Whether to group connected regions, defaults to False.
			:return: The segmented image, and optionally labeled regions if region_grouping is True.
		"""

		segmented_img = np.zeros_like(self.gray_image)
		segmented_img[(self.gray_image >= lower_threshold) & (self.gray_image <= upper_threshold)] = 255

		if region_grouping:
			# Group connected regions using connected component labeling
			labeled_image, num_labels = label(segmented_img)
			print(f"Number of connected regions: {num_labels}")
			return segmented_img, labeled_image
		else:
			return segmented_img

	def _find_peaks(self, histogram: np.ndarray, peaks_min_distance: int = 10):  # NOQA
		peaks = []
		for i in range(1, len(histogram) - 1):
			# larger than the one before it and the one after it.
			if histogram[i] > histogram[i - 1] and histogram[i] > histogram[i + 1]:
				if len(peaks) == 0 or (abs(i - peaks[-1]) >= peaks_min_distance):
					peaks.append(i)

		return peaks

	def peak_histogram_segmentation(self, peaks_min_distance: int = 10):
		"""
		Perform segmentation by identifying peaks in the histogram.
		
		:parameter:
			:param peaks_min_distance: Minimum distance between peaks in the histogram, defaults to 10.
		:return: The segmented image.

		:raises ValueError: If not enough peaks are found for segmentation.
		"""
		histogram = Histogram(self.gray_image).getHistogram()
		# Find the peaks
		peaks = self._find_peaks(histogram=histogram, peaks_min_distance=peaks_min_distance)
		# the build in SciPy equivalent function
		# peaks, _ = find_peaks(x=histogram, distance=10)
		# print(peaks, "shape\n",peaks.shape)
		print(peaks, "shape\n", len(peaks))

		if len(peaks) < 2:
			raise ValueError("❌Not enough peaks found for segmentation.❌")

		# sort peaks based on their height (frequency)
		# False will sort ascending, True will sort descending. Default is False
		sorted_peaks = sorted(peaks, key=lambda x: histogram[x], reverse=True)
		peak_1, peak_2 = sorted_peaks[0], sorted_peaks[1]
		# middle value between the top 2 peaks
		threshold = (peak_1 + peak_2) // 2
		# print(peak_1, peak_2)
		segmented_img = np.zeros_like(self.gray_image)
		segmented_img[(self.gray_image >= threshold)] = 255
		# the commented dual threshold have an issue
		# that if the two peaks doesn't hold true for any pixels it would produce a black image (zeros)
		# return self.manual_histogram_segmentation(lower_threshold=peak_1, upper_threshold=peak_2)
		return segmented_img

	def valley_histogram_segmentation(self, peaks_min_distance: int = 10):
		"""
		Perform segmentation by identifying valleys between peaks in the histogram.

		:parameter:
			:param peaks_min_distance: Minimum distance between peaks in the histogram, defaults to 10.
		:return: The segmented image.
		:raises ValueError: If not enough peaks are found for valley segmentation.
		"""
		histogram = Histogram(self.gray_image).getHistogram()
		peaks = self._find_peaks(histogram=histogram, peaks_min_distance=peaks_min_distance)
		if len(peaks) <= 2:
			raise ValueError(
				"❌Not enough peaks found for segmentation.❌, valley histogram segmentation require to have at least 3 peaks ensure that the peaks_min_distance is valid")  # NOQA
		# sort peaks based on their height (frequency)
		# False will sort ascending, True will sort descending. Default is False
		sorted_peaks = sorted(peaks, key=lambda x: histogram[x], reverse=True)
		peak_1, peak_2 = sorted_peaks[0], sorted_peaks[1]
		minpeak, maxpeak = min(peak_1, peak_2), max(peak_1, peak_2)

		print(peak_1, peak_2)
		# print(peaks)
		# arg min finds the index of the min -> the bin in the histogram
		valley_min = histogram[minpeak:maxpeak].argmin()  # index of the minimum between the two peaks
		segmented_img = np.zeros_like(self.gray_image)
		segmented_img[(self.gray_image >= valley_min)] = 255
		return segmented_img

	def adaptive_histogram_segmentation(self, peaks_min_distance: int = 10):
		"""
		Perform adaptive histogram segmentation using mean-based adjustments.

		:parameter:
			:param peaks_min_distance: Minimum distance between peaks in the histogram, defaults to 10.
		:return: The segmented image after adaptive processing.

		:raises ValueError: If not enough peaks are found for segmentation.
		"""
		# first pass segmentation (first 5 steps in the algorithm is the same as valley segmentation)
		img = self.valley_histogram_segmentation(peaks_min_distance=peaks_min_distance)

		# Step 6: Calculate New Thresholds from Mean Segments
		# Compute new thresholds based on the means of pixel intensities in the segmented regions
		segment_mean_1 = np.mean(self.gray_image[img == 255])
		segment_mean_2 = np.mean(self.gray_image[img == 0])

		# Step 7: Adjust Thresholds Using New Means
		# new_threshold = int((segment_mean_1 + segment_mean_2) // 2)
		minmean, maxmean = min(segment_mean_1, segment_mean_2), max(segment_mean_1, segment_mean_2)

		# Step 8: Second-Pass Segmentation
		final_segmented_img = self.manual_histogram_segmentation(lower_threshold=minmean,  # NOQA
																 upper_threshold=maxmean)  # NOQA

		return final_segmented_img
