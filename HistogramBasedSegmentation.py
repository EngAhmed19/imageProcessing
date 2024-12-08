import numpy as np
from helperFunctions import custImageToGray
from HistogramEqualization import Histogram
from Filtering import Filtering
# we can use _guassian_blure or contrastBaseSmoothing for preprocessing
from AdvancedEdgeDetection import AdvancedEdgeDetection
from enum import Enum
from scipy.ndimage import label


class NoiseReductionStrategy(Enum):
	GuassianSmoothing = "Guassian Smoothing"
	MedianFiltering = "Median Filtering"


class HistogramBasedSegmentation:
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
		if image is None:
			image = self.image
		if self.noise_reduction_strategy == NoiseReductionStrategy.MedianFiltering:
			filter_ = Filtering(image)
			return filter_.applyMedianFilter(kernel_size=self.kernel_size)
		elif self.noise_reduction_strategy == NoiseReductionStrategy.GuassianSmoothing:
			filter_ = AdvancedEdgeDetection(image)
			return filter_.guassian_blure(sigma=self.sigma)

	def contrast_enhancement(self, image: np.ndarray = None):
		if image is None:
			image = self.image
		# return the equalized image
		return Histogram(image).histogramEqualization()[1]

	def preprocess(self, active_noiseReduction=False, active_contrast_enhancement=False):
		print(f"Image preprocess (1): gray image is copied")
		if active_noiseReduction:
			self.gray_image = self.noiseRedution(image=self.gray_image)
			print(f"Image preprocess (2): Noise Reduction is applied")
		if active_contrast_enhancement:
			self.gray_image = self.contrast_enhancement(image=self.gray_image)
			print(f"Image preprocess (3): Contrast Enhancement is applied")
		return self.gray_image

	def manual_histogram_segmentation(self, lower_threshold: np.floating[int], upper_threshold: np.floating[int],
									  region_grouping: bool = False):  # NOQA
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
		histogram = Histogram(self.gray_image).getHistogram()
		# Find the peaks
		peaks = self._find_peaks(histogram=histogram, peaks_min_distance=peaks_min_distance)

		if len(peaks) < 2:
			raise ValueError("❌Not enough peaks found for segmentation.❌")

		# sort peaks based on their height (frequency)
		# False will sort ascending, True will sort descending. Default is False
		sorted_peaks = sorted(peaks, key=lambda x: histogram[x], reverse=True)
		peak_1, peak_2 = sorted_peaks[0], sorted_peaks[1]
		threshold = (peak_1 + peak_2) // 2
		# print(peak_1, peak_2)
		segmented_img = np.zeros_like(self.gray_image)
		segmented_img[(self.gray_image >= threshold)] = 255
		# the commented dual threshold have an issue
		# that if the two peaks doesn't hold true for any pixels it would produce a black image (zeros)
		# return self.manual_histogram_segmentation(lower_threshold=peak_1, upper_threshold=peak_2)
		return segmented_img

	def valley_histogram_segmentation(self, peaks_min_distance: int = 10):
		histogram = Histogram(self.gray_image).getHistogram()
		peaks = self._find_peaks(histogram=histogram, peaks_min_distance=peaks_min_distance)
		if len(peaks) <= 2:
			raise ValueError(
				"❌Not enough peaks found for segmentation.❌, valley histogram segmentation require to have at least 3 peaks ensure that the peaks_min_distance is valid")  # NOQA
		# sort peaks based on their height (frequency)
		# False will sort ascending, True will sort descending. Default is False
		sorted_peaks = sorted(peaks, key=lambda x: histogram[x], reverse=True)
		peak_1, peak_2 = sorted_peaks[0], sorted_peaks[1]
		print(peak_1, peak_2)
		# print(peaks)
		# arg min finds the index of the min -> the bin in the histogram
		valley_min = histogram[peak_2:peak_1].argmin()  # index of the minimum between the two peaks
		segmented_img = np.zeros_like(self.gray_image)
		segmented_img[(self.gray_image >= valley_min)] = 255
		return segmented_img

	def adaptive_histogram_segmentation(self, peaks_min_distance: int = 10):
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
		final_segmented_img = self.manual_histogram_segmentation(lower_threshold=minmean, upper_threshold=maxmean)

		return final_segmented_img
