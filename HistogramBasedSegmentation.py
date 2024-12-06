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
    def __init__(self, image:np.ndarray, noise_reduction_strategy:NoiseReductionStrategy=NoiseReductionStrategy.MedianFiltering,sigma:float = 2, kernel_size:int = 5):
        if image is None or not isinstance(image, np.ndarray):
            raise ValueError("The image must be specified. please provide an image, and must be a valid numpy array")
        else:
            self.image = image
            self.gray_image = custImageToGray(image)
            self.noise_reduction_strategy = noise_reduction_strategy
            self.sigma = sigma
            self.kernel_size = kernel_size
    def noiseRedution(self, image:np.ndarray=None):
        if image is None:
            image = self.image
        if self.noise_reduction_strategy == NoiseReductionStrategy.MedianFiltering:
            filter = Filtering(image)
            return filter.applyMedianFilter(kernel_size=self.kernel_size)
        elif self.noise_reduction_strategy == NoiseReductionStrategy.GuassianSmoothing:
            filter = AdvancedEdgeDetection(image)
            return filter._guassian_blure(sigma=self.sigma)
    def contrast_enhancment(self, image:np.ndarray=None):
        if image is None:
            image = self.image
        # return the equalized image
        return Histogram(image).histogramEqualization()[1]
        
    def preprocess(self, active_noiseReduction=False, active_contrastEnhancment=False):
        cpy_img = self.gray_image.copy()
        print(f"Image preprocess (1): gray image is copied")
        if active_noiseReduction:
            cpy_img = self.noiseRedution(image=cpy_img)
            print(f"Image preprocess (2): Noise Reduction is applied")
        if active_contrastEnhancment:
            cpy_img = self.contrast_enhancment(image=cpy_img)
            print(f"Image preprocess (3): Contrast Enhancment is applied")
        return cpy_img
    def manual_histogram_segmentation(self, lower_threshold: int, upper_threshold: int, region_grouping: bool = False):
        segmented_img = np.zeros_like(self.gray_image)
        segmented_img[(self.gray_image >= lower_threshold) & (self.gray_image <= upper_threshold)] = 255

        if region_grouping:
            # Group connected regions using connected component labeling
            labeled_image, num_labels = label(segmented_img)
            print(f"Number of connected regions: {num_labels}")
            return segmented_img, labeled_image
        else:
            return segmented_img
        

    def _find_peaks(self, histogram: np.ndarray, peaks_min_distance:int =10):
        peaks = []
        for i in range(1, len(histogram)-1):
            # larger than the one before it and the one after it.
            if histogram[i] > histogram[i-1] and histogram[i] > histogram[i+1]:
                # Applying peak spacing
                if len(peaks)>1:
                    print(i,"->", peaks[-1])
                if len(peaks) == 0 or (abs(i - peaks[-1]) >= peaks_min_distance):
                    peaks.append(i)
                
        return peaks
    
    def peak_histogram_segmentation(self, peaks_min_distance:int =10):
        histogram = Histogram(self.gray_image).getHistogram()
        # Find the peaks
        peaks = self._find_peaks(histogram=histogram, peaks_min_distance=peaks_min_distance)


        if len(peaks) < 2:
            raise ValueError("❌Not enough peaks found for segmentation.❌")
        
        # sort peaks based on their height (frequency)
        # False will sort ascending, True will sort descending. Default is False
        sorted_peaks = sorted(peaks, key=lambda x: histogram[x], reverse=True)
        peak_1, peak_2 = sorted_peaks[0], sorted_peaks[1]
        threshold = (peak_1+peak_2)//2
        print(peak_1, peak_2)
        segmented_img = np.zeros_like(self.gray_image)
        segmented_img[(self.gray_image >= threshold)] = 255
        # the commented dual threshold have an issue 
        # that if the two peaks doesn't hold true for any pixels it would produce a black iamge (zeros)
        # return self.manual_histogram_segmentation(lower_threshold=peak_1, upper_threshold=peak_2)
        return segmented_img
    
    def valley_histogram_segmentation(self, peaks_min_distance:int =10):
        histogram = Histogram(self.gray_image).getHistogram()
        peaks = self._find_peaks(histogram=histogram, peaks_min_distance=peaks_min_distance)
        if len(peaks) <= 2:
            raise ValueError("❌Not enough peaks found for segmentation.❌, valley histogram segmentation require to have at least 3 peaks ensure that the peaks_min_distance is valid")
        # sort peaks based on their height (frequency)
        # False will sort ascending, True will sort descending. Default is False
        sorted_peaks = sorted(peaks, key=lambda x: histogram[x], reverse=True)
        peak_1, peak_2 = sorted_peaks[0], sorted_peaks[1]
        print(peaks)
        # argmin finds the index of the min -> the bin in the histogram
        valley_min = histogram[peak_1:peak_2].argmin() #index of the minimum between the two peaks
        segmented_img = np.zeros_like(self.gray_image)
        segmented_img[(self.gray_image >= valley_min)] = 255
        return segmented_img
        



    





    
    

