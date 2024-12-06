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
            print(f"Image preprocess (2): Noise Reduction is done")
        if active_contrastEnhancment:
            cpy_img = self.contrast_enhancment(image=cpy_img)
            print(f"Image preprocess (2): Noise Reduction is done")
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



    
    

