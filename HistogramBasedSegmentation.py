import numpy as np
from helperFunctions import custImageToGray
from HistogramEqualization import Histogram
from Filtering import Filtering
# we can use _guassian_blure or contrastBaseSmoothing for preprocessing
from AdvancedEdgeDetection import AdvancedEdgeDetection
from enum import Enum

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
    def noiseRedution(self):
        if self.noise_reduction_strategy == NoiseReductionStrategy.MedianFiltering:
            filter = Filtering(image=self.image)
            return filter.applyMedianFilter(kernel_size=self.kernel_size)
        elif self.noise_reduction_strategy == NoiseReductionStrategy.GuassianSmoothing:
            filter = AdvancedEdgeDetection(image=self.image)
            return AdvancedEdgeDetection._guassian_blure(sigma=self.sigma)
    def contrast_enhancment(self):
        # return the equalized image
        return Histogram(self.image).histogramEqualization()[1]
    def preprocess(self):
        cpy_img = self.gray_image.copy()
        cpy_img = self.noiseRedution()
        cpy_img = self.contrast_enhancment()
        return cpy_img
    

    
    

