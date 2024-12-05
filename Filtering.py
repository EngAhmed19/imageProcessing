import numpy as np
from helperFunctions import convolution, custImageToGray, custGenericFilter

class Filtering:
    def __init__(self, image:np.ndarray):
        if image is None or not isinstance(image, np.ndarray):
            raise ValueError("The image must be specified. please provide an image, and must be a valid numpy array")
        else:
            self.image = image
            self.gray_image = custImageToGray(image)
  
            self.low_pass_filter = np.array(
                [
                    [0, -1, 0], 
                    [-1, 5, -1], 
                    [0, -1, 0]
                ]
            )

            self.high_pass_filter = (1/6)*np.array(
                [
                    [0, 1, 0],
                    [1, 2, 1], 
                    [0, 1, 0]
                ]
            )

    def applyLowPassFilter(self, low_pass_filter:np.ndarray = None):
        cpy_img = self.gray_image.copy()
        if low_pass_filter is None:
            filter = self.low_pass_filter.copy()
        else:
            filter = low_pass_filter
        return convolution(cpy_img, filter)

        
    def applyHighPassFilter(self, high_pass_filter:np.ndarray = None):
        cpy_img = self.gray_image.copy()
        if high_pass_filter is None:
            filter = self.high_pass_filter.copy()
        else:
            filter = high_pass_filter
        return convolution(cpy_img, filter)
        
    def _medianFunction(self, neighborhood: np.ndarray) -> float:
        return np.median(neighborhood)
    def applyMedianFilter(self, kernel_size:int = 5):
        """
        Applies Median Filter 
        The median filter is considered a non-linear filter 
        and does not fit into the typical categories of low-pass or high-pass filters that are usually associated with linear filters.
        But it behave similary to low-pass filter.
        Linear filters is where we apply convoltuon operation, a weighted sum calculated by a sliding window.
        """
        filtered_image = custGenericFilter(self.gray_image, function=self._medianFunction, kernel_size=kernel_size, padding=True)
        return filtered_image
    

