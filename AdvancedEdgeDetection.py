import numpy as np
from helperFunctions import custImageToGray, convolution, custGenericFilter

class AdvancedEdgeDetection:
    def __init__(self, image:np.ndarray):
        self.image = image
        self.gray_image = custImageToGray(image)
        # self.gray_image = convertImageToGray(image)

    def _homogeneityFunction(self, neighborhood: np.ndarray) -> float:
        """
        Function to calculate homogeneity for a given neighborhood.
        The homogeneity operator computes the maximum absolute difference
        between the center pixel and the surrounding pixels.
        """
        center_pixel = neighborhood[len(neighborhood) // 2, len(neighborhood) // 2]
        diff = np.abs(neighborhood - center_pixel)
        return np.max(diff)
       

    def homogeneityOperator(self, area_size: int = 3, threshold: int = None)-> np.ndarray:
        """
        Applies the Homogeneity Operator for edge detection.
        
        Args:
            area_size (int): The size of the neighborhood (odd number).
            threshold (int): The threshold for binary edge map, a number between 0 and 1.
        
        Returns:
            np.ndarray: The thresholded edge map.

        """
        filtered_image = custGenericFilter(self.gray_image, function=self._homogeneityFunction, kernel_size=area_size, padding=True)
        if threshold:
            threshold = threshold
        else:
            threshold = np.mean(filtered_image) + np.std(filtered_image)
        
        print(f"threshold {threshold}")
        # Convert to 0 or 255 (binary image)
        edge_map = (filtered_image > threshold).astype(np.uint8) * 255
        # print(f"Filterd image in the class  {filtered_image.shape}")
        # print(f"Edge map in the class  {edge_map.shape}")
        return edge_map
    def differenceOperator(self):
        pass
    def differenceOfGaussians(self):
        pass
    def contrastBaseEdgeDetecto(self):
        pass
    def varianceEdgeDetector(self):
        pass
    def rangeEdgeDetector(self):
        pass

