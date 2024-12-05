import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mplt

from helperFunctions import custDynamicThreshold, ThresholdStrategy

from HistogramBasedSegmentation import HistogramBasedSegmentation, NoiseReductionStrategy

# Load or generate a test image (for example, a random image or one from a file)
image = mplt.imread("images/logo.png")

# Initialize the HistogramBasedSegmentation object
segmentation = HistogramBasedSegmentation(
    image=image,
    noise_reduction_strategy=NoiseReductionStrategy.GuassianSmoothing, 
    sigma=2, kernel_size=11
)

# Preprocess the image (apply noise reduction and contrast enhancement)
preprocessed_image = segmentation.preprocess()

# Apply manual histogram segmentation with chosen thresholds
lower_threshold = custDynamicThreshold(preprocessed_image, strategy=ThresholdStrategy.MEAN_MINUS_STD)
upper_threshold = custDynamicThreshold(preprocessed_image, strategy=ThresholdStrategy.MEDIAN_PLUS_STD)
segmented_image = segmentation.manual_histogram_segmentation(lower_threshold, upper_threshold)

# Visualize the original, preprocessed, and segmented images using matplotlib
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Original Image
axes[0].imshow(image, cmap='gray')
axes[0].set_title("Original Image")
axes[0].axis('off')

# Preprocessed Image
axes[1].imshow(preprocessed_image, cmap='gray')
axes[1].set_title("Preprocessed Image")
axes[1].axis('off')

# Segmented Image
axes[2].imshow(segmented_image, cmap='gray')
axes[2].set_title("Segmented Image")
axes[2].axis('off')

plt.tight_layout()
plt.show()
