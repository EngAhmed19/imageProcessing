import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mplt

from helperFunctions import custDynamicThreshold, ThresholdStrategy

from HistogramBasedSegmentation import HistogramBasedSegmentation, NoiseReductionStrategy
from HistogramEqualization import Histogram
from helperFunctions import custImageToGray

# # Load or generate a test image (for example, a random image or one from a file)
# image = mplt.imread("images/logo.png")
# image = mplt.imread("images/nature.jpg")
image = mplt.imread("images/flower.png")

# image = mplt.imread("images/bad_light_1.jpg")
# print(f"image shape {image.shape}, min {image.min()}, max {image.max()}")
# print(f"image2 shape {image2.shape}, min {image2.min()}, max {image2.max()}")

# Initialize the HistogramBasedSegmentation object
segmentation = HistogramBasedSegmentation(
    image=image,
    noise_reduction_strategy=NoiseReductionStrategy.GuassianSmoothing, 
    sigma=2, kernel_size=11
)

# Preprocess the image (apply noise reduction and contrast enhancement)
preprocessed_image = segmentation.preprocess(active_contrast_enhancement=False, active_noiseReduction=False)

# contrast increase the peaks 
# Apply manual histogram segmentation with chosen thresholds
lower_threshold = custDynamicThreshold(preprocessed_image, strategy=ThresholdStrategy.MEAN_MINUS_STD)
upper_threshold = custDynamicThreshold(preprocessed_image, strategy=ThresholdStrategy.MEDIAN_PLUS_STD)
# segmented_image = segmentation.manual_histogram_segmentation(lower_threshold, upper_threshold)
# segmented_image = segmentation.peak_histogram_segmentation(peaks_min_distance=30)
# segmented_image = segmentation.peak_histogram_segmentation()

# segmented_image = segmentation.peak_histogram_segmentation(peaks_min_distance=200)
segmented_image = segmentation.valley_histogram_segmentation(peaks_min_distance=50)

# segmented_image = segmentation.adaptive_histogram_segmentation(peaks_min_distance=50)

# Visualize the original, preprocessed, and segmented images using matplotlib
fig, axes = plt.subplots(1, 4, figsize=(20, 5))

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

# histogram
axes[3].hist(preprocessed_image.ravel(), bins=256, range=(0, 255), alpha=0.7)
axes[3].set_title("Histogram of the preprocessed_image")
axes[3].axis('off')


plt.tight_layout()
plt.show()
