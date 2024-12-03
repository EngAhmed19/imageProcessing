import numpy as np
import matplotlib.image as mplt
from AdvancedEdgeDetection import AdvancedEdgeDetection
from BasicEdgeDetection import BasicEdgeDetection

image = mplt.imread('images/bad_light_3.jpg') # Simulated grayscale image (replace with actual image)
# print(f"image shape when readed {image.shape}")
# edge_detector = AdvancedEdgeDetection(image)
basic_edge_detector = BasicEdgeDetection(image)
basic_edge_detector_contrast = BasicEdgeDetection(image, contrast_based_smoothing=True)


# Apply the homogeneity operator with kernel size 3 and threshold 50
# edges = edge_detector.homogeneityOperator(area_size=3)
# print(f"image shape when edged {edges.shape}")

# Apply the difference operator 
# edges = edge_detector.differenceOperator()
# print(f"image shape when edged {edges.shape}")

# Basic Edge 
edges = basic_edge_detector.sobelEdgeDetection()
edges_contrast = basic_edge_detector_contrast.sobelEdgeDetection()

print(f"image shape when edged {edges.shape}")
print(f"image shape when edged {edges_contrast.shape}")

import matplotlib.pyplot as plt
# Create a subplot with 1 row and 2 columns
fig, axs = plt.subplots(1, 2, figsize=(12, 6))

# Plot the edges image
axs[0].imshow(edges, cmap='gray')
axs[0].set_title("Edge Detection (Normal)")
axs[0].axis('off')  # Hide axes for cleaner presentation

# Plot the edges_contrast image
axs[1].imshow(edges_contrast, cmap='gray')
axs[1].set_title("Edge Detection (With Contrast Smoothing)")
axs[1].axis('off')  # Hide axes for cleaner presentation

# Show the plot
plt.tight_layout()  # Adjust the layout to avoid overlap
plt.show()