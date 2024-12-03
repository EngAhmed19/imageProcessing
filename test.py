import numpy as np
import matplotlib.image as mplt
from AdvancedEdgeDetection import AdvancedEdgeDetection
image = mplt.imread('images/flower.png') # Simulated grayscale image (replace with actual image)
print(f"image shape when readed {image.shape}")
edge_detector = AdvancedEdgeDetection(image)


# Apply the homogeneity operator with kernel size 3 and threshold 50
edges = edge_detector.homogeneityOperator(area_size=3)
print(f"image shape when edged {edges.shape}")

# Show the result (for example, using matplotlib if needed)
import matplotlib.pyplot as plt
plt.imshow(edges, cmap='gray')
plt.show()
