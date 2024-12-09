import numpy as np

class Mask:
    def __init__(self):
        pass


# def apply_circular_mask(image_path, center=None, radius=None, opacity=0.5):
#     """
#     Applies a circular mask to an image and returns the result with reduced opacity.

#     Args:
#         image_path (str): Path to the input image.
#         center (tuple): (x, y) coordinates for the mask's center. Default is the center of the image.
#         radius (int): Radius of the circular mask. Default is half the smaller image dimension.
#         opacity (float): Opacity of the mask. Value should be between 0 (transparent) and 1 (opaque).
#     """
#     # Read the image
#     image = cv2.imread(image_path)
#     if image is None:
#         raise ValueError("Image not found or invalid image path.")

#     # Get image dimensions
#     height, width = image.shape[:2]

#     # Set default center and radius if not provided
#     if center is None:
#         center = (width // 2, height // 2)
#     if radius is None:
#         radius = min(width, height) // 2

#     # Create a mask with the same dimensions as the image
#     mask = np.zeros((height, width), dtype=np.uint8)

#     # Draw a white filled circle on the mask
#     cv2.circle(mask, center, radius, 255, -1)

#     # Apply the mask to the image
#     masked_image = cv2.bitwise_and(image, image, mask=mask)

#     # Blend the masked image with the original image to reduce opacity
#     result_image = cv2.addWeighted(image, 1 - opacity, masked_image, opacity, 0)

#     return result_image
# import matplotlib.pyplot as plt
# from PIL import Image
# # make a two subplots
# fig, ax = plt.subplots(1, 3, figsize=(12, 6))
# ax[0].imshow(Image.open('images/woody.png'))
# ax[1].imshow(apply_circular_mask('images/woody.png', radius=230, opacity=0.6))
# ax[2].imshow(apply_circular_mask('images/woody.png',center=(0,30), radius=100, opacity=0.7))
# plt.show()

