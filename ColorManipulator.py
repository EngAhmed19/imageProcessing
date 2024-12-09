import numpy as np
import matplotlib.pyplot as plt

class ColorManipulator:
    def __init__(self, image: np.ndarray):
        """
        Initialize the ColorManipulator with an image.
        
        :parameter:
            :param image (np.ndarray): A 3D NumPy array representing the image.
        """
        if not isinstance(image, np.ndarray):
            raise ValueError("Input must be a NumPy array.")
        
        # If the image is grayscale (2D), convert it to RGB by duplicating the channel
        if len(image.shape) == 2:  # Grayscale image (2D)
            image = np.stack([image] * 3, axis=-1)
        
        # If the image has 4 channels (RGBA), remove the alpha channel
        elif len(image.shape) == 3 and image.shape[2] == 4:
            image = image[:, :, :3]  # Keep only the first 3 channels (RGB)
        
        self.image = image




    def apply_color_filter(self, red_factor: float, green_factor: float, blue_factor: float):
        """
        Apply a color filter to the image.
        
        :parameter:
            :param red_factor (float): Multiplicative factor for the red channel.
            :param green_factor (float): Multiplicative factor for the green channel.
            :param blue_factor (float): Multiplicative factor for the blue channel.
        >>> color = ColorManipulator(image=image)
        >>> out_image = color.apply_color_filter(1, 3, 1)
        """
        cpy_img = self.image.copy()
        for i in range(cpy_img.shape[0]):
            for j in range(cpy_img.shape[1]):
                r, g, b = cpy_img[i, j]
                cpy_img[i, j] = [
                    np.clip(r * red_factor, 0, 255),
                    np.clip(g * green_factor, 0, 255),
                    np.clip(b * blue_factor, 0, 255),
                ]
        return cpy_img




# # Example image as a NumPy array (replace with an actual image read operation)
# image = plt.imread("woody.png")

# # Initialize the ImageManipulator class
# manipulator = ColorManipulator(image)

# # Apply a color filter
# manipulator.apply_color_filter(1, 3, 1)

# # Retrieve the manipulated image
# output_image = manipulator.get_image()

# # Display the final image
# plt.imshow(output_image)
# plt.axis('off')  # Hide axis labels
# plt.show()
