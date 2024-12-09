class ImageFlipper:
    def __init__(self, image):
        self.image = image

    def flip(self, flip_type):

        if flip_type == "horizontal":
            flipped_image = self.image[:, ::-1]

        elif flip_type == "vertical":
            flipped_image = self.image[::-1, :]
        
        elif flip_type == "horizontal-vertical":
            # 180-degree rotation
            flipped_image = self.image[::-1, ::-1]

        else:
            raise ValueError("flip_type must be 'horizontal', 'vertical' or 'horizontal-vertical' ")

        return flipped_image
