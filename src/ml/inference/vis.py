import numpy as np

from PIL.JpegImagePlugin import JpegImageFile

def visualise_image(image:JpegImageFile) -> None:
    """
    Visualises an image using matplotlib.
    
    Args:
        image (JpegImageFile): The image to be visualised.
    """
    import matplotlib.pyplot as plt

    # Convert the image tensor to a numpy array

    image_np = np.array(image)

    # Display the image
    plt.imshow(image_np)
    plt.axis('off')
    plt.show()