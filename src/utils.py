from io import BytesIO
from typing import Union

import numpy as np
from PIL import Image
from viam.logging import getLogger
from viam.media.video import CameraMimeType, ViamImage

LOGGER = getLogger(__name__)
SUPPORTED_IMAGE_TYPE = [
    CameraMimeType.JPEG,
    CameraMimeType.PNG,
    CameraMimeType.VIAM_RGBA,
]
LIBRARY_SUPPORTED_FORMATS = ["JPEG", "PNG", "VIAM_RGBA"]


def decode_image(image: Union[Image.Image, ViamImage]) -> np.ndarray:
    """decode image to BGR numpy array

    Args:
        raw_image (Union[Image.Image, RawImage])

    Returns:
        np.ndarray: BGR numpy array
    """
    if isinstance(image, ViamImage):
        if image.mime_type not in SUPPORTED_IMAGE_TYPE:
            LOGGER.error(
                f"Unsupported image type: {image.mime_type}. Supported types are {SUPPORTED_IMAGE_TYPE}."  # pylint: disable=line-too-long
            )
            raise ValueError(f"Unsupported image type: {image.mime_type}.")

        im = Image.open(BytesIO(image.data), formats=LIBRARY_SUPPORTED_FORMATS).convert(
            "RGB"
        )  # convert in RGB png openened in RGBA
        return np.array(im)

    res = image.convert("RGB")
    rgb = np.array(res)
    return rgb
    # bgr = rgb[...,::-1]
    # return bgr


def dist_to_conf_sigmoid(dist, steep=10):
    """
    Converts a distance metric to a confidence score using a sigmoid function.

    Args:
        dist (float): The distance metric.
        steep (int, optional): The steepness of the sigmoid curve. Default is 10.

    Returns:
        float: The confidence score derived from the distance metric.
    """
    return 1 / (1 + np.exp((steep * (dist - 0.5))))


def check_ir(img, acc=10):
    """
    Checks if an image is infrared by verifying if all pixel values are the same across the red and green channels # pylint: disable=line-too-long
    in a central region of the image.

    Args:
        img (numpy.ndarray): The input image.
        acc (int, optional): The accuracy parameter determining the size of the central region to check. Default is 10. # pylint: disable=line-too-long

    Returns:
        bool: True if the image is identified as infrared, False otherwise.
    """
    h, w = img.shape[0], img.shape[1]
    r = img[h // 2 : h // 2 + h // acc, w // 2 : w // 2 + w // acc, 0]
    g = img[h // 2 : h // 2 + h // acc, w // 2 : w // 2 + w // acc, 1]
    return (r == g).all()
