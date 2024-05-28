from typing import Union
from viam.media.video import ViamImage
from PIL import Image
from viam.logging import getLogger
from viam.media.video import CameraMimeType
import numpy as np

LOGGER = getLogger(__name__)
SUPPORTED_IMAGE_TYPE = [
    CameraMimeType.JPEG,
    CameraMimeType.PNG,
    CameraMimeType.VIAM_RGBA,
]


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
                f"Unsupported image type: {image.mime_type}. Supported types are {SUPPORTED_IMAGE_TYPE}."
            )
            raise ValueError(f"Unsupported image type: {image.mime_type}.")
        im = Image.open(image.data).convert(
            "RGB"
        )  # convert in RGB png openened in RGBA
        return np.array(im)

    res = image.convert("RGB")
    rgb = np.array(res)
    return rgb
    # bgr = rgb[...,::-1]
    # return bgr


def dist_to_conf_sigmoid(dist, steep=10):
    return 1 / (1 + np.exp((steep * (dist - 0.5))))


def check_ir(img, acc=10):
    h, w = img.shape[0], img.shape[1]
    r = img[h // 2 : h // 2 + h // acc, w // 2 : w // 2 + w // acc, 0]
    g = img[h // 2 : h // 2 + h // acc, w // 2 : w // 2 + w // acc, 1]
    return (r == g).all()
