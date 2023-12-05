from deepface.commons import distance as dst
from typing import Union
from viam.media.video import RawImage
from PIL import Image
from viam.logging import getLogger
import numpy as np
LOGGER = getLogger(__name__)
    
def euclidian_l2(source_embed,target_embed ):
    return dst.findEuclideanDistance(
                        dst.l2_normalize(source_embed),
                        dst.l2_normalize(target_embed),
                    )
    
    
def decode_image(image: Union[Image.Image, RawImage])-> np.ndarray:
    """decode image to BGR numpy array

    Args:
        raw_image (Union[Image.Image, RawImage])

    Returns:
        np.ndarray: BGR numpy array
    """
    if type(image) == RawImage:
        im = Image.open(image.data).convert('RGB') #convert in RGB png openened in RGBA
        return np.array(im)[...,::-1]

    res = image.convert('RGB')
    rgb = np.array(res)
    bgr = rgb[...,::-1]
    return bgr

