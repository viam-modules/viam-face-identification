from deepface.commons import distance as dst
from typing import Union
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from viam.media.video import RawImage
from PIL import Image
from viam.logging import getLogger
import numpy as np
LOGGER = getLogger(__name__)

def draw_detections(img, detections, output_path):
    plt.imshow(img)
    for detection in detections:
        text = detection['class_name']+ ": " + str(round(detection["confidence"],3))
        x_min, y_min = detection["x_min"], detection["y_min"]
        x_max, y_max = detection["x_max"], detection["y_max"]

        # Create a rectangle patch
        rect = patches.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min,
                                linewidth=2, edgecolor='r', facecolor='none')
        
        # Add the rectangle to the current axes
        plt.gca().add_patch(rect)

        # Add text near the rectangle
        plt.text(x_min, y_min - 10, text, color='r', fontsize=12, bbox=dict(facecolor='white', alpha=0.7))

        # Save the modified image
    plt.savefig(output_path)
    
    
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

