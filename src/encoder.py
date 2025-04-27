"""
This module provides an Encoder class
to compute a representation/embedding of a face.
"""

import numpy as np
import numpy.random as rd
from torchvision.utils import save_image
import torch

from src.models import utils


class Encoder:
    """
    A class used to compute embeddings of a face image using a specified model.

    Attributes:
        model_name (str): The name of the face recognition model.
        transform (callable): The transformation function to preprocess the face image.
        translator (callable): The function to translate infrared images to three channels.
        face_recognizer (callable): The face recognition model to compute embeddings.
        align (tuple): A tuple containing alignment settings.
        normalization (bool): If True, normalize the embeddings.
        debug (bool): If True, save intermediate images for debugging purposes.

    Methods:
        encode(face, is_ir):
            Computes the embedding of the given face image.
    """

    def __init__(self, model_name, align, normalization, debug) -> None:
        self.model_name = model_name
        self.transform, self.translator, self.face_recognizer = utils.get_all(
            self.model_name
        )
        self.align = (align,)
        self.normalization = normalization
        self.debug = debug
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def encode(self, face, is_ir):
        """
        Computes the embedding of the given face image.

        Args:
            face (numpy.ndarray): The input face image.
            is_ir (bool): Flag indicating if the input image is infrared.

        Returns:
            numpy.ndarray: The computed embedding of the face image.
        """
        try:
            img = self.transform(face)
        except Exception as e:
            raise Exception(f"encountered error in encoder transform: {str(e)}") from e
        if self.debug:
            _id = rd.randint(0, 10000)
            save_image(img, f"./transformed_{_id}.png")
        if is_ir:
            try:
                three_channeled_image = self.translator(img)
            except Exception as e:
                raise Exception(f"encountered error in encoder translator: {str(e)}") from e
            if self.debug:
                save_image(three_channeled_image, f"./three_channel_{_id}.png")
        else:
            three_channeled_image = img
            if self.debug:
                save_image(
                    three_channeled_image, f"./three_channel_no_translate{_id}.png"
                )
        try:
            if isinstance(three_channeled_image, np.ndarray):
                face_tensor = torch.from_numpy(three_channeled_image).to(self.device)
            elif isinstance(three_channeled_image, torch.Tensor):
                face_tensor = three_channeled_image.to(self.device)
            else:
                raise TypeError("Input to face recognition encoder must be numpy array or torch tensor")
            embed = self.face_recognizer(face_tensor)
        except Exception as e:
            raise Exception(f"encountered error in encoder face recognizer: {str(e)}") from e
        return embed[0]
