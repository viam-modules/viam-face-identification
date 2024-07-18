# pylint: disable=possibly-used-before-assignment
import os
import sys

import torch
import torchvision.transforms.functional as F
from facenet_pytorch import InceptionResnetV1
from onnx2torch import convert
from torch import Tensor

from .pdt import PDT


def find_threshold(distance_metric):
    """
    Returns the threshold value based on the specified distance metric.

    Args:
        distance_metric (str): The distance metric to use. Can be 'euclidean', 'euclidean_l2', or 'cosine'. # pylint: disable=line-too-long

    Returns:
        float: The threshold value corresponding to the distance metric.

    Raises:
        ValueError: If the distance metric is not one of 'euclidean', 'euclidean_l2', or 'cosine'.
    """
    if distance_metric == "euclidean":
        return 1.1
    if distance_metric == "euclidean_l2":
        return 1.1
    if distance_metric == "cosine":
        return 0.4
    raise ValueError(
        f"Distance metric must be one of: 'euclidean', 'euclidean_l2', 'cosine' but got {distance_metric} instead."  # pylint: disable=line-too-long
    )


def resource_path(relative_path):
    """
    Get the absolute path to a resource file, considering different environments.

    Args:
        relative_path (str): The relative path to the resource file.

    Returns:
        str: The absolute path to the resource file.
    """
    try:
        # PyInstaller creates a temp folder and stores path in _MEIPASS
        base_path = sys._MEIPASS  # pylint: disable=duplicate-code,protected-access

    except Exception:  # pylint: disable=broad-exception-caught
        base_path = os.path.abspath(os.path.join("src", "models"))

    return os.path.join(base_path, relative_path)


def get_all(model_name):
    """
    Retrieves necessary components (transform, translator, face_recognizer) based on the given model name. # pylint: disable=line-too-long

    Args:
        model_name (str): The name of the model. Can be 'sface' or 'facenet'.

    Returns:
        tuple: A tuple containing:
            - transform (function): A function to transform images for the specified model.
            - translator (PDT): An instance of PDT (Pose Discrimination Transformer) for the model.
            - face_recognizer (torch.nn.Module): The face recognition model pretrained on the specified architecture. # pylint: disable=line-too-long

    Notes:
        This function assumes that the necessary checkpoints and configurations are stored in specific directories. # pylint: disable=line-too-long
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if model_name == "sface":
        mean = [0.0, 0.0, 0.0]
        std = [0.5, 0.5, 0.5]

        relative_path = os.path.join(
            "checkpoints", "face_recognition_sface_2021dec.onnx"
        )

        path_to_encoder_checkpoint = resource_path(relative_path)

        face_recognizer = convert(path_to_encoder_checkpoint).eval()

        path_to_translator_checkpoint = os.path.join("checkpoints", "pdt_sface.pt")

    if model_name == "facenet":
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]

        face_recognizer = InceptionResnetV1(pretrained="vggface2").eval()
        path_to_translator_checkpoint = os.path.join("checkpoints", "pdt_facenet.pt")

    face_recognizer.requires_grad_(False)
    face_recognizer.to(device)

    def transform(img):
        if not isinstance(img, Tensor):
            img = F.to_tensor(img)
        img = F.normalize(img, mean=mean, std=std)
        img = F.resize(img, (112, 112), antialias=True)
        return img.unsqueeze(0)

    relative_path = resource_path(path_to_translator_checkpoint)
    checkpoint = torch.load(relative_path, map_location=torch.device(device))
    translator = PDT(pool_features=6, use_se=False, use_bias=False, use_cbam=True)
    translator.load_state_dict(checkpoint["model_state_dict"])
    translator.eval()
    translator.requires_grad_(False)
    translator.to(device)

    return transform, translator, face_recognizer
