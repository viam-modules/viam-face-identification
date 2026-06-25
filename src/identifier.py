# pylint: disable=consider-using-dict-items

"""
This module provides an Identifier class
that has an extractor and an encoder to compute and compare
face embeddings.
"""

import math
import os
import pickle
import uuid
from pathlib import Path
from io import BytesIO

from pathvalidate import sanitize_filepath
import numpy as np
from PIL import Image
from viam.logging import getLogger

from src.distance import cosine_distance, distance_norm_l1, distance_norm_l2
from src.encoder import Encoder
from src.extractor import Extractor
from src.models import utils
from src.utils import check_ir, dist_to_conf_sigmoid

LOGGER = getLogger(__name__)


class Identifier:
    """
    A class to identify known faces by computing and comparing embeddings to known embeddings.

    Attributes:
        model_name (str): The name of the face recognition model.
        extractor (Extractor): The extractor object for extracting faces from images.
        encoder (Encoder): The encoder object for computing face embeddings.
        picture_directory (str): The directory containing images of known faces.
        known_embeddings (dict): A dictionary of known face embeddings.
        distance (callable): The distance metric function for comparing embeddings.
        identification_threshold (float): The threshold for identifying a face as known.
        sigmoid_steepness (float): The steepness of the sigmoid function for confidence calculation.
        debug (bool): If True, enables debug mode.

    Methods:
        compute_known_embeddings():
            Computes embeddings for known faces from the picture directory.
        get_detections(img):
            Computes face detections and identifications in the input image.
        compare_face_to_known_faces(face, is_ir, unknown_label="unknown"):
            Encodes the face, calculates its distances with known faces, and returns the best match and the confidence. # pylint: disable=line-too-long
    """

    def __init__(
        self,
        detector_backend: str,
        extraction_threshold: float,
        grayscale: bool,
        enforce_detection: bool,
        align: bool,
        model_name: str,
        normalization: str,
        picture_directory: str,
        distance_metric_name: str,
        identification_threshold: float,
        sigmoid_steepness: float,
        debug: bool = False,
    ):
        self.model_name = model_name

        target_size = (112, 112)  # same for 'sface' and 'facenet'

        self.extractor = Extractor(
            target_size=target_size,
            extracting_model=detector_backend,
            extraction_threshold=extraction_threshold,
            grayscale=grayscale,
            enforce_detection=enforce_detection,
            align=align,
            debug=debug,
        )

        self.encoder = Encoder(
            model_name=model_name, align=align, normalization=normalization, debug=debug
        )

        self.picture_directory = picture_directory
        self.model_name = model_name
        self.known_embeddings = {}

        # Precomputed gallery embeddings are persisted here so that startup just
        # reads them instead of re-encoding every known face on every reconfigure.
        self.embeddings_path = os.path.join(picture_directory, "embeddings.pkl")
        # Anything that changes the embeddings themselves. If this differs from
        # what's stored in the cache file, the cache is stale and is recomputed,
        # so changing the model in config can't silently serve old embeddings.
        self._embeddings_signature = "|".join(
            str(p)
            for p in (
                model_name,
                normalization,
                align,
                detector_backend,
                extraction_threshold,
                grayscale,
            )
        )

        if distance_metric_name == "cosine":
            self.distance = cosine_distance
        if distance_metric_name == "manhattan":
            self.distance = distance_norm_l1
        elif distance_metric_name == "euclidean":
            self.distance = distance_norm_l2

        if identification_threshold is None:
            self.identification_threshold = utils.find_threshold(
                distance_metric=distance_metric_name
            )  # ideally, this would also depend on the FR model
        else:
            self.identification_threshold = identification_threshold

        self.sigmoid_steepness = sigmoid_steepness
        self.debug = True

    def write_embedding(self, image:BytesIO, ext:str, embedding_name:str):
        """
        Writes a new embedding file into the picture directory.
        """
        file_path = f"{self.picture_directory}/{embedding_name}"
        file_path = sanitize_filepath(file_path)
        if not os.path.exists(file_path):
            Path(file_path).mkdir(exist_ok=True)
        file_name = f"{uuid.uuid4()}.{ext}"
        sanitized = sanitize_filepath(f"{file_path}/{file_name}")
        with open(sanitized, "wb") as f:
            f.write(image.getvalue())
            LOGGER.info("Wrote %s as embedding", sanitized)

    def load_known_embeddings(self) -> bool:
        """
        Reads precomputed embeddings from disk.

        Returns True if a usable cache was loaded; False if it's missing,
        unreadable, or was built with a different model config (stale signature),
        signalling the caller to recompute.
        """
        if not os.path.exists(self.embeddings_path):
            LOGGER.info("No precomputed embeddings at %s", self.embeddings_path)
            return False
        try:
            with open(self.embeddings_path, "rb") as f:
                payload = pickle.load(f)
        except Exception:  # pylint: disable=broad-exception-caught
            LOGGER.exception(
                "Failed to read precomputed embeddings at %s", self.embeddings_path
            )
            return False
        if payload.get("signature") != self._embeddings_signature:
            LOGGER.warning(
                "Precomputed embeddings at %s were built with a different model "
                "config; recomputing.",
                self.embeddings_path,
            )
            return False
        self.known_embeddings = payload.get("embeddings", {})
        LOGGER.info(
            "Loaded precomputed embeddings from %s: %s labelled group(s).",
            self.embeddings_path,
            len(self.known_embeddings),
        )
        return True

    def save_known_embeddings(self):
        """
        Persists the current embeddings (tagged with the model signature) so a
        later startup can read them instead of recomputing.
        """
        payload = {
            "signature": self._embeddings_signature,
            "embeddings": self.known_embeddings,
        }
        try:
            with open(self.embeddings_path, "wb") as f:
                pickle.dump(payload, f)
            LOGGER.info(
                "Saved precomputed embeddings to %s (%s group(s)).",
                self.embeddings_path,
                len(self.known_embeddings),
            )
        except Exception:  # pylint: disable=broad-exception-caught
            LOGGER.exception("Failed to save embeddings to %s", self.embeddings_path)

    def compute_known_embeddings(self):
        """
        Computes embeddings for known faces from the picture directory and
        persists them to disk via save_known_embeddings().
        """
        known = {}
        all_entries = os.listdir(self.picture_directory)
        directories = [
            entry
            for entry in all_entries
            if os.path.isdir(os.path.join(self.picture_directory, entry))
        ]
        for directory in directories:
            label_path = os.path.join(self.picture_directory, directory)
            embeddings = []
            for file in os.listdir(label_path):
                if (
                    (".jpg" in file.lower())
                    or (".jpeg" in file.lower())
                    or (".png" in file.lower())
                ):
                    im = Image.open(label_path + "/" + file).convert(
                        "RGB"
                    )  # convert in RGB because png are RGBA
                    img = np.array(im)
                    r = img[:, :, 0]
                    g = img[:, :, 1]
                    is_ir = (r == g).all()
                    faces = self.extractor.extract_faces(img)
                    for face, _, _ in faces:
                        embed = self.encoder.encode(face, is_ir)
                        embeddings.append(embed)
                else:
                    LOGGER.warning(
                        "Ignoring unsupported file type: %s. Only .jpg, .jpeg, and .png files are supported.",  # pylint: disable=line-too-long
                        file,
                    )

            known[directory] = embeddings

        self.known_embeddings = known
        self.save_known_embeddings()

    def get_detections(self, img):
        """
        Computes face detections and identifications in the input image.

        Faces whose minimum distance to known embeddings
        is greater than the threshold are labelled as 'unknown'.
        Args:
            img (numpy.ndarray): The input image in RGB format.

        Returns:
            list: A list of dictionaries containing detection results.
        """
        detections = []
        is_ir = check_ir(img)
        faces = self.extractor.extract_faces(img)
        for face, face_region, _ in faces:
            match, conf = self.compare_face_to_known_faces(face, is_ir)
            detection = {
                "confidence": conf,
                "class_name": match,
                "x_min": face_region["x"],
                "y_min": face_region["y"],
                "x_max": face_region["x"] + face_region["w"],
                "y_max": face_region["y"] + face_region["h"],
            }

            if img.shape[0] != 0 and img.shape[1] != 0:
                detection.update(
                    {
                        "x_min_normalized": face_region["x"] / img.shape[1],
                        "y_min_normalized": face_region["y"] / img.shape[0],
                        "x_max_normalized": (face_region["x"] + face_region["w"]) / img.shape[1],
                        "y_max_normalized": (face_region["y"] + face_region["h"]) / img.shape[0],
                    }
                )

            detections.append(detection)

        return detections

    def compare_face_to_known_faces(self, face, is_ir, unknown_label: str = "unknown"):
        """
        Encodes the face, calculates its distances with known faces and
        returns the best match and the confidence.

        Args:
            face (np.array): extracted face of size self.target_size

        Returns:
            label, confidence (str, float):
        """
        source_embed = self.encoder.encode(face, is_ir)
        match = None
        min_dist = math.inf
        for label in self.known_embeddings:
            for target_embed in self.known_embeddings[label]:
                dist = self.distance(source_embed, target_embed)
                if dist < min_dist:
                    match, min_dist = label, dist

        if min_dist < self.identification_threshold:
            return match, dist_to_conf_sigmoid(min_dist, self.sigmoid_steepness)
        return unknown_label, 1 - dist_to_conf_sigmoid(min_dist, self.sigmoid_steepness)
