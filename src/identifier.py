from src.extractor import Extractor
from src.encoder import Encoder
from src.utils import dist_to_conf_sigmoid, check_ir
import os
from PIL import Image
import numpy as np
import math
from viam.logging import getLogger
from src.models import utils
from src.distance import euclidean_distance, cosine_distance, euclidean_l2_distance

LOGGER = getLogger(__name__)


class Identifier:
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
        self.known_embeddings = dict()

        if distance_metric_name == "cosine":
            self.distance = cosine_distance
        if distance_metric_name == "euclidean":
            self.distance = euclidean_distance
        elif distance_metric_name == "euclidean_l2":
            self.distance = euclidean_l2_distance

        if identification_threshold is None:
            self.identification_threshold = utils.find_threshold(
                distance_metric=distance_metric_name
            )  # ideally, this would also depend on the FR model
        else:
            self.identification_threshold = identification_threshold

        self.sigmoid_steepness = sigmoid_steepness
        self.debug = True

    def compute_known_embeddings(self):
        all_entries = os.listdir(self.picture_directory)
        directories = [
            entry
            for entry in all_entries
            if os.path.isdir(os.path.join(self.picture_directory, entry))
        ]
        for dir in directories:
            label_path = os.path.join(self.picture_directory, dir)
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
                    faces = self.extractor.extract_faces(img, is_ir)
                    for face, _, _ in faces:
                        embed = self.encoder.encode(face, is_ir)
                        embeddings.append(embed)
                else:
                    LOGGER.warn(
                        f"Ignoring unsupported file type: {file}. Only .jpg, .jpeg, and .png files are supported."
                    )
            self.known_embeddings[dir] = embeddings

    def get_detections(self, img):
        """
        compute face detections and identifications in the input
        image. Faces whose minimum distance at best embedding are
        greater than the threshold are labelled as 'unknown'.
        Args:
            img (np.array): RGB format

        Returns:
            [Detections]: _description_
        """
        detections = []
        is_ir = check_ir(img)
        faces = self.extractor.extract_faces(img, is_ir)
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
