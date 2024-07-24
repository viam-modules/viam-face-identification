import math
import os
import sys

import cv2 as cv
import numpy as np
import torchvision.transforms.functional as F
from numpy import random as rd
from torchvision.utils import save_image
from viam.logging import getLogger

from src.distance import distance_norm_l2
from src.models.yunet import YuNet

LOGGER = getLogger(__name__)
backend_target_pairs = [
    [cv.dnn.DNN_BACKEND_OPENCV, cv.dnn.DNN_TARGET_CPU],
    [cv.dnn.DNN_BACKEND_CUDA, cv.dnn.DNN_TARGET_CUDA],
    [cv.dnn.DNN_BACKEND_CUDA, cv.dnn.DNN_TARGET_CUDA_FP16],
    [cv.dnn.DNN_BACKEND_TIMVX, cv.dnn.DNN_TARGET_NPU],
    [cv.dnn.DNN_BACKEND_CANN, cv.dnn.DNN_TARGET_NPU],
]


backend_id = backend_target_pairs[0][0]
target_id = backend_target_pairs[0][1]


class Extractor:
    """
    Initializes the Extractor with the specified parameters.

    Args:
        target_size (tuple): The target size of the extracted face images.
        extraction_threshold (float): The confidence threshold for face extraction.
        extracting_model (str): The name of the face extraction model to be used.
        margin (float, optional): Margin added around detected faces for extraction. Defaults to 0.1. # pylint: disable=line-too-long
        grayscale (bool, optional): If True, convert images to grayscale before processing. Defaults to False.
        enforce_detection (bool, optional): If True, enforce face detection. Defaults to False.
        align (bool, optional): If True, align faces based on eye coordinates. Defaults to True.
        debug (bool, optional): If True, save intermediate images for debugging purposes. Defaults to False.

    Raises:
        ValueError: If an unknown extracting model is specified.
    """

    def __init__(
        self,
        target_size: (int, int),  # type: ignore
        extraction_threshold: float,
        extracting_model: str,
        margin: float = 0.1,
        grayscale=False,
        enforce_detection=False,
        align=True,
        debug: bool = False,
    ) -> None:
        self.model_name = extracting_model
        self.target_size = target_size
        self.extraction_threshold = extraction_threshold
        self.grayscale = grayscale
        self.enforce_detection = enforce_detection
        self.align = align
        self.margin = margin

        def resource_path(relative_path):
            """Get absolute path to resource, works for dev and for PyInstaller"""
            try:
                # PyInstaller creates a temp folder and stores path in _MEIPASS
                base_path = sys._MEIPASS  # pylint: disable=E1101,W0212
            except Exception:  # pylint: disable=broad-exception-caught
                base_path = os.path.abspath(os.path.join("src", "models"))

            return os.path.join(base_path, relative_path)

        if extracting_model == "yunet":
            path_to_yunet = os.path.join(
                "checkpoints", "face_detection_yunet_2023mar.onnx"
            )
            path_to_yunet = resource_path(path_to_yunet)
            self.detector = YuNet(
                modelPath=path_to_yunet,
                inputSize=[480, 480],
                confThreshold=0.6,
                nmsThreshold=0.2,
                topK=5000,
                backendId=backend_id,
                targetId=target_id,
            )
        else:
            raise ValueError(f"unknown extracting model: {extracting_model}")

        self.debug = debug

    def extract_faces(self, img):
        """
        Extracts and aligns faces from the input image.

        Args:
            img (numpy.ndarray): The input image from which faces are to be extracted.
            is_ir (bool): Flag indicating if the input image is infrared.

        Returns:
            list: A list of tuples containing the extracted face image, the region object, and a confidence score. # pylint: disable=line-too-long
        """
        face_objs = []
        img_h, img_w = img.shape[0], img.shape[1]

        self.detector.setInputSize([img_w, img_h])
        faces = self.detector.infer(img)
        for face in faces:
            (x, y, w, h, x_re, y_re, x_le, y_le) = list(map(int, face[:8]))

            h_margin_px = int(h * self.margin)
            w_margin_px = int(w * self.margin)

            detected_face = img[
                max(0, y - h_margin_px) : min(y + h + h_margin_px, img_h),
                max(0, x - w_margin_px) : min(x + w + w_margin_px, img_w),
            ].copy()
            if self.align:
                detected_face = self.align_face(
                    detected_face, (x_re, y_re), (x_le, y_le)
                )
            region_obj = {
                "x": x,
                "y": y,
                "w": w,
                "h": h,
            }
            region_obj = {
                "x": max(0, x),
                "y": max(0, y),
                "w": min(w, img_w - x),
                "h": min(h, img_h - y),
            }
            face_objs.append((detected_face, region_obj, 1))
        return face_objs

    def align_face(self, face, left_eye, right_eye):
        """
        Aligns the given face image based on the coordinates of the left and right eyes.

        Args:
            face (numpy.ndarray): The face image to be aligned.
            left_eye (tuple): The (x, y) coordinates of the left eye.
            right_eye (tuple): The (x, y) coordinates of the right eye.

        Returns:
            torch.Tensor: The aligned face image as a tensor.
        """

        left_eye_x, left_eye_y = left_eye
        right_eye_x, right_eye_y = right_eye

        # -----------------------
        # find rotation direction

        if left_eye_y > right_eye_y:
            point_3rd = (right_eye_x, left_eye_y)
            direction = -1  # rotate same direction to clock
        else:
            point_3rd = (left_eye_x, right_eye_y)
            direction = 1  # rotate inverse direction of clock

        # -----------------------
        # find length of triangle edges

        a = distance_norm_l2(np.array(left_eye), np.array(point_3rd))
        b = distance_norm_l2(np.array(right_eye), np.array(point_3rd))
        c = distance_norm_l2(np.array(right_eye), np.array(left_eye))

        # -----------------------

        # apply cosine rule

        if (
            b != 0 and c != 0
        ):  # this multiplication causes division by zero in cos_a calculation
            cos_a = (b * b + c * c - a * a) / (2 * b * c)
            angle = np.arccos(cos_a)  # angle in radian
            angle = (angle * 180) / math.pi  # radian to degree

            # -----------------------
            # rotate base image

            if direction == -1:
                angle = 90 - angle

            face_t = F.to_tensor(face.copy())

            if self.debug:
                _id = rd.randint(0, 10000)
                save_image(face_t, f"./before_rotation_{_id}.png")
            face_t = F.rotate(face_t, angle * direction)
            if self.debug:
                save_image(face_t, f"./after_rotation{_id}.png")
            face_t = F.resize(
                face_t, (self.target_size[0], self.target_size[1]), antialias=True
            )
        return face_t
