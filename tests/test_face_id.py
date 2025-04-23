from src.face_detection_module import FaceIdentificationModule, ENCODERS, EXTRACTORS
from fake_camera import FakeCamera

from viam.services.vision import Detection, Vision
from typing import List, Dict
from google.protobuf.struct_pb2 import Struct
from viam.proto.app.robot import ServiceConfig
import os
import pytest
import base64
import shutil

CAMERA_NAME = "fake-camera"

EXCEPTION_REQUIRED_CAMERA = Exception(
    "A camera name is required for face_identification vision service module."
)
PASSING_PROPERTIES = Vision.Properties(
    classifications_supported=False,
    detections_supported=True,
    object_point_clouds_supported=False,
)

SINGLE_PERSON_PICTURE = 1
PERSONS = ["zidane", "chirac"]
PERSON_TO_ADD = "cotillard"
MIN_CONFIDENCE_PASSING = 0.8

WORKING_CONFIG_DICT = {
    "picture_directory": os.path.join("tests", "img"),
    "camera_name": CAMERA_NAME,
    "extractor_model": "yunet",
    "face_embedding_model": "facenet",
}


def get_config(config_dict: Dict):
    """returns a config populated with picture_directory and camera_name
    attributes.

    Returns:
        ServiceConfig: _description_
    """
    struct = Struct()
    struct.update(dictionary=config_dict)
    config = ServiceConfig(attributes=struct)
    return config


def get_vision_service(config_dict: Dict, people: List):
    service = FaceIdentificationModule("test")
    cam = FakeCamera(CAMERA_NAME, persons=people)
    camera_name = cam.get_resource_name(CAMERA_NAME)
    cfg = get_config(config_dict)
    service.validate_config(cfg)
    service.reconfigure(cfg, dependencies={camera_name: cam})
    return service


class TestFaceReId:
    def test_config(self):
        with pytest.raises(ValueError):
            service = get_vision_service(config_dict={}, people=PERSONS)

    def test_wrong_encoder_config(self):
        cfg = WORKING_CONFIG_DICT.copy()
        cfg["extractor_model"] = "not-a-real-model-name"
        with pytest.raises(ValueError):
            _ = get_vision_service(cfg, people=PERSONS)

    @pytest.mark.asyncio
    async def test_get_properties(self):
        service = FaceIdentificationModule("test")
        p = await service.get_properties()
        assert p == PASSING_PROPERTIES

    @pytest.mark.asyncio
    async def test_get_detections_from_camera(self):
        service = get_vision_service(WORKING_CONFIG_DICT, people=PERSONS)
        for person in PERSONS:
            get_detections_from_camera_result = (
                await service.get_detections_from_camera(
                    CAMERA_NAME, extra={}, timeout=0
                )
            )
            check_detections_output(
                get_detections_from_camera_result, person, MIN_CONFIDENCE_PASSING
            )

    @pytest.mark.asyncio
    async def test_capture_all_from_camera(self):
        service = get_vision_service(WORKING_CONFIG_DICT, people=PERSONS)
        for person in PERSONS:
            capture_all_result = await service.capture_all_from_camera(
                CAMERA_NAME,
                return_image=True,
                return_classifications=True,
                return_detections=True,
                return_object_point_clouds=True,
            )
            check_detections_output(
                capture_all_result.detections, person, MIN_CONFIDENCE_PASSING
            )

    @pytest.mark.asyncio
    async def test_add_embedding(self):
        service = get_vision_service(WORKING_CONFIG_DICT, people=[PERSON_TO_ADD])
        # should fail
        get_detections_from_camera_result = (
            await service.get_detections_from_camera(
                CAMERA_NAME, extra={}, timeout=0
            )
        )
        check_detections_output_fail(
            get_detections_from_camera_result, PERSON_TO_ADD, MIN_CONFIDENCE_PASSING
        )

        # re-init so camera is reset
        service = get_vision_service(WORKING_CONFIG_DICT, people=[PERSON_TO_ADD])
        with open(os.path.join("./tests", "img", PERSON_TO_ADD + ".jpg"), "rb") as image_file:
            b64 = base64.b64encode(image_file.read())
        await service.do_command({"command": "write_embedding", "image_ext": "jpg", "embedding_name": "cotillard", "image_base64": b64})
        await service.do_command({"command": "recompute_embeddings"})
        get_detections_from_camera_result = (
            await service.get_detections_from_camera(
                CAMERA_NAME, extra={}, timeout=0
            )
        )
        check_detections_output(
            get_detections_from_camera_result, PERSON_TO_ADD, MIN_CONFIDENCE_PASSING
        )
        shutil.rmtree(os.path.join("./tests", "img", PERSON_TO_ADD))

    @pytest.mark.asyncio
    async def test_default_camera_behavior(self):
        service = get_vision_service(WORKING_CONFIG_DICT, people=PERSONS)
        
        result = await service.get_detections_from_camera(
            "", extra={}, timeout=0
        )
        assert result is not None

        result = await service.capture_all_from_camera(
            "", return_detections=True
        )
        assert result is not None
        assert result.detections is not None

        with pytest.raises(ValueError) as excinfo:
            await service.get_detections_from_camera(
                "not-cam", extra={}, timeout=0
            )
        assert CAMERA_NAME in str(excinfo.value)
        with pytest.raises(ValueError) as excinfo:
            await service.capture_all_from_camera(
                "not-cam", return_detections=True
            )
        assert CAMERA_NAME in str(excinfo.value)

def check_detections_output(
    detections: List[Detection], target_class: str, target_confidence: float
):
    assert len(detections) == SINGLE_PERSON_PICTURE
    assert detections[0]["class_name"] == target_class
    assert detections[0]["confidence"] > MIN_CONFIDENCE_PASSING
    assert detections[0]["x_min"] is not None
    assert detections[0]["y_min"] is not None
    assert detections[0]["x_max"] is not None
    assert detections[0]["y_max"] is not None
    assert 0.0 < detections[0]["x_min_normalized"] < 1.0
    assert 0.0 < detections[0]["y_min_normalized"] < 1.0
    assert 0.0 < detections[0]["x_max_normalized"] < 1.0
    assert 0.0 < detections[0]["y_max_normalized"] < 1.0 

def check_detections_output_fail(
    detections: List[Detection], target_class: str, target_confidence: float
):
    assert len(detections) == SINGLE_PERSON_PICTURE
    assert detections[0]["class_name"] != target_class
