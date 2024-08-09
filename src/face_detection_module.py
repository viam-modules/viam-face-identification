"""
This module provides a Viam Vision Service module
to perform face Re-Id.
"""

from typing import Any, ClassVar, Dict, List, Mapping, Optional, Sequence
from typing_extensions import Self

from viam.components.camera import Camera
from viam.logging import getLogger
from viam.media.video import CameraMimeType, ViamImage
from viam.module.types import Reconfigurable
from viam.proto.app.robot import ServiceConfig
from viam.proto.common import PointCloudObject, ResourceName
from viam.proto.service.vision import Classification, Detection
from viam.resource.base import ResourceBase
from viam.resource.types import Model, ModelFamily
from viam.services.vision import CaptureAllResult, Vision
from viam.utils import ValueTypes

from src.identifier import Identifier
from src.utils import decode_image

EXCEPTION_REQUIRED_CAMERA = Exception(
    "A camera name is required for face_identification vision service module."
)

EXTRACTORS = ["mediapipe:0", "mediapipe:1", "yunet"]

ENCODERS = ["sface", "facenet"]

DISTANCES = ["cosine", "euclidean", "manhattan"]

LOGGER = getLogger(__name__)


class FaceIdentificationModule(Vision, Reconfigurable):
    """FaceIdentificationModule is a subclass a Viam Vision Service"""

    MODEL: ClassVar[Model] = Model(ModelFamily("viam", "vision"), "face-identification")

    def __init__(self, name: str):
        super().__init__(name=name)
        self.camera = None
        self.camera_name = None
        self.identifier = None

    @classmethod
    def new_service(
        cls, config: ServiceConfig, dependencies: Mapping[ResourceName, ResourceBase]
    ) -> Self:
        """returns new vision service"""
        service = cls(config.name)
        service.reconfigure(config, dependencies)
        return service

    # Validates JSON Configuration
    @classmethod
    def validate_config(cls, config: ServiceConfig) -> Sequence[str]:
        """Validate config and returns a list of dependencies."""
        if "extractor_model" in config.attributes.fields:
            detection_framework = config.attributes.fields[
                "extractor_model"
            ].string_value
            if detection_framework not in EXTRACTORS:
                raise ValueError(
                    "face_extractor_model must be one of: '"
                    + "', '".join(EXTRACTORS)
                    + "'."
                    + "Got:"
                    + detection_framework
                )
        if "face_embedding_model" in config.attributes.fields:
            model_name = config.attributes.fields["face_embedding_model"].string_value
            if model_name not in ENCODERS:
                raise ValueError(
                    "face embedding model (encoder) must be one of: '"
                    + "', '".join(ENCODERS)
                    + "'."
                    + "Got:"
                    + model_name
                )
        if "distance_metric" in config.attributes.fields:
            distance = config.attributes.fields["distance_metric"].string_value
            if not distance:
                if distance not in DISTANCES:
                    if distance == "euclidean_l2":
                        LOGGER.error(
                            "Names of distance metrics has been updated in release 0.6.1t to %s",
                            DISTANCES,
                        )

                    raise ValueError(
                        "distance_metric attribute must be one of: '"
                        + "', '".join(DISTANCES)
                        + "'."
                        + "Got:"
                        + distance
                    )
        camera_name = config.attributes.fields["camera_name"].string_value
        if camera_name == "":
            raise ValueError(
                "A camera name is required for face_identification vision service module."
            )
        return [camera_name]

    def reconfigure(
        self, config: ServiceConfig, dependencies: Mapping[ResourceName, ResourceBase]
    ):
        self.camera_name = config.attributes.fields["camera_name"].string_value
        self.camera = dependencies[Camera.get_resource_name(self.camera_name)]

        def get_attribute_from_config(attribute_name: str, default, of_type=None):
            if attribute_name not in config.attributes.fields:
                return default

            if default is None:
                if of_type is None:
                    raise ValueError(
                        "If default value is None, of_type argument can't be empty"
                    )
                type_default = of_type
            else:
                type_default = type(default)

            if type_default is bool:
                return config.attributes.fields[attribute_name].bool_value
            if type_default is int:
                return int(config.attributes.fields[attribute_name].number_value)
            if type_default is float:
                return config.attributes.fields[attribute_name].number_value
            if type_default is str:
                return config.attributes.fields[attribute_name].string_value
            if type_default is dict:
                return dict(config.attributes.fields[attribute_name].struct_value)

            raise ValueError("can't parse attribute from config.")

        detector_backend = get_attribute_from_config("extractor_model", "yunet")
        extraction_threshold = get_attribute_from_config(
            "extractor_confidence_threshold", 0.6
        )
        grayscale = get_attribute_from_config("grayscale", False)
        enforce_detection = get_attribute_from_config(
            "always_run_face_recognition", False
        )
        align = get_attribute_from_config("align", True)
        model_name = get_attribute_from_config("face_embedding_model", "facenet")
        normalization = get_attribute_from_config("normalization", "base")
        picture_directory = config.attributes.fields["picture_directory"].string_value
        distance_metric_name = get_attribute_from_config("distance_metric", "cosine")
        identification_threshold = get_attribute_from_config(
            "identification_threshold", None, float
        )
        sigmoid_steepness = get_attribute_from_config("sigmoid_steepness", 10.0)
        self.identifier = Identifier(
            detector_backend=detector_backend,
            extraction_threshold=extraction_threshold,
            grayscale=grayscale,
            enforce_detection=enforce_detection,
            align=align,
            model_name=model_name,
            normalization=normalization,
            picture_directory=picture_directory,
            distance_metric_name=distance_metric_name,
            identification_threshold=identification_threshold,
            sigmoid_steepness=sigmoid_steepness,
            debug=False,
        )
        self.identifier.compute_known_embeddings()
        LOGGER.info("Found %s labelled groups.", len(self.identifier.known_embeddings))

    async def get_properties(
        self,
        *,
        extra: Optional[Mapping[str, Any]] = None,
        timeout: Optional[float] = None,
    ) -> Vision.Properties:
        return Vision.Properties(
            classifications_supported=False,
            detections_supported=True,
            object_point_clouds_supported=False,
        )

    async def capture_all_from_camera(
        self,
        camera_name: str,
        return_image: bool = False,
        return_classifications: bool = False,
        return_detections: bool = False,
        return_object_point_clouds: bool = False,
        *,
        extra: Optional[Mapping[str, Any]] = None,
        timeout: Optional[float] = None,
    ):
        viam_im = await self.camera.get_image(mime_type=CameraMimeType.JPEG)
        detections = None
        if return_detections:
            img = decode_image(viam_im)
            detections = self.identifier.get_detections(img)

        if not return_image:
            viam_im = None

        return CaptureAllResult(image=viam_im, detections=detections)

    async def get_object_point_clouds(
        self,
        camera_name: str,
        *,
        extra: Optional[Dict[str, Any]] = None,
        timeout: Optional[float] = None,
        **kwargs,
    ) -> List[PointCloudObject]:
        raise NotImplementedError

    async def get_detections(
        self,
        image: ViamImage,
        *,
        extra: Mapping[str, Any],
        timeout: float,
    ) -> List[Detection]:
        img = decode_image(image)
        return self.identifier.get_detections(img)

    async def get_classifications(
        self,
        image: ViamImage,
        count: int,
        *,
        extra: Optional[Mapping[str, Any]] = None,
        timeout: Optional[float] = None,
    ) -> List[Classification]:
        return NotImplementedError

    async def get_classifications_from_camera(
        self,
        camera_name: str,
        count: int,
        *,
        extra: Optional[Mapping[str, Any]] = None,
        timeout: Optional[float] = None,
    ) -> List[Classification]:
        return NotImplementedError

    async def get_detections_from_camera(
        self, camera_name: str, *, extra: Mapping[str, Any], timeout: float
    ) -> List[Detection]:
        im = await self.camera.get_image(mime_type=CameraMimeType.JPEG)
        img = decode_image(im)
        return self.identifier.get_detections(img)

    async def do_command(
        self,
        command: Mapping[str, ValueTypes],
        *,
        timeout: Optional[float] = None,
        **kwargs,
    ):
        if command["command"] == "recompute_embeddings":
            self.identifier.known_embeddings = {}
            self.identifier.compute_known_embeddings()
            LOGGER.info("Embeddings recomputed!")
            return {"result": "Embeddings recomputed!"}
        raise NotImplementedError
