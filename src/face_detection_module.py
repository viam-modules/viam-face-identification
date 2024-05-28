from typing import ClassVar, List, Mapping, Sequence, Any, Dict, Optional, Union
from viam.media.video import CameraMimeType
from typing_extensions import Self
from viam.components.camera import Camera
from viam.proto.service.vision import Classification, Detection
from viam.services.vision import Vision
from viam.media.video import ViamImage
from viam.module.types import Reconfigurable
from viam.resource.types import Model, ModelFamily
from viam.proto.app.robot import ServiceConfig
from viam.proto.common import PointCloudObject, ResourceName
from viam.resource.base import ResourceBase
from viam.utils import ValueTypes
from viam.logging import getLogger
from .identifier import Identifier
from .utils import decode_image
from PIL import Image

EXTRACTORS = ["mediapipe:0", "mediapipe:1", "yunet"]

ENCODERS = ["sface", "facenet"]

LOGGER = getLogger(__name__)


class FaceIdentificationModule(Vision, Reconfigurable):
    MODEL: ClassVar[Model] = Model(ModelFamily("viam", "vision"), "face-identification")

    def __init__(self, name: str):
        super().__init__(name=name)

    @classmethod
    def new_service(
        cls, config: ServiceConfig, dependencies: Mapping[ResourceName, ResourceBase]
    ) -> Self:
        service = cls(config.name)
        service.reconfigure(config, dependencies)
        return service

    # Validates JSON Configuration
    @classmethod
    def validate_config(cls, config: ServiceConfig) -> Sequence[str]:
        detection_framework = (
            config.attributes.fields["extractor_model"].string_value or "yunet"
        )
        if detection_framework not in EXTRACTORS:
            raise Exception(
                "face_extractor_model must be one of: '"
                + "', '".join(EXTRACTORS)
                + "'."
            )
        model_name = (
            config.attributes.fields["face_embedding_model"].string_value or "facenet"
        )
        if model_name not in ENCODERS:
            raise Exception(
                "face embedding model (encoder) must be one of: '"
                + "', '".join(ENCODERS)
                + "'."
            )
        camera_name = config.attributes.fields["camera_name"].string_value
        if camera_name == "":
            raise Exception(
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
                    raise Exception(
                        "If default value is None, of_type argument can't be empty"
                    )
                type_default = of_type
            else:
                type_default = type(default)

            if type_default == bool:
                return config.attributes.fields[attribute_name].bool_value
            elif type_default == int:
                return int(config.attributes.fields[attribute_name].number_value)
            elif type_default == float:
                return config.attributes.fields[attribute_name].number_value
            elif type_default == str:
                return config.attributes.fields[attribute_name].string_value
            elif type_default == dict:
                return dict(config.attributes.fields[attribute_name].struct_value)

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
        LOGGER.info(f" Found {len(self.identifier.known_embeddings)} labelled groups.")

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
        extra: Mapping[str, Any],
    ) -> List[Classification]:
        return NotImplementedError

    async def get_classifications_from_camera(self) -> List[Classification]:
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
            self.identifier.known_embeddings = dict()
            self.identifier.compute_known_embeddings()
            LOGGER.info("Embeddings recomputed!")
            return {"result": "Embeddings recomputed!"}
        else:
            raise NotImplementedError
