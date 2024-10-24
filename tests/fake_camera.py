import os
from typing import Any, Coroutine, List, Tuple, Union

from PIL import Image
from viam.components.camera import Camera
from viam.gen.component.camera.v1.camera_pb2 import GetPropertiesResponse
from viam.media.utils import pil
from viam.media.video import CameraMimeType, NamedImage, ViamImage
from viam.proto.common import ResponseMetadata


def read_image(name: str):
    return Image.open(os.path.join("./tests", "img", name + ".jpg"))


class FakeCamera(Camera):
    def __init__(self, name: str, persons: List):
        super().__init__(name=name)
        self.count = -1
        self.images = [read_image(person) for person in persons]

    async def get_image(self, mime_type: str = "") -> Coroutine[Any, Any, ViamImage]:
        self.count += 1
        print(f"self.count: {self.count}")
        print(f"len(images){len(self.images)}")
        if self.count > len(self.images):
            return IndexError("Already read all the images passed as input")
        return pil.pil_to_viam_image(self.images[self.count], CameraMimeType.JPEG)

    async def get_images(
        self,
    ) -> Coroutine[Any, Any, Tuple[Union[List[NamedImage], ResponseMetadata]]]:
        raise NotImplementedError

    async def get_properties(self) -> Coroutine[Any, Any, GetPropertiesResponse]:
        raise NotImplementedError

    async def get_point_cloud(self) -> Coroutine[Any, Any, Tuple[Union[bytes, str]]]:
        raise NotImplementedError
