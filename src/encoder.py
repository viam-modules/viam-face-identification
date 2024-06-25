from src.models import utils
from torchvision.utils import save_image
import numpy.random as rd


class Encoder:
    def __init__(self, model_name, align, normalization, debug) -> None:
        self.model_name = model_name
        self.transform, self.translator, self.face_recognizer = utils.get_all(
            self.model_name
        )
        self.align = (align,)
        self.normalization = normalization
        self.debug = debug

    def encode(self, face, is_ir):
        img = self.transform(face)
        if self.debug:
            _id = rd.randint(0, 10000)
            save_image(img, f"./transformed_{_id}.png")
        if is_ir:
            three_channeled_image = self.translator(img)
            if self.debug:
                save_image(three_channeled_image, f"./three_channel_{_id}.png")
        else:
            three_channeled_image = img
            if self.debug:
                save_image(
                    three_channeled_image, f"./three_channel_no_translate{_id}.png"
                )
        embed = self.face_recognizer(three_channeled_image)
        return embed[0]
