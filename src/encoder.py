from deepface.DeepFace import represent
from .ir import utils

DEEPFACE_ENCODERS = [
  "VGG-Face", 
  "Facenet", 
  "Facenet512", 
  "OpenFace", 
  "DeepFace", 
  "DeepID", 
  "ArcFace", 
  "Dlib", 
  "SFace",
]

class Encoder:
    def __init__(self, 
                 model_name, 
                 align, 
                 normalization) -> None:
        self.model_name = model_name
        if self.model_name == 'ir':
            self.transform, self.translator, self.resnet = utils.get_all()
        self.align = align, 
        self.normalization = normalization
        
    def encode(self, face):
        if self.model_name in DEEPFACE_ENCODERS:
            embedding = represent(img_path=face, 
                                model_name=self.model_name, 
                                detector_backend="skip",
                                align=self.align,
                                normalization=self.normalization)
            return embedding[0]["embedding"]
        if self.model_name == 'ir':
            img = self.transform(face[0])
            r = img[0, :, :] 
            g = img[1, :, :]
            if (r==g).all():
                translated_img = self.translator(img.unsqueeze(0)) 
            else:
                translated_img =img.unsqueeze(0)
            embed = self.resnet(translated_img)
            return embed.numpy()[0]
            