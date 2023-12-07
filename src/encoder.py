from deepface.DeepFace import represent

class Encoder:
    def __init__(self, 
                 model_name, 
                 align, 
                 normalization) -> None:
        self.model_name = model_name
        self.align = align, 
        self.normalization = normalization
        
    def encode(self, face):
        embedding = represent(img_path=face, 
                              model_name=self.model_name, 
                              detector_backend="skip",
                              align=self.align,
                              normalization=self.normalization)
        return embedding[0]["embedding"]