from deepface.commons import functions
from viam.logging import getLogger

LOGGER = getLogger(__name__)

class Extractor:
    def __init__(self,
                 target_size: (int, int),
                 extraction_threshold: float, 
                 detector_backend:str, 
                 grayscale=False, 
                 enforce_detection=False, 
                 align = True) -> None:
        
        self.target_size = target_size
        self.detector_backend = detector_backend
        self.extraction_threshold = extraction_threshold
        self.grayscale = grayscale
        self.enforce_detection = enforce_detection
        self.align =  align
       
    
    def extract_faces(self, img):
        """_summary_
        
        Extracts face using self.extractor

        Args:
            img (np.array(h, w, 3)): BGR format

        Returns:
            list of: (face, region (x, y, w, h), confidence), 
            face.shape: target_size
            region: is the region of the face in the original image coordinate.
            confidence varies depending on the detector
        """        
        faces = functions.extract_faces(img=img,
                                            target_size=self.target_size,
                                            detector_backend=self.detector_backend,
                                            grayscale=False,
                                            enforce_detection=self.enforce_detection,
                                            align=self.align)
        res = []
        for face in faces:
            if face[2]>self.extraction_threshold:
                res.append(face)    
        return res