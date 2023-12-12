# VIAM FACE IDENTIFICATION MODULE

This is a [Viam module](https://docs.viam.com/extend/modular-resources/) providing a model of vision service for face identification relying on [deepface](https://github.com/serengil/deepface)

<p align="center">
 <img src="https://github.com/viam-labs/viam-face-identification/blob/main/img/results.png" width=100%, height=100%>
 </p>

## Getting started

To use this module, follow these instructions to [add a module from the Viam Registry](https://docs.viam.com/modular-resources/configure/#add-a-module-from-the-viam-registry) and select the `viam:vision:deepface-identification` model from the [`deepface-identification` module](https://app.viam.com/module/viam/deepface_identification).
This module implements the method `GetDetections()` of the [vision service API](https://docs.viam.com/services/vision/#api).

## Installation with `pip install` 

```
pip install -r requirements.txt
```

## Configure your `deepface_identification` vision service

> [!NOTE]  
> Before configuring your vision service, you must [create a robot](https://docs.viam.com/manage/fleet/robots/#add-a-new-robot).

Navigate to the **Config** tab of your robot’s page in [the Viam app](https://app.viam.com/). Click on the **Services** subtab and click **Create service**. Select the `Vision` type, then select the `deepface_identification` model. Enter a name for your service and click **Create**.

### Example
In the example bellow,  all faces in any pictures in the directory `French_Team`, will have an embedding associated to the label `Bleus`. The supported format for images of known face are PNG and JPEG.
#### Example of directory tree and config
```
path
└── to
    └── known_faces
        └── Zinedine_Zidane
        │   └── zz_1.png
        │   └── zz_2.jpeg
        │   └── zz_3.jpeg
        │ 
        └── Jacques_Chirac
        │   └── jacques_1.jpeg
        │
        └── French_Team
        |   └── ribery.jpeg
        |   └── vieira.png
        |   └── thuram.jpeg
        |   └── group_picture.jpeg
        │ 
        └── Italian_Team
            └── another_group_picture.png
```

#### Example of config for the above directory tree

```json
{
  "modules": [
    {
      "executable_path": "/path/to/run.sh",
      "name": "myidentifier",
      "type": "local"
    }
  ],
  "services": [
    {
      "attributes": {
        "camera_name": "cam",
        "dataset_path": "/path/to/known_faces",
        "label_and_directories": {
          "jacques": "Jacques_Chirac",
          "zizou": "Zinedine_Zidane",
          "Bleus": "French_Team",
          "Azzurri": "Italian_Team"
        }
      },
      "name": "detector-module",
      "type": "vision",
      "namespace": "rdk",
      "model": "viam:vision:deepface-identification"
    }
  ]
}
```



### Attributes description

The following attributes are available to configure your deepface module:


| Name                       | Type   | Inclusion    | Default | Description                                                                                  |
| -------------------------- | ------ | ------------ | ------- | -------------------------------------------------------------------------------------------- |
| `camera_name`              | string | **Required** |         | Camera name to be used as input for identifying faces.                                        |
| `extractor_model`          | string | Optional     | `'opencv'`| Model to be used to extract faces before computing embedding. See [available extractors](#extractors-and-encoders-available).                                 |
| `extraction_threshold`     | int    | Optional     | `3`     | Confidence threshold for face extraction.                                                    |
| `grayscale`                | bool   | Optional     | `False` | Convert input images to grayscale before processing if set to `True`.                         |
| `enforce_detection`        | bool   | Optional     | `False` | Set this value to true to raise an error if no faces are detected in the input image. This is a risky parameter; it is safer to check the number of detections and confidences associated (output of `get_detections()`). |
| `align`                    | bool   | Optional     | `True`  | Perform facial alignment during the face embedding process if set to `True`.                   |
| `face_embedding_model`     | string | Optional     | `'ArcFace'`| Model used for generating face embeddings. See [available encoding models](#extractors-and-encoders-available).                                                   |
| `normalization`            | string | Optional     | `'base'`  | Normalization method applied to face embeddings. |
| `dataset_path`             | string | Optional     |        | Path to the dataset used for face recognition.                                               |
| `label_and_directories`    | dict   | Optional     |         | Dictionary mapping class labels to corresponding directories in the dataset. See [example](#example-of-tree-file).                |
| `distance_metric`          | string | Optional     | `'cosine'`| Distance metric used for face recognition. This attribute can be set to `'cosine'`, `'euclidean'` and `'euclidean_l2'`.                                                  |
| `identification_threshold` | float  | Optional     |    | Threshold for identifying faces. Faces with similarity scores below this threshold are considered `'unknown'`. This value should depend on both `face_embedding_model` and `distance_metric`. **WARNING**: If left empty, the module will assign a value from [this table](#thresholds-for-face-recoignition-models-and-similarity-distances) depending on model and metric. If you want the module to return all detections without any threshold, `identification_threshold` should be set to `0`. |
| `sigmoid_steepness`            | float | Optional     | `10`  | Steepness of the function mapping confidence to distance. See [here](#distance-to-confidence-function) for plots with different values.  |

## Supplementaries
#### Extractors and encoders available

| Extractors    | Encoders      |
|---------------|---------------|
| ` 'opencv' `      | ` 'VGG-Face' `    |
| ` 'ssd' `         | ` 'Facenet' `     |
| ` 'dlib' `        | ` 'Facenet512' `  |
| ` 'mtcnn' `       | ` 'OpenFace' `    |
| ` 'retinaface' `  | ` 'DeepFace' `    |
| ` 'mediapipe' `   | ` 'DeepID' `      |
| ` 'yolov8' `      | ` 'ArcFace' `     |
| ` 'yunet' `       | ` 'Dlib' `        |
| ` 'fastmtcnn' `   | ` 'SFace' `       |




#### Thresholds for face Recoignition Models and similarity distances
Value assigned to `identification_threshold` if empty. For source, see [`Deepface.commons.distance.findThreshold()`](https://github.com/serengil/deepface/blob/master/deepface/commons/distance.py#L28).

| Model       | `cosine` threshold | `euclidean` threshold | `euclidean_l2` threshold |
|-------------|-------------------|--------------------|-----------------------|
| `'VGG-Face'`    | 0.40              | 0.60               | 0.86                  |
| `'Facenet'`     | 0.40              | 10.0               | 0.80                  |
| `'Facenet512'`  | 0.30              | 23.56              | 1.04                  |
| `'ArcFace'`     | 0.68              | 4.15               | 1.13                  |
| `'Dlib'`        | 0.07              | 0.6                | 0.4                   |
| `'SFace'`       | 0.593             | 10.734             | 1.055                 |
| `'OpenFace'`    | 0.10              | 0.55               | 0.55                  |
| `'DeepFace'`    | 0.23              | 64.0               | 0.64                  |
| `'DeepID'`      | 0.015             | 45.0               | 0.17                  |

#### Distance to confidence function 
The function that maps confidence to distance is given by:
$c = \frac{1}{1 + e^{s \cdot (d - 0.5)}}$, where $s$ is `sigmoid_steepness`, $c$ is the confidence and $d$ the distance.

<p align="center">
 <img src="https://github.com/viam-labs/viam-face-identification/blob/main/img/sigmoid_plot.png" width=50%, height=50%>
 </p>