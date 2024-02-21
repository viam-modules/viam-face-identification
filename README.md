# VIAM FACE IDENTIFICATION MODULE

This is a [Viam module](https://docs.viam.com/extend/modular-resources/) providing a model of vision service for face identification relying on [deepface](https://github.com/serengil/deepface)

<p align="center">
 <img src="https://github.com/viam-labs/viam-face-identification/blob/main/img/results.png" width=100%, height=100%>
 </p>

## Getting started

To use this module, follow these instructions to [add a module from the Viam Registry](https://docs.viam.com/modular-resources/configure/#add-a-module-from-the-viam-registry) and select the `viam:vision:deepface-identification` model from the [`deepface-identification` module](https://app.viam.com/module/viam/deepface-identification).
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
In the example below, all faces detected in any pictures within the directory `French_Team` will have an embedding associated with the label `French_Team`. The supported image formats for known faces are PNG and JPEG.
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
        "picture_directory": "/path/to/known_faces",
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
| `extractor_model`          | string | Optional     | `'yunet'`| Model to be used to extract faces before computing embedding. See [available extractors](#extractors-and-encoders-available).                                 |
| `extraction_threshold`     | int    | Optional     | `.6`     | Confidence threshold for face extraction.                                                    |
| `grayscale`                | bool   | Optional     | `False` | Convert input images to grayscale before processing if set to `True`.                         |
| `always_run_face_recognition`        | bool   | Optional     | `False` | Set this value to true to raise an error if no faces are detected in the input image. This is a risky parameter; it is safer to check the number of detections and confidences associated (output of `get_detections()`). |
| `align`                    | bool   | Optional     | `True`  | Perform facial alignment during the face embedding process if set to `True`.                   |
| `face_embedding_model`     | string | Optional     | `'facenet'`| Model used for generating face embeddings. See [available encoding models](#extractors-and-encoders-available).                                                   |
| `picture_directory`             | string | Optional     |        | Path to the dataset used for face recognition.                                               |
| `distance_metric`          | string | Optional     | `'cosine'`| Distance metric used for face recognition. This attribute can be set to `'cosine'`, `'euclidean'` and `'euclidean_l2'`.                                                  |
| `identification_threshold` | float  | Optional     |    | Threshold for identifying faces. Faces with similarity scores below this threshold are considered `'unknown'`. This value should depend on both `face_embedding_model` and `distance_metric`. **WARNING**: If left empty, the module will assign a value from [this table](#thresholds-for-face-recoignition-models-and-similarity-distances) depending on model and metric. If you want the module to return all detections without any threshold, `identification_threshold` should be set to `0`. |
| `sigmoid_steepness`            | float | Optional     | `10`  | Steepness of the function mapping confidence to distance. See [here](#distance-to-confidence-function) for plots with different values.  |



## IR input support
***in progress***


## Supplementaries
#### Extractors and encoders available
| Encoders      |
|---------------|
| ` 'facenet' `    |
| ` 'sface' ` (in progress)    |


| Extractors    |
|---------------|
| ` 'yunet' `| 



#### Thresholds for face recognition models and similarity distances
The value assigned to `identification_threshold` if empty. For the source, see [`Deepface.commons.distance.findThreshold()`](https://github.com/serengil/deepface/blob/master/deepface/commons/distance.py#L28).

| Model       | `cosine` threshold | `euclidean` threshold | `euclidean_l2` threshold |
|-------------|-------------------|--------------------|-----------------------|
| `'facenet'`    | 0.35             | 1.1             | 1.1                  |

#### Distance to confidence function 
The function that maps confidence to distance is given by:
$c = \frac{1}{1 + e^{s \cdot (d - 0.5)}}$, where $s$ is `sigmoid_steepness`, $c$ is the confidence and $d$ the distance.

<p align="center">
 <img src="https://github.com/viam-labs/viam-face-identification/blob/main/img/sigmoid_plot.png" width=50%, height=50%>
 </p>
