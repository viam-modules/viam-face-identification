# VIAM FACE IDENTIFICATION MODULE

This is a [Viam module](https://docs.viam.com/extend/modular-resources/) providing a model of vision service for face identification using the siamese network setup.

<p align="center">
 <img src="https://github.com/viam-labs/viam-face-identification/blob/main/img/results.png" width=100%, height=100%>
 </p>

## Getting started

To use this module, follow these instructions to [add a module from the Viam Registry](https://docs.viam.com/modular-resources/configure/#add-a-module-from-the-viam-registry) and select the `viam:vision:face-identification` model from the [`face-identification` module](https://app.viam.com/module/viam/deepface-identification).
This module implements the method `GetDetections()` of the [vision service API](https://docs.viam.com/services/vision/#api).

## Installation with `pip install` 

```
pip install -r requirements.txt
```

## Configure your `face_identification` vision service

> [!NOTE]  
> Before configuring your vision service, you must [create a robot](https://docs.viam.com/manage/fleet/robots/#add-a-new-robot).

Navigate to the **Config** tab of your robot’s page in [the Viam app](https://app.viam.com/). Click on the **Services** subtab and click **Create service**. Select the `Vision` type, then select the `face_identification` model. Enter a name for your service and click **Create**.

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
      "model": "viam:vision:face-identification"
    }
  ]
}
```


### Attributes description

The following attributes are available to configure your deepface module:


| Name                          | Type   | Inclusion    | Default     | Description                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          |
| ----------------------------- | ------ | ------------ | ----------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `camera_name`                 | string | **Required** |             | Camera name to be used as input for identifying faces.                                                                                                                                                                                                                                                                                                                                                                                                                                               |
| `extractor_model`             | string | Optional     | `'yunet'`   | Model to be used to extract faces before computing embedding. See [available extractors](#extractors-and-encoders-available).                                                                                                                                                                                                                                                                                                                                                                        |
| `extraction_threshold`        | int    | Optional     | `.6`        | Confidence threshold for face extraction.                                                                                                                                                                                                                                                                                                                                                                                                                                                            |
| `grayscale`                   | bool   | Optional     | `False`     | Convert input images to grayscale before processing if set to `True`.                                                                                                                                                                                                                                                                                                                                                                                                                                |
| `always_run_face_recognition` | bool   | Optional     | `False`     | Set this value to true to raise an error if no faces are detected in the input image. This is a risky parameter; it is safer to check the number of detections and confidences associated (output of `get_detections()`).                                                                                                                                                                                                                                                                            |
| `align`                       | bool   | Optional     | `True`      | Perform facial alignment during the face embedding process if set to `True`.                                                                                                                                                                                                                                                                                                                                                                                                                         |
| `face_embedding_model`        | string | Optional     | `'facenet'` | Model used for generating face embeddings. See [available encoding models](#extractors-and-encoders-available).                                                                                                                                                                                                                                                                                                                                                                                      |
| `picture_directory`           | string | Optional     |             | Path to the dataset used for face recognition.                                                                                                                                                                                                                                                                                                                                                                                                                                                       |
| `distance_metric`             | string | Optional     | `'cosine'`  | Distance metric used for face recognition. This attribute can be set to `'cosine'`, `'manhattan'` ( = norm L1, or Manhattan distance) and `'euclidean'`. Default and recommanded is `'cosine'`.                                                                                                                                                                                                                                                                                                      |
| `identification_threshold`    | float  | Optional     |             | Threshold for identifying faces. Faces with similarity scores below this threshold are considered `'unknown'`. This value should depend on both `face_embedding_model` and `distance_metric`. **WARNING**: If left empty, the module will assign a value from [this table](#thresholds-for-face-recoignition-models-and-similarity-distances) depending on model and metric. If you want the module to return all detections without any threshold, `identification_threshold` should be set to `0`. |
| `sigmoid_steepness`           | float  | Optional     | `10`        | Steepness of the function mapping confidence to distance. See [here](#distance-to-confidence-function) for plots with different values.

## Vision Service API

The event-manager resource implements the [rdk vision service API](https://docs.viam.com/services/vision/#api).

### do_command()

Examples:

```python
await em.do_command({"command": "recompute_embeddings"}) # recompute embeddings from picture_directory
await em.do_command({"command": "write_embedding", "image_ext": "jpg", "embedding_name": "cotillard", "image_base64": b64}) # Write a new image grouped with embedding name "cotillard", from a base64 jpg image
```

#### recompute_embeddings

Recompute embeddings from picture_directory

#### write_embedding

Write a new image embedding based on base64 encoded image passed in as *image_base64*.
*embedding_name* must be included, and will dictate if it is grouped with an existing embedding of this name or a new embedding.
*image_ext* is required, and is the extension of the image passed in.

Note that new embedding images will be written with an auto-generated UUID-based filename.

## IR input support
To support IR inputs and cross-domain face recognition, we reproduced [Prepended Domain Transformer: Heterogeneous Face Recognition without Bells and Whistle (*Marcel et al.*)](https://arxiv.org/abs/2210.06529). 
<p align="center">
 <img src="https://github.com/viam-labs/viam-face-identification/blob/main/img/pdt.jpg" width=100%, height=100%>
<br>
 <sub><sup>Image Source: Marcel et al., 2022 <sub><sup>
 </p>


## Supplementaries
#### Extractors and encoders available
| Encoders                  |
| ------------------------- |
| ` 'facenet' `             |
| ` 'sface' ` (in progress) |


| Extractors  |
| ----------- |
| ` 'yunet' ` |



#### Thresholds for face recognition models and similarity distances
The value assigned to `identification_threshold` if empty.


| Model       | `cosine` threshold | `manhattan` threshold | `euclidean` threshold |
| ----------- | ------------------ | --------------------- | --------------------- |
| `'facenet'` | 0.35               | 1.1                   | 1.1                   |
| `'sface'`   | *in progress*      | *in progress*         | *in progress*         |

#### Distance to confidence function 
The function that maps confidence to distance is given by:
$c = \frac{1}{1 + e^{s \cdot (d - 0.5)}}$, where $s$ is `sigmoid_steepness`, $c$ is the confidence and $d$ the distance.

<p align="center">
 <img src="https://github.com/viam-labs/viam-face-identification/blob/main/img/sigmoid_plot.png" width=50%, height=50%>
 </p>


## References
1. Anjith George, Amir Mohammadi, and Sebastien Marcel. "Prepended Domain Transformer: Heterogeneous Face Recognition without Bells and Whistles." *IEEE Transactions on Information Forensics and Security*, 2022. [Link](
https://doi.org/10.48550/arXiv.2210.06529)
2. Balestriero, R., & LeCun, Y. (2022). Contrastive and Non-Contrastive Self-Supervised Learning Recover Global and Local Spectral Embedding Methods. In *Advances in Neural Information Processing Systems* (Vol. 35, pp. 26671–26685). Curran Associates, Inc. [Link](https://proceedings.neurips.cc/paper_files/paper/2022/file/aa56c74513a5e35768a11f4e82dd7ffb-Paper-Conference.pdf)


## PyInstaller build instructions

IMPORT NOTE: DEVELOPMENT SHOULD BE ON PYTHON3.11 or lower!

Run this to create your virtual environment:
```
./setup.sh
```

Run this to create your virtual environment:
Activate it bby running:
```
source .venv/bin/activate
```

Make sure that the requirements are installed:
```
pip3 install -r requirements.txt
```

Build the executable `dist/main`
```
python -m PyInstaller --onefile --hidden-import="googleapiclient" --add-data "./src/models/checkpoints:checkpoints"  src/main.py
```
