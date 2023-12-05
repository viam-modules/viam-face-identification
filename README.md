# viam-face-identification **IN PROGRESS**

This is a [Viam module](https://docs.viam.com/extend/modular-resources/) providing a model of vision service for face identification relying on [deepface](https://github.com/serengil/deepface)

## Getting started

To use this module, follow these instructions to [add a module from the Viam Registry](https://docs.viam.com/modular-resources/configure/#add-a-module-from-the-viam-registry) and select the `viam:vision:deepface-identification` model from the [`deepface-identification` module](https://app.viam.com/module/viam/deepface_identification).
This module implements the method `GetDetections()` of the [vision service API](https://docs.viam.com/services/vision/#api).

### Installation with `pip install` -> Not here yet

```
pip install -r requirements.txt
```

## Configure your `deepface_identification` vision service

> [!NOTE]  
> Before configuring your vision service, you must [create a robot](https://docs.viam.com/manage/fleet/robots/#add-a-new-robot).

Navigate to the **Config** tab of your robot’s page in [the Viam app](https://app.viam.com/). Click on the **Services** subtab and click **Create service**. Select the `Vision` type, then select the `deepface_identification` model. Enter a name for your service and click **Create**.


#### Example of dataset directory tree of known faces
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
| `extractor_model`          | string | Optional     | `opencv`| Model to be used to extract faces before computing embedding.                                 |
| `extraction_threshold`     | int    | Optional     | `3`     | Confidence threshold for face extraction.                                                    |
| `grayscale`                | bool   | Optional     | `False` | Convert input images to grayscale before processing if set to `True`.                         |
| `enforce_detection`        | bool   | Optional     | `False` | Set this value to true to raise a `ValueError` if no faces are detected in the input image. This is a risky parameter; it is safer to check the number and confidence of the output of `get_detections()`. |
| `align`                    | bool   | Optional     | `True`  | Perform facial alignment during the face embedding process if set to `True`.                   |
| `face_embedding_model`     | string | Optional     | `ArcFace`| Model used for generating face embeddings.                                                   |
| `normalization`            | string | Optional     | `base`  | Normalization method applied to face embeddings. |
| `dataset_path`             | string | Optional     |        | Path to the dataset used for face recognition.                                               |
| `label_and_directories`    | dict   | Optional     |         | Dictionary mapping class labels to corresponding directories in the dataset. See [example](#example-of-tree-file).                |
| `distance_metric`          | string | Optional     | `cosine`| Distance metric used for face recognition. This attribute can be set to `cosine`, `euclidean` and `euclidean_l2`.                                                  |
| `identification_threshold` | float  | Optional     | `None`  | Threshold for identifying faces. If set, faces with similarity scores below this threshold are considered unidentified. |
