# Model for pedestrian and bicycle crossings segmentation in QGIS

![Results](media/res_7.jpg)

## Model

The model is based on Segformer architecture with ResNet-50 backbone. The model was exported to onxx format and can be downloaded from the [onnx_model](onnx_model) folder.

## Data

The data are aerial images from the [Poznan 2022 aerial orthophoto high resolution](https://qms.nextgis.com/geoservices/5693/) map available in QGIS.
The images were created with the [Deepness plugin](https://plugins.qgis.org/plugins/deepness/), with the following parameters:
- `Resolution [cm/px]`: 10.00
- `Tile size [px]`: 512
- `Batch size`: 1
- `Tile overlap [%]`: 10

The images come from the following locations:
- 0 - 348 - PUT Campus and surrounding areas,
- 349 - 449 - Plewiska and Komorniki,
- 450 - 497 - Skórzewo,
- 498 - 520 - Ławica and Junikowo,
- 521 - 595 - Sołacz, Winiary, and South-Western Podolany ,
- 596 - 795 - Grunwald, Górczyn, Łazarz (from [https://universe.roboflow.com/zpo/pedestrian_bicycle_crossings](https://universe.roboflow.com/zpo/pedestrian_bicycle_crossings))*,
- 796 - 842 - Sejny (North-Eastern Poland) - from Geoportal Polska Orthophoto Poland,
- 843 - 945 - Żegrze (from [https://universe.roboflow.com/obraz/obrazy](https://universe.roboflow.com/obraz/obrazy))*,
- 946 - 1057 - Wilda and Dębiec (from [https://universe.roboflow.com/obrazy-tisrw/obrazy-kublo](https://universe.roboflow.com/obrazy-tisrw/obrazy-kublo))*,
- 1058 - 1385 - Winogrady and Szeląg (from [https://universe.roboflow.com/zpo/pedestrian_bicycle_crossings](https://universe.roboflow.com/zpo/pedestrian_bicycle_crossings))*.

\* Images from the Roboflow platform were adjust to be in the same format as the images from CVAT. Additionally, they were checked for mistakes in the annotations, and if necessary, they were corrected.

## Dependencies for training

The following dependencies are required to train the model:
- install PyTorch (https://pytorch.org/get-started/locally/) - check the proper version for your system,
- install requirements from `requirements.txt` file:
```bash
pip install -r requirements.txt
```
- to use the Segformer model, upgrade the library with:
```bash
pip install --upgrade git+https://github.com/qubvel/segmentation_models.pytorch
```

Installing package 
```bash
pip install -e .
```

## How to train the model

The training script allows you to train models with different architectures and backbones. The available models are:

### Segformer (ResNet-50):
```bash
python scripts/run_training.py --model segformer_resnet50 
```

### FPN (ResNet-34):
```bash
python scripts/run_training.py --model fpn_resnet34 
```

### UNet++ (EfficientNet-B0):
```bash
python scripts/run_training.py --model unet_effb0
```
### UNet++ (ResNet-34)

```bash
python scripts/run_training.py --model unet_resnet34
```
### UNet++ (MobileNetV2)
```bash
python scripts/run_training.py --model unet_mobilenetv2 
```

### DeepLabV3 (ResNet-34):
```bash
python scripts/run_training.py --model deeplabv3_resnet34 
```

## Model evaluation
```bash
python scripts/run_evaluation.py --model unet_effb0 --checkpoint checkpoints/<checkpoint_folder>/best-checkpoint.ckpt --format ckpt --num_samples 5
```
Example
```bash
python scripts/run_evaluation.py --model deeplabv3plus_resnet34 --checkpoint checkpoints/run_20250105-144034/best-checkpoint.ckpt --format ckpt --num_samples 10  --batch_size 4
```

## Training process and metrics on the test set
![Training](media/segformer_resnet50_combined_loss_longer_training.png)
![Metrics](media/metrics_segformer_resnet50_combined_loss_longer_training.jpg)

## Predicted results

![Results](media/res_1.jpg)
![Results](media/res_2.jpg)
![Results](media/res_3.jpg)
![Results](media/res_4.jpg)
![Results](media/res_5.jpg)
![Results](media/res_6.jpg)

