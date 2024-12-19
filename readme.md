# Dataset for crosswalk segmentation in QGIS

## Dependencies for training

The following dependencies are required to train the model:
- install PyTorch (https://pytorch.org/get-started/locally/) - check the proper version for your system,
- install requirements from `requirements.txt` file:
```bash
pip install -r requirements.txt
```
!!!!!!
The folder for masks has been updated to annotated_data due to errors with loading data

## Data

The data are aerial images from the [Poznan 2022 aerial orthophoto high resolution](https://qms.nextgis.com/geoservices/5693/) map available in QGIS.
The images were created with the [Deepness plugin](https://plugins.qgis.org/plugins/deepness/), with the following parameters:
- `Resolution [cm/px]`: 10.00
- `Tile size [px]`: 512
- `Batch size`: 1
- `Tile overlap [%]`: 10

The images come from the following locations:
- 
- 349 - 449 - Plewiska and Komorniki,
- 450 - 497 - Skórzewo,
- 498 - 520 - Ławica and Junikowo,
- 521 - 595 - Sołacz, Winiary, and South-Western Podolany 
## Trening model
```bash
python model.py
```
## Weights to model
```bash
python imbalance_weights.py
```
## Visualization predictions 
```bash
python evaluate.py --checkpoint checkpoints/best-checkpoint.ckpt --images_dir data/data --masks_dir data/annotated_data/all_in_one --num_samples 5
```
## Visualization metrics
```bash
python visualize_metrics.py --csv_path logs/segmentation_model/version_0/metrics.csv --title_suffix "(Segmentacja wieloklasowa)"
```
