import json
import onnx

model_path = '../checkpoints/run_20250118-141251/segformer_resnet50_dfl.onnx'
model = onnx.load(model_path)

class_names = {
    0: 'background',
    1: 'pedestrian_crossing',
    2: 'bike_crossing'
}

m1 = model.metadata_props.add()
m1.key = 'model_type'
m1.value = json.dumps('Segmentor')

m2 = model.metadata_props.add()
m2.key = 'class_names'
m2.value = json.dumps(class_names)

# optional, if you want to standarize input after normalisation
m4 = model.metadata_props.add()
m4.key = 'standardization_mean'
m4.value = json.dumps([0.485, 0.456, 0.406])

m5 = model.metadata_props.add()
m5.key = 'standardization_std'
m5.value = json.dumps([0.229, 0.224, 0.225])

# Get the path without onnx extension and add '_w_metadata.onnx'
model_path_to_save = model_path[:-5] + '_w_metadata.onnx'
onnx.save(model, model_path_to_save)
