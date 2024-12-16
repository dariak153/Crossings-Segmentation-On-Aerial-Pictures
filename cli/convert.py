import torch
import torch.onnx
from lightningmodule import segmentationModule

# Path to the saved checkpoint file
ckpt_path = '../saved_models/best_model.ckpt'

# Load the model
model = segmentationModule.MySegmentationModel.load_from_checkpoint(ckpt_path)
model.eval()  # Set the model to evaluation mode

# Dummy input for the ONNX export (change the size to match your input)
dummy_input = torch.randn(1, 3, 512, 512, device='cuda')  # Adjust the dimensions as necessary

# Path to save the ONNX model
onnx_path = '../saved_models/best_model.onnx'

# Export the model to ONNX
torch.onnx.export(
    model,                  # Model being run
    dummy_input,            # Model input (or a tuple for multiple inputs)
    onnx_path,              # Where to save the model
    export_params=True,     # Store the trained parameter weights inside the model file
    opset_version=11,       # ONNX version to export to
    do_constant_folding=True,  # Whether to execute constant folding for optimization
    input_names=['input'],  # Input name for the graph
    output_names=['output']  # Output name for the graph
)
print(f"Model has been converted to ONNX and saved at {onnx_path}")
