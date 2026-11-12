"""Convert the Iranian car-name ResNet18 (RezaGooner/Multi-Task-Iranian-Vehicle-Monitoring-System, MIT)
to ONNX.

Usage:
    python scripts/convert_iranian.py <car_name_model.pth> <out.onnx>

The checkpoint is loaded with weights_only=True (no arbitrary pickle code execution).
Input : float32 [1, 3, 220, 165]  (H=220, W=165, RGB, ImageNet mean/std) - exactly the
        torchvision Resize((220, 165)) used by the upstream training/inference code.
Output: float32 [1, 27] logits (apply softmax in the app).
"""
import sys
import torch
from torchvision import models

src, dst = sys.argv[1], sys.argv[2]
NUM_CLASSES = 27

model = models.resnet18(weights=None)
model.fc = torch.nn.Linear(model.fc.in_features, NUM_CLASSES)
state = torch.load(src, map_location="cpu", weights_only=True)
model.load_state_dict(state)
model.eval()

dummy = torch.zeros(1, 3, 220, 165)
torch.onnx.export(
    model,
    dummy,
    dst,
    input_names=["input"],
    output_names=["logits"],
    opset_version=17,
    dynamo=False,
)
print("wrote", dst)
