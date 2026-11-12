"""Convert therealcyberlord/stanford-car-vit-patch16 (Apache-2.0, ViT-B/16 fine-tuned on Stanford Cars,
196 classes) to ONNX and optionally quantize the weights to int8 to shrink the download.

Usage:
    python scripts/convert_international.py <hf_model_dir> <out_dir>

Writes:
    <out_dir>/international-cars.fp32.onnx   (reference, ~330 MB, NOT shipped)
    <out_dir>/international-cars.onnx         (dynamic int8 weights, ~85 MB, shipped)
    <out_dir>/international-labels.raw.json   (id2label from the model config)

Input : float32 [1, 3, 224, 224]  RGB, (x/255 - 0.5) / 0.5, plain bilinear resize to 224x224
Output: float32 [1, 196] logits.
"""
import json
import os
import sys

import torch
from transformers import ViTForImageClassification

src, out = sys.argv[1], sys.argv[2]
os.makedirs(out, exist_ok=True)

model = ViTForImageClassification.from_pretrained(src).eval()
# Use plain (non-SDPA) attention so the ONNX graph only has widely supported ops.
try:
    model.config._attn_implementation = "eager"
except Exception:
    pass

fp32 = os.path.join(out, "international-cars.fp32.onnx")
torch.onnx.export(
    model,
    (torch.zeros(1, 3, 224, 224),),
    fp32,
    input_names=["pixel_values"],
    output_names=["logits"],
    opset_version=17,
    dynamo=False,
)
print("wrote", fp32, os.path.getsize(fp32) // 1_000_000, "MB")

json.dump(
    {int(k): v for k, v in model.config.id2label.items()},
    open(os.path.join(out, "international-labels.raw.json"), "w"),
    indent=1,
)

from onnxruntime.quantization import QuantType, quantize_dynamic

q = os.path.join(out, "international-cars.onnx")
quantize_dynamic(fp32, q, weight_type=QuantType.QInt8)
print("wrote", q, os.path.getsize(q) // 1_000_000, "MB")
