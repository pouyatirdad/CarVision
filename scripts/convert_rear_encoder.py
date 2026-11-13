"""Export the OpenCLIP image encoder used by the rear-view recogniser to ONNX (+ weight-only 8-bit MatMulNBits).

Development only. Default model: OpenCLIP ViT-B-32, pretrained "datacomp_xl_s13b_b90k" (weights: MIT licence,
https://huggingface.co/laion/CLIP-ViT-B-32-DataComp.XL-s13B-b90K). See docs/REAR-VIEW.md for why this encoder.

Usage:
    python scripts/convert_rear_encoder.py <out_dir> [arch] [pretrained]

Writes:
    <out_dir>/rear-encoder.fp32.onnx   (reference, NOT shipped)
    <out_dir>/rear-encoder.onnx        (8-bit MatMulNBits, block 128; copy to public/models/ under a NEW versioned name,
                                        e.g. rear-encoder-dc-b32-v2.onnx, and update MODELS.rear.file - the Cache API is keyed by URL)

Input : float32 [1, 3, 224, 224] RGB; resize the SHORT side to 224 (bicubic), centre-crop 224x224,
        then (x/255 - mean) / std with the CLIP mean/std (see MODELS.rear in src/config.js).
Output: float32 [1, D] image embedding (not normalised; the head L2-normalises it).
"""
import os
import sys

import numpy as np
import torch

import open_clip

out = sys.argv[1]
arch = sys.argv[2] if len(sys.argv) > 2 else "ViT-B-32"
pretrained = sys.argv[3] if len(sys.argv) > 3 else "datacomp_xl_s13b_b90k"
os.makedirs(out, exist_ok=True)

model, _, preprocess = open_clip.create_model_and_transforms(arch, pretrained=pretrained)
model.eval()
print(preprocess)


class Visual(torch.nn.Module):
    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, x):
        return self.m.encode_image(x)


fp32 = os.path.join(out, "rear-encoder.fp32.onnx")
dummy = torch.randn(1, 3, 224, 224)
torch.onnx.export(Visual(model), dummy, fp32, input_names=["pixel_values"], output_names=["embedding"], opset_version=17, dynamo=False)

# Weight-only 8-bit block quantisation (MatMulNBits). Measured on real car photos (docs/REAR-VIEW.md):
#   dynamic int8 (quantize_dynamic, activations quantised too): mean cosine vs fp32 0.969  -> unusable
#   MatMulNBits 4-bit, block 32:                               mean cosine 0.959          -> unusable
#   MatMulNBits 8-bit:                                         mean cosine 0.9999         -> shipped
# CLIP activations have large outliers, which is why activation quantisation fails.
import onnx  # noqa: E402
from onnxruntime.quantization.matmul_nbits_quantizer import MatMulNBitsQuantizer  # noqa: E402

int8 = os.path.join(out, "rear-encoder.onnx")
block = int(os.environ.get("BLOCK", "128"))
q = MatMulNBitsQuantizer(onnx.load(fp32), block_size=block, is_symmetric=True, bits=8)
q.process()
q.model.save_model_to_file(int8, use_external_data_format=False)

import onnxruntime as ort  # noqa: E402

x = np.random.default_rng(0).standard_normal((1, 3, 224, 224)).astype(np.float32)
with torch.no_grad():
    ref = model.encode_image(torch.tensor(x)).numpy()
for p in (fp32, int8):
    s = ort.InferenceSession(p, providers=["CPUExecutionProvider"])
    y = s.run(None, {"pixel_values": x})[0]
    cos = float((y * ref).sum() / np.linalg.norm(y) / np.linalg.norm(ref))
    print(os.path.basename(p), f"{os.path.getsize(p) / 1e6:.1f} MB", "cosine vs torch:", round(cos, 5))
