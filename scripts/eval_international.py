"""Measure top-1 / top-5 accuracy of an ONNX international classifier on a folder of Stanford Cars
test images (images + labels.json mapping '<idx>' -> class name), and report the max-softmax
confidence distribution (used later to set routing thresholds).

Usage:
    python scripts/eval_international.py <model.onnx> <raw_labels.json> <images_dir> [limit]
"""
import json, os, sys
import numpy as np
import onnxruntime as ort
from PIL import Image

model, raw_labels, folder = sys.argv[1:4]
limit = int(sys.argv[4]) if len(sys.argv) > 4 else 10**9
id2label = {int(k): v for k, v in json.load(open(raw_labels)).items()}
name2id = {v: k for k, v in id2label.items()}
truth = json.load(open(os.path.join(folder, "labels.json")))

sess = ort.InferenceSession(model, providers=["CPUExecutionProvider"])
inp = sess.get_inputs()[0].name
top1 = top5 = n = skipped = 0
confs_ok, confs_bad = [], []
for key, name in list(truth.items())[:limit]:
    if name not in name2id:
        skipped += 1
        continue
    path = os.path.join(folder, f"{key}.jpg")
    im = Image.open(path).convert("RGB").resize((224, 224), Image.BILINEAR)
    x = ((np.asarray(im, np.float32) / 255 - 0.5) / 0.5).transpose(2, 0, 1)[None]
    lg = sess.run(None, {inp: x})[0][0]
    p = np.exp(lg - lg.max()); p /= p.sum()
    order = np.argsort(-p)
    n += 1
    ok = order[0] == name2id[name]
    top1 += ok
    top5 += name2id[name] in order[:5]
    (confs_ok if ok else confs_bad).append(float(p[order[0]]))
print(f"images={n} skipped(label-mismatch)={skipped}")
print(f"top1={top1 / n:.4f} top5={top5 / n:.4f}")
print("median conf correct:", np.median(confs_ok) if confs_ok else None, " median conf wrong:", np.median(confs_bad) if confs_bad else None)
