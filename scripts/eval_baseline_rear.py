"""Baseline: the v1 classifiers (Stanford-Cars ViT int8 + Iranian ResNet18) and the v1 routing rule on a rear-view
test set collected with scripts/collect-commons.mjs. usage: python scripts/eval_baseline_rear.py <testset_dir> <repo>"""
import json
import os
import sys

import numpy as np
import onnxruntime as ort
from PIL import Image

testset, repo = sys.argv[1:3]
models = os.path.join(repo, "public", "models")
manifest = [m for m in json.load(open(os.path.join(testset, "manifest.json"), encoding="utf-8")) if m["view"] == "rear" and m["file"] not in {"rear/kia-sportage/0417.jpg", "rear/toyota-corolla/0212.jpg", "rear/toyota-rav4/0290.jpg", "rear/toyota-rav4/0300.jpg"}]

IR = {2: "ikco-dena", 3: "renault-tondar-90", 7: "ikco-peugeot-pars", 8: "ikco-peugeot-206", 9: "ikco-peugeot-206-sd", 10: "ikco-peugeot-207",
      11: "ikco-peugeot-405", 13: "saipa-pride", 15: "saipa-pride", 16: "saipa-quick", 18: "saipa-kia-rio", 19: "saipa-saina", 20: "ikco-samand",
      21: "saipa-shahin", 22: "ikco-soren", 23: "ikco-tara", 24: "saipa-tiba", 25: "saipa-tiba"}
INTL = {53: "hyundai-elantra", 159: "hyundai-elantra", 67: "toyota-corolla", 75: "bmw-3-series", 94: "bmw-3-series", 108: "hyundai-tucson",
        129: "toyota-camry", 179: "mercedes-c-class"}

det = ort.InferenceSession(os.path.join(models, "vehicle-detector.onnx"), providers=["CPUExecutionProvider"])
vit = ort.InferenceSession(os.path.join(models, "international-cars.onnx"), providers=["CPUExecutionProvider"])
res = ort.InferenceSession(os.path.join(models, "iranian-cars.onnx"), providers=["CPUExecutionProvider"])


def crop(img):
    s = 416
    w, h = img.size
    r = min(s / w, s / h)
    rs = img.resize((max(1, int(w * r)), max(1, int(h * r))), Image.BILINEAR)
    canvas = np.full((s, s, 3), 114, np.float32)
    a = np.asarray(rs, np.float32)[:, :, ::-1]
    canvas[: a.shape[0], : a.shape[1]] = a
    out = det.run(None, {det.get_inputs()[0].name: canvas.transpose(2, 0, 1)[None]})[0][0]
    grids, strides = [], []
    for st in (8, 16, 32):
        n = s // st
        yv, xv = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
        grids.append(np.stack([xv, yv], -1).reshape(-1, 2))
        strides.append(np.full((n * n, 1), st))
    g, st = np.concatenate(grids), np.concatenate(strides)
    xy = (out[:, :2] + g) * st
    wh = np.exp(out[:, 2:4]) * st
    scores = out[:, 4:5] * out[:, 5:]
    best = None
    for c in (2, 5, 7):
        for i in np.where(scores[:, c] > 0.35)[0]:
            x1, y1 = (xy[i] - wh[i] / 2) / r
            x2, y2 = (xy[i] + wh[i] / 2) / r
            key = scores[i, c] * (x2 - x1) * (y2 - y1)
            if best is None or key > best[0]:
                best = (key, (max(0, x1), max(0, y1), min(w, x2), min(h, y2)))
    return img.crop(tuple(int(v) for v in best[1])) if best else img


def softmax(x):
    e = np.exp(x - x.max())
    return e / e.sum()


def tensor(img, w, h, mean, std):
    a = np.asarray(img.resize((w, h), Image.BILINEAR), np.float32) / 255.0
    return ((a - mean) / std).transpose(2, 0, 1)[None].astype(np.float32)


stats = {"n": 0, "vit_in_set": 0, "vit_top1": 0, "ir_in_set": 0, "ir_top1": 0, "routed_correct": 0, "routed_wrong": 0, "uncertain": 0}
per = {}
for m in manifest:
    img = crop(Image.open(os.path.join(testset, m["file"])).convert("RGB"))
    pv = softmax(vit.run(None, {vit.get_inputs()[0].name: tensor(img, 224, 224, 0.5, 0.5)})[0][0])
    pr = softmax(res.run(None, {res.get_inputs()[0].name: tensor(img, 165, 220, np.array([0.485, 0.456, 0.406]), np.array([0.229, 0.224, 0.225]))})[0][0])
    cls = m["cls"]
    stats["n"] += 1
    if cls in INTL.values():
        stats["vit_in_set"] += 1
        stats["vit_top1"] += INTL.get(int(pv.argmax())) == cls
    if cls in IR.values():
        stats["ir_in_set"] += 1
        stats["ir_top1"] += IR.get(int(pr.argmax())) == cls
    if pv.max() >= 0.8:
        pred = INTL.get(int(pv.argmax()), "intl-other")
    elif pr.max() >= 0.8:
        pred = IR.get(int(pr.argmax()), "ir-other")
    else:
        pred = None
    if pred is None:
        stats["uncertain"] += 1
    elif pred == cls:
        stats["routed_correct"] += 1
    else:
        stats["routed_wrong"] += 1
    p = per.setdefault(cls, [0, 0])
    p[0] += 1
    p[1] += pred == cls

print(json.dumps(stats))
print("ViT top-1 on its own covered classes: %.1f%% of %d" % (100 * stats["vit_top1"] / max(1, stats["vit_in_set"]), stats["vit_in_set"]))
print("ResNet18 top-1 on its own covered classes: %.1f%% of %d" % (100 * stats["ir_top1"] / max(1, stats["ir_in_set"]), stats["ir_in_set"]))
print("Current routing on ALL rear photos: correct %.1f%%, wrong %.1f%%, uncertain %.1f%%" % tuple(100 * stats[k] / stats["n"] for k in ("routed_correct", "routed_wrong", "uncertain")))
for k, (n, c) in sorted(per.items()):
    print(f"  {k:22s} n={n:3d} routed-correct={c}")
