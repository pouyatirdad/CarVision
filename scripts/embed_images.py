"""Development tool: embed every image of a class-folder zip (<split>/<class>/<file>; label = parent folder) with the
CLIP image encoder, after cropping the main vehicle with the app's YOLOX detector (same as the browser pipeline).

usage: python scripts/embed_images.py <zip> <out.npz> public/models/vehicle-detector.onnx ViT-B-32 datacomp_xl_s13b_b90k --pad
       [--files list.txt] [--limit N]
  --pad    centre the crop on a black square (what the browser does, MODELS.rear.fit = "pad")
  --files  only embed these zip members; --limit N  deterministic subset of N images
Writes emb (float32, not normalised), labels, files, found (1 = a vehicle was detected).
"""
import io
import sys
import zipfile

import numpy as np
import onnxruntime as ort
import torch
from PIL import Image

import open_clip

zpath, out, detp, arch, pretrained = sys.argv[1:6]
model, _, preprocess = open_clip.create_model_and_transforms(arch, pretrained=pretrained)
tok = open_clip.get_tokenizer(arch)
model.eval()
torch.set_num_threads(8)
det = ort.InferenceSession(detp, providers=["CPUExecutionProvider"])


def crop(img):
    s = 416
    w, h = img.size
    r = min(s / w, s / h)
    rs = img.resize((max(1, int(w * r)), max(1, int(h * r))), Image.BILINEAR)
    canvas = np.full((s, s, 3), 114, np.float32)
    a = np.asarray(rs, np.float32)[:, :, ::-1]
    canvas[: a.shape[0], : a.shape[1]] = a
    o = det.run(None, {det.get_inputs()[0].name: canvas.transpose(2, 0, 1)[None]})[0][0]
    grids, strides = [], []
    for st in (8, 16, 32):
        n = s // st
        yv, xv = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
        grids.append(np.stack([xv, yv], -1).reshape(-1, 2))
        strides.append(np.full((n * n, 1), st))
    g, st = np.concatenate(grids), np.concatenate(strides)
    xy = (o[:, :2] + g) * st
    wh = np.exp(o[:, 2:4]) * st
    sc = o[:, 4:5] * o[:, 5:]
    best = None
    for c in (2, 5, 7):
        for i in np.where(sc[:, c] > 0.35)[0]:
            x1, y1 = (xy[i] - wh[i] / 2) / r
            x2, y2 = (xy[i] + wh[i] / 2) / r
            k = sc[i, c] * (x2 - x1) * (y2 - y1)
            if best is None or k > best[0]:
                best = (k, (max(0, x1), max(0, y1), min(w, x2), min(h, y2)))
    return (img.crop(tuple(int(v) for v in best[1])), 1) if best else (img, 0)


z = zipfile.ZipFile(zpath)
names = [n for n in z.namelist() if n.lower().endswith((".jpg", ".jpeg", ".png", ".webp"))]
if "--files" in sys.argv:  # only these zip members (one per line)
    wanted = set(open(sys.argv[sys.argv.index("--files") + 1], encoding="utf-8").read().split("\n"))
    names = [n for n in names if n in wanted]
if "--limit" in sys.argv:  # deterministic subset for quick comparisons
    lim = int(sys.argv[sys.argv.index("--limit") + 1])
    # evenly spaced over the whole sorted list (a stride + truncation would drop the last classes alphabetically)
    names = sorted(names)
    names = [names[int(i)] for i in np.linspace(0, len(names) - 1, min(lim, len(names)))]
embs, labels, files, found = [], [], [], []
batch, meta = [], []


def flush():
    with torch.no_grad():
        f = torch.nn.functional.normalize(model.encode_image(torch.stack(batch)), dim=-1).numpy()
    embs.extend(f)
    for m in meta:
        files.append(m[0])
        labels.append(m[1])
        found.append(m[2])
    batch.clear()
    meta.clear()


for i, n in enumerate(names):
    try:
        img = Image.open(io.BytesIO(z.read(n))).convert("RGB")
    except Exception:
        continue
    c, ok = crop(img)
    if "--pad" in sys.argv:  # keep aspect ratio: centre the crop on a black square (what the browser does)
        side = max(c.size)
        sq = Image.new("RGB", (side, side), (0, 0, 0))
        sq.paste(c, ((side - c.size[0]) // 2, (side - c.size[1]) // 2))
        c = sq
    batch.append(preprocess(c))
    meta.append((n, n.replace("\\", "/").split("/")[-2], ok))
    if len(batch) == 64:
        flush()
    if i % 1000 == 0:
        print(i, len(names), flush=True)
if batch:
    flush()
np.savez(out, emb=np.stack(embs).astype(np.float32), labels=np.array(labels), files=np.array(files), found=np.array(found))
print("done", len(embs))
