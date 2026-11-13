"""Train and evaluate the rear-view classification head that runs on top of the CLIP image encoder.

Offline/development only (Python is never used at runtime). Inputs are image embeddings of detector crops
(padded to a square, see docs/REAR-VIEW.md) produced with the SAME encoder that ships in
public/models/rear-encoder.onnx (fp32 torch vs shipped 8-bit: cosine 0.9998).

  python scripts/train_rear_head.py <config.json> <out_dir>

config.json:
{
  "arch": "ViT-B-32", "pretrained": "laion2b_s34b_b79k", "modelId": "...", "lambda": 1e-3, "viewpointMin": 0.2,
  "train": "<npz>",   # main training images, labels = class keys           (e.g. SIVD train split)
  "test": "<npz>",    # main held-out test images, labels = class keys      (e.g. SIVD test split)
  "extraGroups": "<json>",  # optional {file: group} for the cross-fitting folds (photographer / uploader)
  "extra": "<npz>",   # optional extra rear photos, labels = class keys OR database vehicle ids (e.g. Wikimedia
                      # Commons). Used with 2-fold CROSS-FITTING: each half is tested by a head trained on the other half.
  "classes": {"<key>": {"vehicleId": "...", "names": ["prompt name", ...], "variant": "..."}}
}
Every npz has emb (N x D float32), labels, files.

Steps
  1. Zero-shot viewpoint scores (CLIP text prompts) -> keep rear / rear-three-quarter images only.
  2. Leakage control: test images that are near-duplicates (cosine >= 0.95) of any training image are dropped.
  3. Head = linear layer initialised from the CLIP text embedding of each class ("a photo of the rear of a
     <name>"), fine-tuned with class-balanced cross-entropy + an L2 pull towards the initialisation; classes without
     training images keep their text-prompt (zero-shot) weights. Temperature calibrated on a 15 % validation split.
  4. Metrics: top-1/top-3, vehicle-level top-1, per-class precision/recall, accuracy vs confidence, unknown-vehicle
     rejection (leave-classes-out), zero-shot baseline. Written to <out_dir>/report.json.
  5. Final head trained on train + all extra images -> <out_dir>/rear-head.json (src/ai/rear-head.js).
"""
import base64
import json
import os
import sys

import numpy as np
import torch

import open_clip

cfg = json.load(open(sys.argv[1], encoding="utf-8"))
out_dir = sys.argv[2]
os.makedirs(out_dir, exist_ok=True)
torch.manual_seed(0)
LAMBDA = cfg.get("lambda", 1e-3)
VIEW_MIN = cfg.get("viewpointMin", 0.2)  # same as REAR.viewpointMin in src/config.js
TRAINED_MIN = 10  # a class with >= this many training images is reported as "trained", 1..9 "few-shot", 0 "zero-shot"

model, _, _ = open_clip.create_model_and_transforms(cfg["arch"], pretrained=cfg["pretrained"])
tokenizer = open_clip.get_tokenizer(cfg["arch"])
model.eval()


def text_emb(prompts):
    with torch.no_grad():
        t = torch.nn.functional.normalize(model.encode_text(tokenizer(prompts)), dim=-1).mean(0)
        return torch.nn.functional.normalize(t, dim=0).numpy()


# ------------------------------------------------------------------------------------------------ viewpoint
VIEW_PROMPTS = {
    "rear": ["a photo of the rear of a car", "a photo of the back of a car", "a rear view photo of a car", "the tail lights and trunk of a car"],
    "rear-side": ["a rear three-quarter view of a car", "a photo of a car seen from behind at an angle"],
    "front": ["a photo of the front of a car", "a front view photo of a car", "the headlights and grille of a car"],
    "front-side": ["a front three-quarter view of a car", "a photo of a car seen from the front at an angle"],
    "side": ["a side view photo of a car", "a photo of a car in profile"],
    "other": ["a photo of a car interior", "a photo of a car engine", "a photo with no car"],
}
VKEYS = list(VIEW_PROMPTS)
V = np.stack([text_emb(p) for p in VIEW_PROMPTS.values()])


def p_rear(emb):
    z = 100 * emb @ V.T
    z = np.exp(z - z.max(1, keepdims=True))
    p = z / z.sum(1, keepdims=True)
    return p[:, [VKEYS.index("rear"), VKEYS.index("rear-side")]].sum(1)


# ------------------------------------------------------------------------------------------------ data
classes = cfg["classes"]
keys = list(classes)
kidx = {k: i for i, k in enumerate(keys)}
C = len(keys)
vid = np.array([classes[k]["vehicleId"] for k in keys])
vids = sorted(set(vid))
vix = {v: i for i, v in enumerate(vids)}
M = np.zeros((C, len(vids)))  # class -> vehicle aggregation
for c in range(C):
    M[c, vix[vid[c]]] = 1
T = np.stack([text_emb([f"a photo of the rear of a {n}, a type of car." for n in classes[k]["names"]] + [f"a photo of a {n}." for n in classes[k]["names"]]) for k in keys])
report = {"model": {"arch": cfg["arch"], "pretrained": cfg["pretrained"]}, "lambda": LAMBDA, "viewpointMin": VIEW_MIN}


def load(path):
    d = np.load(path, allow_pickle=False)
    return d["emb"].astype(np.float32), d["labels"].astype(str), d["files"].astype(str)


Xtr, ltr, _ = load(cfg["train"])
Xte, lte, _ = load(cfg["test"])
ptr, pte = p_rear(Xtr), p_rear(Xte)
report["viewpoint_share"] = {
    k: {"train": int((ltr == k).sum()), "train_rear": int(((ltr == k) & (ptr >= VIEW_MIN)).sum()), "test": int((lte == k).sum()), "test_rear": int(((lte == k) & (pte >= VIEW_MIN)).sum())}
    for k in sorted(set(ltr) | set(lte))
}
m = (ptr >= VIEW_MIN) & np.isin(ltr, keys)
Xtr, ytr = Xtr[m], np.array([kidx[k] for k in ltr[m]])
m = (pte >= VIEW_MIN) & np.isin(lte, keys)
Xte, yte = Xte[m], np.array([kidx[k] for k in lte[m]])
dup = (Xte @ Xtr.T).max(1) >= 0.95
report["leakage"] = {"test_rear_before": int(len(Xte)), "near_duplicates_removed": int(dup.sum())}
Xte, yte = Xte[~dup], yte[~dup]

# extra set: label -> class key when unambiguous (vehicle with a single class); else test-only at vehicle level
key_of_vehicle = {}
for k in keys:
    key_of_vehicle.setdefault(classes[k]["vehicleId"], []).append(k)
if cfg.get("extra"):
    Xex, lex, fex = load(cfg["extra"])
    pex = p_rear(Xex)
    report["extra_viewpoint_rear_accepted"] = float((pex >= VIEW_MIN).mean() * 100)
    vex = np.array([classes[l]["vehicleId"] if l in classes else l for l in lex])
    m = (pex >= VIEW_MIN) & np.isin(vex, vids)
    Xex, lex, vex, fex = Xex[m], lex[m], vex[m], fex[m]
    yex_key = np.array([kidx[l] if l in classes else (kidx[key_of_vehicle[v][0]] if len(key_of_vehicle[v]) == 1 else -1) for l, v in zip(lex, vex)])
    yex_veh = np.array([vix[v] for v in vex])
    # folds by GROUP (e.g. photographer / uploader, "extraGroups": {file: group}) so that a photo series of the same
    # car never lands in both halves; groups alternate within each vehicle
    groups = json.load(open(cfg["extraGroups"], encoding="utf-8")) if cfg.get("extraGroups") else {}
    gex = np.array([groups.get(f, f) or f for f in fex])
    fold = np.zeros(len(Xex), int)
    gfold = {}
    for v in sorted(set(vex)):
        for i, g in enumerate(sorted(set(gex[vex == v]) - set(gfold))):
            gfold[g] = i % 2
    fold = np.array([gfold[g] for g in gex])
    report["extra_groups"] = int(len(set(gex)))
else:
    Xex = np.zeros((0, Xtr.shape[1]), np.float32)
    yex_key = yex_veh = fold = np.zeros(0, int)


# ------------------------------------------------------------------------------------------------ model
def softmax(z):
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def train_head(X, y, lam=LAMBDA, epochs=300, lr=1e-2):
    W0 = torch.tensor(T * 100.0, dtype=torch.float32)  # CLIP logit scale
    W = W0.clone().requires_grad_(True)
    b = torch.zeros(C, requires_grad=True)
    Xt, yt = torch.tensor(X), torch.tensor(y)
    opt = torch.optim.Adam([W, b], lr=lr)
    counts = np.bincount(y, minlength=C).astype(np.float32)
    present = max(1, int((counts > 0).sum()))
    # balanced class weights N / (K * n_c)
    wcls = torch.tensor(np.where(counts > 0, len(y) / (present * np.maximum(counts, 1)), 0.0), dtype=torch.float32)
    for _ in range(epochs):
        opt.zero_grad()
        loss = torch.nn.functional.cross_entropy(Xt @ W.T + b, yt, weight=wcls) + lam * ((W - W0) ** 2).sum() / C
        loss.backward()
        opt.step()
    return W.detach().numpy(), b.detach().numpy()


def calibrate(logits, y):
    best = (1e9, 1.0)
    for t in np.exp(np.linspace(np.log(0.2), np.log(5), 60)):
        p = softmax(logits / t)
        best = min(best, (-np.log(p[np.arange(len(y)), y] + 1e-12).mean(), t))
    return float(best[1])


def kmeans(X, k, iters=25, seed=0):
    """Tiny spherical k-means (cosine) for per-class prototypes."""
    k = min(k, len(X))
    rng = np.random.default_rng(seed)
    Cn = X[rng.choice(len(X), k, replace=False)].copy()
    for _ in range(iters):
        a = (X @ Cn.T).argmax(1)
        for j in range(k):
            if (a == j).any():
                v = X[a == j].mean(0)
                Cn[j] = v / np.linalg.norm(v)
    return Cn


PROTOS_PER_CLASS = 8
OPEN_SET_KEEP = 0.90  # the similarity threshold keeps 90 % of known-class validation images


def prototypes(X, y):
    Xn = X / np.linalg.norm(X, axis=1, keepdims=True)
    return {c: kmeans(Xn[y == c], PROTOS_PER_CLASS) for c in range(C) if (y == c).sum() >= 3}


def similarity(X, pred, protos):
    """Cosine similarity of each image to the nearest prototype of its predicted class (nan if the class has none)."""
    Xn = X / np.linalg.norm(X, axis=1, keepdims=True)
    return np.array([(protos[p] @ x).max() if p in protos else np.nan for x, p in zip(Xn, pred)])


def fit(X, y, seed=0):
    """Train on 85 %, calibrate the temperature and the open-set similarity threshold on a stratified 15 % validation
    split, then refit head and prototypes on everything."""
    rng = np.random.default_rng(seed)
    val = np.zeros(len(y), bool)
    for c in range(C):
        idx = np.where(y == c)[0]
        rng.shuffle(idx)
        if len(idx) >= 4:
            val[idx[: max(1, int(0.15 * len(idx)))]] = True
    W, b = train_head(X[~val], y[~val])
    temp = calibrate(X[val] @ W.T + b, y[val])
    sim = similarity(X[val], (X[val] @ W.T + b).argmax(1), prototypes(X[~val], y[~val]))
    thr = float(np.nanquantile(sim, 1 - OPEN_SET_KEEP))
    W, b = train_head(X, y)
    return W, b, temp, prototypes(X, y), thr


def probs(X, head):
    W, b, t = head[:3]
    return softmax((X @ W.T + b) / t)


def topk(P, y, k):
    return float(np.mean([y[i] in np.argsort(-P[i])[:k] for i in range(len(y))]) * 100) if len(y) else None


def coverage(P, y):
    pred = P.argmax(1)
    return {str(t): {"accepted": float((P.max(1) >= t).mean() * 100), "accuracy_of_accepted": float((pred[P.max(1) >= t] == y[P.max(1) >= t]).mean() * 100) if (P.max(1) >= t).any() else None} for t in (0.35, 0.5, 0.6, 0.7, 0.8, 0.9)}


# ------------------------------------------------------------------------------------------------ evaluation
report["zero_shot_test"] = {"top1": topk(softmax(100 * Xte @ T.T), yte, 1), "top3": topk(softmax(100 * Xte @ T.T), yte, 3), "n": int(len(yte))}
heads = []
ex_P = np.zeros((len(Xex), len(vids)))
ex_tested = np.zeros(len(Xex), bool)
train_counts = []
for f in (0, 1) if len(Xex) else (0,):
    tr_ex = (fold != f) & (yex_key >= 0)
    X = np.concatenate([Xtr, Xex[tr_ex]])
    y = np.concatenate([ytr, yex_key[tr_ex]])
    head = fit(X, y)
    heads.append(head)
    train_counts.append(np.bincount(y, minlength=C))
    te = fold == f
    if len(Xex) and te.any():
        # leakage control inside the extra set as well
        ok = te & ((Xex @ X.T).max(1) < 0.95)
        ex_P[ok] = probs(Xex[ok], head) @ M
        ex_tested |= ok

P = probs(Xte, heads[0])
pred = P.argmax(1)
report["main_test"] = {"top1": topk(P, yte, 1), "top3": topk(P, yte, 3), "n": int(len(yte)), "temperature": heads[0][2], "top1_vehicle": float(np.mean(vid[pred] == vid[yte]) * 100)}
report["main_confidence_coverage"] = coverage(P, yte)
report["per_class"] = {}
for c, k in enumerate(keys):
    tp, npred, ntrue = int(((pred == c) & (yte == c)).sum()), int((pred == c).sum()), int((yte == c).sum())
    report["per_class"][k] = {"n_test": ntrue, "n_train": int(train_counts[0][c]), "precision": (tp / npred * 100) if npred else None, "recall": (tp / ntrue * 100) if ntrue else None}

if ex_tested.any():
    Pe, ye = ex_P[ex_tested], yex_veh[ex_tested]
    n_sivd = np.bincount(ytr, minlength=C)
    kind_v = {}
    for c in range(C):
        k = "sivd-trained" if n_sivd[c] >= TRAINED_MIN else "commons-only"
        if kind_v.get(vid[c]) != "sivd-trained":
            kind_v[vid[c]] = k
    zs = softmax(100 * Xex[ex_tested] @ T.T) @ M
    ex = {}
    for kind in ("sivd-trained", "commons-only"):
        mm = np.array([kind_v[vids[c]] == kind for c in ye])
        if mm.any():
            ex[kind] = {"n": int(mm.sum()), "top1": topk(Pe[mm], ye[mm], 1), "top3": topk(Pe[mm], ye[mm], 3), "zero_shot_top1": topk(zs[mm], ye[mm], 1), "zero_shot_top3": topk(zs[mm], ye[mm], 3)}
    ex["all"] = {"n": int(len(ye)), "top1": topk(Pe, ye, 1), "top3": topk(Pe, ye, 3)}
    report["extra_crossfit"] = ex
    report["extra_confidence_coverage"] = coverage(Pe, ye)
    report["extra_per_vehicle"] = {vids[c]: {"n": int((ye == c).sum()), "top1": topk(Pe[ye == c], ye[ye == c], 1), "top3": topk(Pe[ye == c], ye[ye == c], 3), "kind": kind_v[vids[c]]} for c in sorted(set(ye))}

# unknown-vehicle rejection: remove ~25 % of the SIVD-trained classes from training, measure how often their test
# images still get a confident (wrong) answer
rng = np.random.default_rng(0)
trained = [c for c in range(C) if (ytr == c).sum() >= TRAINED_MIN]
rng.shuffle(trained)
held = set(trained[: max(1, len(trained) // 4)])
mtr = ~np.isin(ytr, list(held))
Wl, bl, tl, pl, thl = fit(Xtr[mtr], ytr[mtr])
known = [c for c in range(C) if c not in held]
Pl = softmax((Xte @ Wl[known].T + bl[known]) / tl)
unk = np.isin(yte, list(held))
report["open_set"] = {"held_out_classes": [keys[c] for c in held]}
for t in (0.5, 0.6, 0.7, 0.8, 0.9):
    report["open_set"]["maxprob>=" + str(t)] = {"unknown_rejected": float((Pl[unk].max(1) < t).mean() * 100), "known_accepted": float((Pl[~unk].max(1) >= t).mean() * 100)}
sl = similarity(Xte, np.array(known)[Pl.argmax(1)], pl)
okl = ~(sl < thl)  # nan (no prototypes) counts as "not rejected"
report["open_set"]["prototype_similarity"] = {"threshold": thl, "unknown_rejected": float((~okl[unk]).mean() * 100), "known_accepted": float(okl[~unk].mean() * 100)}
for t in (0.5, 0.6):
    acc = okl & (Pl.max(1) >= t)
    report["open_set"][f"similarity_and_maxprob>={t}"] = {"unknown_rejected": float((~acc[unk]).mean() * 100), "known_accepted": float(acc[~unk].mean() * 100)}

# ------------------------------------------------------------------------------------------------ final head
Xall = np.concatenate([Xtr, Xex[yex_key >= 0]])
yall = np.concatenate([ytr, yex_key[yex_key >= 0]])
W, b, temp, protos, sim_thr = fit(Xall, yall)
n_all = np.bincount(yall, minlength=C)
f32 = lambda a: base64.b64encode(np.ascontiguousarray(a, dtype="<f4").tobytes()).decode()
kind = lambda n: "trained" if n >= TRAINED_MIN else "few-shot" if n > 0 else "zero-shot"
head = {
    "modelId": cfg["modelId"],
    "encoder": {"arch": cfg["arch"], "pretrained": cfg["pretrained"], "dim": int(W.shape[1])},
    "temperature": temp,
    "viewpoint": {"labels": VKEYS, "rear": ["rear", "rear-side"], "logitScale": 100, "embeddings": f32(V)},
    "classes": [{"key": k, "vehicleId": classes[k]["vehicleId"], "variant": classes[k].get("variant"), "kind": kind(int(n_all[i])), "trainImages": int(n_all[i])} for i, k in enumerate(keys)],
    "weights": f32(W),
    "bias": f32(b),
    # open-set check: cosine to the nearest prototype (k-means of the class's training embeddings) of the predicted
    # class; below openSetMinSimilarity -> "probably a vehicle the model does not know". Classes without prototypes
    # (zero-shot) cannot be checked.
    "openSetMinSimilarity": sim_thr,
    "prototypes": {"counts": [int(len(protos[c])) if c in protos else 0 for c in range(C)], "vectors": f32(np.concatenate([protos[c] for c in range(C) if c in protos]))},
}
json.dump(head, open(os.path.join(out_dir, "rear-head.json"), "w", encoding="utf-8"))
report["final_head"] = {"openSetMinSimilarity": sim_thr, "temperature": temp, "train_images": int(len(yall)), "classes_by_kind": {k: sum(1 for c in head["classes"] if c["kind"] == k) for k in ("trained", "few-shot", "zero-shot")}}
json.dump(report, open(os.path.join(out_dir, "report.json"), "w", encoding="utf-8"), indent=1)
print(json.dumps({k: report[k] for k in report if k not in ("per_class", "viewpoint_share", "extra_per_vehicle")}, indent=1))
