"""
image_utils.py - everything image-related for the facial recognition gate.

Commands (run from the repo root):
  python src/image_utils.py setup --members alice bob carol david
      -> creates data/images/<member>/ folders + image_paths.csv (where each photo goes)
  python src/image_utils.py check
      -> confirms every expected photo is present and readable
  python src/image_utils.py train
      -> builds data/image_features.csv and models/face_model.pkl from the real photos
  python src/image_utils.py predict path/to/face.jpg
      -> recognised member, or ACCESS DENIED

Used by app.py:  from src.image_utils import predict_face
"""

import argparse
import os
import pickle
import sys
from pathlib import Path

import cv2
import matplotlib
import numpy as np
import pandas as pd
from skimage.feature import hog, local_binary_pattern
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score, log_loss
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# ---- Paths (relative to the repo root, so it works from src/, notebooks/ or app.py) ----
ROOT = Path(__file__).resolve().parents[1]
IMAGE_DIR = ROOT / "images"                     # images/member_01/member_01_neutral.jpg
UNKNOWN_DIR = IMAGE_DIR / "unknown"
MANIFEST = ROOT / "data" / "image_paths.csv"
FEATURES_CSV = ROOT / "data" / "image_features.csv"
MODEL_PKL = ROOT / "models" / "face_model.pkl"
FIGURES = ROOT / "outputs" / "figures"
FACE_BOXES = ROOT / "data" / "face_boxes.csv"   # optional manual boxes: filename,x,y,w,h

EXPRESSIONS = ["neutral", "smile", "surprised"]
VALID_EXT = {".jpg", ".jpeg", ".png"}
IMG_SIZE = (128, 128)
SEED = 42
META_COLS = ["member", "expression", "augmentation", "source_image"]


# ---- Setup & checks ------------------------------------------------------------
def setup(members):
    """Create the folder structure and image_paths.csv (the expected path of every photo)."""
    rows = []
    for m in [m.lower().strip() for m in members]:
        (IMAGE_DIR / m).mkdir(parents=True, exist_ok=True)
        for e in EXPRESSIONS:
            rows.append({"member": m, "expression": e, "role": "authorized",
                         "path": str((IMAGE_DIR / m / f"{m}_{e}.jpg").relative_to(ROOT))})
    UNKNOWN_DIR.mkdir(parents=True, exist_ok=True)
    rows.append({"member": "unknown", "expression": "neutral", "role": "unauthorized",
                 "path": str((UNKNOWN_DIR / "non_member_1.jpg").relative_to(ROOT))})
    pd.DataFrame(rows).to_csv(MANIFEST, index=False)
    print(f"Created folders under {IMAGE_DIR}/ and {MANIFEST} ({len(rows)} expected images)")


def check():
    """Return True if every photo listed in image_paths.csv exists and can be opened."""
    if not MANIFEST.exists():
        sys.exit("Run `python src/image_utils.py setup --members ...` first.")
    ok = True
    for _, r in pd.read_csv(MANIFEST).iterrows():
        p = ROOT / r["path"]
        found = p if p.exists() else next((p.with_suffix(s) for s in (".jpeg", ".png")
                                           if p.with_suffix(s).exists()), None)
        status = "OK" if found is not None and cv2.imread(str(found)) is not None else "MISSING"
        ok &= status == "OK"
        print(f"  [{status:7}] {r['path']}")
    print("All images present." if ok else "Some images are missing - add them before training.")
    return ok


# ---- Loading & face cropping -----------------------------------------------------
def _load_cascades():
    """Two Haar detectors; the second (alt2) catches faces the default one misses."""
    try:
        return [cv2.CascadeClassifier(os.path.join(cv2.data.haarcascades, f))
                for f in ("haarcascade_frontalface_default.xml", "haarcascade_frontalface_alt2.xml")]
    except AttributeError:   # OpenCV 5 dropped it: pip install "opencv-python<5"
        print("[WARN] Face detector unavailable - using upper-centre crop.")
        return []


_CASCADES = _load_cascades()


def _skin_fraction(img, box):
    """Share of skin-coloured pixels (YCrCb rule) in the centre of a box; rejects
    false detections on clothing patterns or backgrounds."""
    x, y, w, h = box
    roi = img[y + h // 4:y + 3 * h // 4, x + w // 4:x + 3 * w // 4]
    ycc = cv2.cvtColor(roi, cv2.COLOR_RGB2YCrCb)
    mask = cv2.inRange(ycc, (0, 133, 77), (255, 180, 135))
    return mask.mean() / 255


def detect_face(img):
    """Largest plausible face box (x, y, w, h) or None. Tries strict settings first, then looser ones."""
    gray = cv2.equalizeHist(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY))
    min_side = max(40, min(gray.shape) // 10)
    for scale, neighbours in ((1.1, 5), (1.05, 3)):
        for cascade in _CASCADES:
            faces = cascade.detectMultiScale(gray, scaleFactor=scale, minNeighbors=neighbours,
                                             minSize=(min_side, min_side))
            faces = [f for f in faces if _skin_fraction(img, f) > 0.3]
            if faces:
                return max(faces, key=lambda f: f[2] * f[3])
    return None


def load_manual_boxes():
    """Manual face boxes for photos the detector gets wrong (documented, not hidden)."""
    if not FACE_BOXES.exists():
        return {}
    df = pd.read_csv(FACE_BOXES)
    return {r.filename: (int(r.x), int(r.y), int(r.w), int(r.h)) for r in df.itertuples()}


def load_records(root=IMAGE_DIR):
    boxes = load_manual_boxes()
    recs = []
    for p in sorted(Path(root).rglob("*")):
        if p.suffix.lower() not in VALID_EXT:
            continue
        img = cv2.imread(str(p))
        if img is None:
            continue
        stem = p.stem.lower()
        recs.append({"member": p.parent.name.lower(), "path": str(p.relative_to(ROOT)),
                     "expression": next((e for e in EXPRESSIONS if stem.endswith(e)), "other"),
                     "image": cv2.cvtColor(img, cv2.COLOR_BGR2RGB),
                     "box": boxes.get(p.name)})
    return recs


def crop_face(img, box=None):
    box = box if box is not None else detect_face(img)
    if box is not None:
        x, y, w, h = box
        pad = int(0.15 * w)
        img = img[max(y - pad, 0):y + h + pad, max(x - pad, 0):x + w + pad]
    else:   # fallback: in portraits the face is usually in the upper-centre
        h, w = img.shape[:2]; s = int(min(h, w) * 0.5)
        y0 = int(h * 0.1); x0 = (w - s) // 2
        img = img[y0:y0 + s, x0:x0 + s]
    return cv2.resize(img, IMG_SIZE)


# ---- Augmentation ----------------------------------------------------------------
def augment(img):
    rng = np.random.default_rng(SEED)
    h, w = img.shape[:2]
    rot = lambda a: cv2.warpAffine(img, cv2.getRotationMatrix2D((w / 2, h / 2), a, 1),
                                   (w, h), borderMode=cv2.BORDER_REFLECT)
    gray = cv2.cvtColor(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY), cv2.COLOR_GRAY2RGB)
    return {
        "original": img,
        "rotate_+15": rot(15),
        "rotate_-15": rot(-15),
        "hflip": cv2.flip(img, 1),
        "grayscale": gray,
        "brightness_1.3": np.clip(img * 1.3, 0, 255).astype(np.uint8),
        "gaussian_noise": np.clip(img + rng.normal(0, 12, img.shape), 0, 255).astype(np.uint8),
    }


# ---- Features --------------------------------------------------------------------
def extract_features(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    hist = []
    for c, rng in enumerate([(0, 180), (0, 256), (0, 256)]):
        h = cv2.calcHist([hsv], [c], None, [16], rng).flatten()
        hist.append(h / (h.sum() + 1e-8))
    g = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    hog_f = hog(cv2.resize(g, (64, 64)), orientations=9, pixels_per_cell=(16, 16),
                cells_per_block=(2, 2), block_norm="L2-Hys")
    lbp = local_binary_pattern(g, 8, 1, method="uniform")
    lbp_h, _ = np.histogram(lbp.ravel(), bins=10, range=(0, 10))
    return np.concatenate(hist + [hog_f, lbp_h / (lbp_h.sum() + 1e-8)])


def feature_names():
    n = len(extract_features(np.zeros((*IMG_SIZE, 3), np.uint8)))
    return [f"hist_{i}" for i in range(48)] + [f"hog_{i}" for i in range(n - 58)] + \
           [f"lbp_{i}" for i in range(10)]


def build_features(records):
    names = feature_names()
    rows = []
    for r in records:
        for aug, im in augment(crop_face(r["image"], r.get("box"))).items():
            rows.append({"member": r["member"], "expression": r["expression"], "augmentation": aug,
                         "source_image": os.path.basename(r["path"]),
                         **dict(zip(names, extract_features(im)))})
    return pd.DataFrame(rows)


# ---- Model -----------------------------------------------------------------------
def _models(n, f):
    return {"LogisticRegression": Pipeline([("s", StandardScaler()),
                                            ("p", PCA(min(30, n - 1, f), random_state=SEED)),
                                            ("c", LogisticRegression(max_iter=2000))]),
            "RandomForest": Pipeline([("s", StandardScaler()),
                                      ("c", RandomForestClassifier(300, random_state=SEED))])}


def _nn_dist(train_X, X):
    sc = StandardScaler().fit(train_X)
    d, _ = NearestNeighbors(n_neighbors=1).fit(sc.transform(train_X)).kneighbors(sc.transform(X))
    return d[:, 0] / np.sqrt(np.asarray(train_X).shape[1])


def train():
    recs = load_records()
    members = sorted({r["member"] for r in recs if r["member"] != "unknown"})
    if len(members) < 2:
        sys.exit("Need photos of at least 2 members in data/images/. Run `check` first.")

    FEATURES_CSV.parent.mkdir(parents=True, exist_ok=True)
    df_all = build_features(recs)
    df_all.to_csv(FEATURES_CSV, index=False)
    print(f"Saved {FEATURES_CSV}: {df_all.shape[0]} rows x {df_all.shape[1]} cols")

    df = df_all[df_all.member != "unknown"].reset_index(drop=True)
    unknown = df_all[df_all.member == "unknown"]
    fc = [c for c in df.columns if c not in META_COLS]

    # Leave-one-expression-out CV: test photos are never seen (even augmented) in training
    scores, best = {}, None
    for name in ("LogisticRegression", "RandomForest"):
        folds, confs, dists, yt, yp = [], [], [], [], []
        for held in EXPRESSIONS:
            tr, te = df[df.expression != held], df[df.expression == held]
            if te.empty:
                continue
            m = _models(len(tr), len(fc))[name].fit(tr[fc], tr.member)
            pr = m.predict_proba(te[fc]); pred = m.classes_[pr.argmax(1)]; ok = pred == te.member.values
            folds.append([accuracy_score(te.member, pred),
                          f1_score(te.member, pred, average="macro", zero_division=0),
                          log_loss(te.member, pr, labels=m.classes_)])
            confs += list(pr.max(1)[ok]); dists += list(_nn_dist(tr[fc], te[fc])[ok])
            yt += list(te.member); yp += list(pred)
        acc, f1, ll = np.mean(folds, axis=0)
        scores[name] = dict(accuracy=acc, f1=f1, log_loss=ll, confs=confs, dists=dists, yt=yt, yp=yp)
        print(f"{name:20s} accuracy {acc:.3f} | F1 {f1:.3f} | log loss {ll:.3f}")
        if best is None or (f1, -ll) > (scores[best]["f1"], -scores[best]["log_loss"]):
            best = name
    s = scores[best]
    print(f"\nSelected: {best}\n" + classification_report(s["yt"], s["yp"], zero_division=0))

    final = _models(len(df), len(fc))[best].fit(df[fc], df.member)
    conf_t = float(max(0.5, np.percentile(s["confs"], 10))) if s["confs"] else 0.5
    dist_t = float(np.percentile(s["dists"], 95) * 1.1) if s["dists"] else float("inf")
    if len(unknown):
        cu, du = final.predict_proba(unknown[fc]).max(1), _nn_dist(df[fc], unknown[fc])
        print(f"Non-member samples rejected: {((cu < conf_t) | (du > dist_t)).mean():.0%} (n={len(unknown)})")

    MODEL_PKL.parent.mkdir(parents=True, exist_ok=True)
    with open(MODEL_PKL, "wb") as f:
        pickle.dump({"model": final, "model_name": best, "classes": list(final.classes_),
                     "feature_cols": fc, "threshold": conf_t, "dist_threshold": dist_t,
                     "train_X": df[fc].values}, f)
    print(f"Saved {MODEL_PKL}")

    import matplotlib.pyplot as plt
    from sklearn.metrics import ConfusionMatrixDisplay
    FIGURES.mkdir(exist_ok=True)
    ConfusionMatrixDisplay.from_predictions(s["yt"], s["yp"], cmap="Blues")
    plt.title(f"Face recognition - {best} (leave-one-expression-out)")
    plt.savefig(FIGURES / "face_confusion_matrix.png", dpi=120, bbox_inches="tight")
    plt.close()
    return {"best": best, "scores": {k: {m: v[m] for m in ("accuracy", "f1", "log_loss")}
                                     for k, v in scores.items()},
            "threshold": conf_t, "dist_threshold": dist_t}


def predict_face(image_path, model_path=MODEL_PKL):
    """Returns (member_name or None, confidence). None = access denied."""
    with open(model_path, "rb") as f:
        b = pickle.load(f)
    img = cv2.imread(str(image_path))
    if img is None:
        raise FileNotFoundError(image_path)
    x = pd.DataFrame([extract_features(crop_face(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)))],
                     columns=b["feature_cols"])
    pr = b["model"].predict_proba(x)[0]; i = int(pr.argmax()); conf = float(pr[i])
    dist = float(_nn_dist(pd.DataFrame(b["train_X"], columns=b["feature_cols"]), x)[0])
    if conf < b["threshold"] or dist > b["dist_threshold"]:
        return None, conf
    return b["classes"][i], conf


if __name__ == "__main__":
    matplotlib.use("Agg")
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sp = sub.add_parser("setup"); sp.add_argument("--members", nargs="+", required=True)
    sub.add_parser("check"); sub.add_parser("train")
    pp = sub.add_parser("predict"); pp.add_argument("image")
    a = ap.parse_args()
    if a.cmd == "setup":
        setup(a.members)
    elif a.cmd == "check":
        check()
    elif a.cmd == "train":
        train()
    else:
        m, c = predict_face(a.image)
        print(f"Recognised: {m} ({c:.2f})" if m else f"ACCESS DENIED ({c:.2f})")
