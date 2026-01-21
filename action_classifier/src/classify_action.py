# Inference script: loads saved 2D pose sequences (.npy),
# converts them to NTU-25 format, and classifies actions
# using a pretrained ST-GCN (NTU60), printing top-5 scores.

import numpy as np
import torch

import os
import sys
import importlib.util
from labels.collab_score_map import label_to_collab_score

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(THIS_DIR, ".."))

CANDIDATES = [
    os.path.join(REPO_ROOT, "st-gcn", "net", "st_gcn.py"),
    os.path.join(REPO_ROOT, "st-gcn", "st-gcn", "net", "st_gcn.py"),
]

st_gcn_py = next((p for p in CANDIDATES if os.path.isfile(p)), None)
if st_gcn_py is None:
    raise FileNotFoundError(
        "Could not find st_gcn.py. Tried:\n" + "\n".join(CANDIDATES)
    )

spec = importlib.util.spec_from_file_location("st_gcn_module", st_gcn_py)
st_gcn_module = importlib.util.module_from_spec(spec)
STGCN_ROOT = os.path.dirname(os.path.dirname(st_gcn_py))  # .../st-gcn
if STGCN_ROOT not in sys.path:
    sys.path.insert(0, STGCN_ROOT)

print("Added to sys.path:", STGCN_ROOT)

spec.loader.exec_module(st_gcn_module)

Model = st_gcn_module.Model
print("Loaded Model from:", st_gcn_py)

from labels.ntu60_labels import NTU60_LABELS


import argparse
import os

parser = argparse.ArgumentParser(description="ST-GCN inference on pose numpy files")
parser.add_argument(
    "--pose_npy_dir",
    type=str,
    required=True,
    help="Directory containing pose .npy files"
)
parser.add_argument(
    "--weights",
    type=str,
    required=True,
    help="Path to ST-GCN pretrained weights (.pt)"
)

class Args:
    pose_npy_dir = r"..\v2e-model\st-gcn\pose_npys"
    weights = r"..\v2e-model\st-gcn\models\st_gcn.ntu-xsub.pt"

args = Args()

POSE_NPY_DIR = os.path.abspath(args.pose_npy_dir)
WEIGHTS = os.path.abspath(args.weights)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def h36m17_to_ntu25(pose17):
    T = pose17.shape[0]
    pose25 = np.zeros((T, 25, 2), dtype=np.float32)

    pose25[:, 0]  = pose17[:, 0]
    pose25[:, 1]  = pose17[:, 7]
    pose25[:, 2]  = pose17[:, 8]
    pose25[:, 3]  = pose17[:, 10]

    pose25[:, 4]  = pose17[:, 11]
    pose25[:, 5]  = pose17[:, 12]
    pose25[:, 6]  = pose17[:, 13]
    pose25[:, 7]  = pose17[:, 13]

    pose25[:, 8]  = pose17[:, 14]
    pose25[:, 9]  = pose17[:, 15]
    pose25[:, 10] = pose17[:, 16]
    pose25[:, 11] = pose17[:, 16]

    pose25[:, 12] = pose17[:, 4]
    pose25[:, 13] = pose17[:, 5]
    pose25[:, 14] = pose17[:, 6]
    pose25[:, 15] = pose17[:, 6]

    pose25[:, 16] = pose17[:, 1]
    pose25[:, 17] = pose17[:, 2]
    pose25[:, 18] = pose17[:, 3]
    pose25[:, 19] = pose17[:, 3]

    pose25[:, 20] = pose17[:, 8]
    return pose25


def normalize_pose(p):
    center = p[:, 0:1, :]
    p = p - center
    scale = np.linalg.norm(p[:, 2] - p[:, 0], axis=-1, keepdims=True)
    scale = np.maximum(scale, 1e-6)[:, None, :]
    return p / scale


def to_stgcn_input(pose25):
    T, V, _ = pose25.shape
    conf = np.ones((T, V, 1), dtype=np.float32)
    xyz = np.concatenate([pose25, conf], axis=-1)
    x = np.transpose(xyz, (2, 0, 1))
    x = x[None, ..., None]
    return torch.from_numpy(x)


def load_stgcn():
    model = Model(
        in_channels=3,
        num_class=60,
        dropout=0.5,
        edge_importance_weighting=True,
        graph_args={"layout": "ntu-rgb+d", "strategy": "spatial"},
    )
    ckpt = torch.load(WEIGHTS, map_location="cpu")
    model.load_state_dict(ckpt["model"] if "model" in ckpt else ckpt)
    return model.to(DEVICE).eval()


def main():
    model = load_stgcn()

    for root, _, files in os.walk(POSE_NPY_DIR):
        for f in files:
            if not f.endswith(".npy"):
                continue

            path = os.path.join(root, f)
            pose17 = np.load(path)

            pose25 = h36m17_to_ntu25(pose17)
            pose25 = normalize_pose(pose25)

            x = to_stgcn_input(pose25).to(DEVICE)

            with torch.no_grad():
                logits = model(x)[0]
                probs = torch.softmax(logits, dim=-1)

            topk = torch.topk(probs, k=5)
            labels = [NTU60_LABELS[i] for i in topk.indices.tolist()]
            weights = topk.values.tolist()

            collab_score = 0.0
            for lab, w in zip(labels, weights):
                collab_score += w * label_to_collab_score(lab)

            # Optionally convert to +1/-1 by thresholding:
            collab_binary = 1 if collab_score > 0.25 else -1  # tune threshold
            print(f"Collab proxy score: {collab_score:.3f} -> {collab_binary:+d}")
            print("Top-5:", list(zip(labels, [round(w, 3) for w in weights])))

            print(f"\n{f}")
            for score, idx in zip(topk.values.tolist(), topk.indices.tolist()):
                print(f"  {NTU60_LABELS[idx]:20s}  {score:.3f}")

if __name__ == "__main__":
    main()
