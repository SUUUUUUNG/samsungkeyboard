"""Predict P1 / P15 and the xy thumb length for a hand mesh.

    python thumb_model/predict.py path/to/hand.obj [--model thumb_model/model.pkl]

Prints both landmarks in the obj's own coordinate frame and the thumb length,
and writes thumb_model/predictions/<name>_pred_landmarks.ply for viewing.
"""
import argparse
import os
import pickle
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from prep import from_canonical, prepare_mesh  # noqa: E402
from train_eval import MODEL_PATH, predict_all, xy_length  # noqa: E402


def write_ply(path, points, colors):
    with open(path, "w") as fh:
        fh.write("ply\nformat ascii 1.0\n")
        fh.write(f"element vertex {len(points)}\n")
        fh.write("property float x\nproperty float y\nproperty float z\n")
        fh.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        fh.write("end_header\n")
        for p, (r, g, b) in zip(points, colors):
            fh.write(f"{p[0]:.6f} {p[1]:.6f} {p[2]:.6f} {r} {g} {b}\n")


def predict_file(obj_path, bundle):
    m = prepare_mesh(obj_path)
    m["name"] = os.path.splitext(os.path.basename(obj_path))[0]
    preds = predict_all(bundle["models"], m)
    p1 = preds[("P1", bundle["P1_method"])]
    p15 = preds[("P15", bundle["P15_method"])]
    length = xy_length(p1, p15) - bundle["length_bias_mm"]
    return (from_canonical(p1, m["R"], m["origin"]),
            from_canonical(p15, m["R"], m["origin"]), length)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("obj")
    ap.add_argument("--model", default=MODEL_PATH)
    args = ap.parse_args()
    with open(args.model, "rb") as fh:
        bundle = pickle.load(fh)
    p1, p15, length = predict_file(args.obj, bundle)
    print(f"P1  (thumb tip):  {p1[0]:.3f} {p1[1]:.3f} {p1[2]:.3f}")
    print(f"P15 (thumb base): {p15[0]:.3f} {p15[1]:.3f} {p15[2]:.3f}")
    print(f"thumb length (xy): {length:.2f} mm   "
          f"[P1={bundle['P1_method']}, P15={bundle['P15_method']}, bias={bundle['length_bias_mm']:+.2f}]")
    out_dir = os.path.join(HERE, "predictions")
    os.makedirs(out_dir, exist_ok=True)
    name = os.path.splitext(os.path.basename(args.obj))[0]
    out = os.path.join(out_dir, f"{name}_pred_landmarks.ply")
    write_ply(out, [p1, p15], [(255, 40, 40), (40, 80, 255)])
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
