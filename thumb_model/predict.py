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
    ap.add_argument("--mode", choices=("learned", "combined"), default="learned",
                    help="learned: apex/atlas model only; combined: learned + geometric rule (final_pipeline.py)")
    ap.add_argument("--p1", default=None, help="combined mode: P1 source (rule_midpoint | learned_apex); default from final_config.json")
    ap.add_argument("--axis", default=None, help="combined mode: thumb axis definition; default from final_config.json")
    ap.add_argument("--method", default=None, help="combined mode: P15 combination (learned | rule | average | gate | stack); default from final_config.json")
    args = ap.parse_args()
    if args.mode == "combined":
        import json
        cfg_path = os.path.join(HERE, "final_config.json")
        cfg = json.load(open(cfg_path, encoding="utf-8")) if os.path.exists(cfg_path) else {}
        args.p1 = args.p1 or cfg.get("p1_source", "rule_midpoint")
        args.axis = args.axis or cfg.get("axis", "learned")
        args.method = args.method or cfg.get("method", "rule")
        cfg_params = cfg.get("params", {}) if cfg.get("method") == args.method else {}
    with open(args.model, "rb") as fh:
        bundle = pickle.load(fh)
    name = os.path.splitext(os.path.basename(args.obj))[0]
    out_dir = os.path.join(HERE, "predictions")
    os.makedirs(out_dir, exist_ok=True)
    if args.mode == "learned":
        p1, p15, length = predict_file(args.obj, bundle)
        print(f"P1  (thumb tip):  {p1[0]:.3f} {p1[1]:.3f} {p1[2]:.3f}")
        print(f"P15 (thumb base): {p15[0]:.3f} {p15[1]:.3f} {p15[2]:.3f}")
        print(f"thumb length (xy): {length:.2f} mm   "
              f"[P1={bundle['P1_method']}, P15={bundle['P15_method']}, bias={bundle['length_bias_mm']:+.2f}]")
        out = os.path.join(out_dir, f"{name}_pred_landmarks.ply")
        write_ply(out, [p1, p15], [(255, 40, 40), (40, 80, 255)])
    else:
        from final_pipeline import predict_hand
        from align_landmarks import read_obj
        V, F, _ = read_obj(args.obj)
        res = predict_hand(V, F, bundle, p1_source=args.p1, axis=args.axis, method=args.method, params=cfg_params)
        p1, p15, parts = res["P1"], res["P15"], res["parts"]
        print(f"P1  (thumb tip):  {p1[0]:.3f} {p1[1]:.3f} {p1[2]:.3f}   [{args.p1}]")
        print(f"P15 (thumb base): {p15[0]:.3f} {p15[1]:.3f} {p15[2]:.3f}   [axis={args.axis}, method={args.method}]")
        print(f"thumb length (xy): {res['length_mm']:.2f} mm")
        print(f"  rule P15 along-axis t={parts['t_rule']:.1f} (P20 foot t={parts['t_foot']:.1f}), learned P15 t={parts['t_learned']:.1f}")
        out = os.path.join(out_dir, f"{name}_pred_combined.ply")
        write_ply(out, [p1, p15, parts["P20"], parts["P15_rule"], parts["P15_learned"]],
                  [(255, 40, 40), (40, 255, 40), (0, 229, 255), (42, 120, 214), (235, 104, 52)])
        print("  PLY colours: P1 red, P15 green, P20' cyan, rule P15 blue, learned P15 orange")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
