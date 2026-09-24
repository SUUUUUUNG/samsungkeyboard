"""Held-out error of EVERY landmark (P1..P28) for the learned model and a
mean-position baseline, 15-fold leave-one-out.

Learned model: per landmark, the coarse (distance + offset regression) and fine
stages of train_eval.py, with lighter boosting settings so that 19 landmarks x
15 folds finish in under an hour. atlas/apex (P1/P15-specific) are not used
here; their held-out results live in results_cv_hgb.csv / results_test_hgb.csv.
Baseline: each landmark's mean position in the canonical hand frame over the
other 14 hands.
Derived points 7-14 and 28 are computed from the predicted parents with the
SW's own rules (thirds of tip->base, z = 0; wrist midpoint, z = 1).

Outputs (thumb_model/):
  landmark_errors_learned.csv, landmark_errors_baseline.csv   15 x 28 3D errors (mm)
  landmark_preds_learned.npz, landmark_preds_baseline.npz     predictions, obj frame
"""
import csv
import glob
import os
import sys
import time

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)
import train_eval as te  # noqa: E402
from prep import CACHE_DIR, LANDMARKS, OFFSET_RADIUS, from_canonical  # noqa: E402
from align_landmarks import OUT_DIR as ALIGNED_DIR  # noqa: E402

te.N_RANDOM = 3000          # lighter stage-1 sample per hand
MAX_ITER = 150
SEED = 0
ALL_POINTS = [f"P{i}" for i in range(1, 29)]
DERIVED = {"P7": ("P2", "P16", 1 / 3), "P11": ("P2", "P16", 2 / 3), "P8": ("P3", "P17", 1 / 3), "P12": ("P3", "P17", 2 / 3),
           "P9": ("P4", "P18", 1 / 3), "P13": ("P4", "P18", 2 / 3), "P10": ("P5", "P19", 1 / 3), "P14": ("P5", "P19", 2 / 3)}


def hgb():
    return HistGradientBoostingRegressor(max_iter=MAX_ITER, learning_rate=0.08, max_leaf_nodes=31,
                                         min_samples_leaf=20, l2_regularization=1.0, random_state=SEED)


def fit_offsets(X, Y):
    return [hgb().fit(X, Y[:, c]) for c in range(3)]


def add_derived(pts):
    """pts: dict name -> xyz (obj frame) with the 19 measured points; adds 7-14 and 28."""
    for k, (tip, base, f) in DERIVED.items():
        t, b = pts[tip], pts[base]
        pts[k] = np.array([*(t[:2] + f * (b[:2] - t[:2])), 0.0])
    pts["P28"] = np.array([*(0.5 * (pts["P26"][:2] + pts["P27"][:2])), 1.0])
    return pts


def main():
    subjects = {os.path.basename(p)[:-4]: te.load_subject(p) for p in sorted(glob.glob(os.path.join(CACHE_DIR, "*.npz")))}
    names = list(subjects)
    gt = {n: np.loadtxt(os.path.join(ALIGNED_DIR, f"{n}_aligned.lnd"))[:, 1:] for n in names}
    rng = np.random.default_rng(SEED)
    err = {"learned": {}, "baseline": {}}
    preds = {"learned": {}, "baseline": {}}
    for held in names:
        t0 = time.time()
        m = subjects[held]
        train = [subjects[n] for n in names if n != held]
        pl, pb = {}, {}
        for lm in LANDMARKS:
            X, y = te.stage1_rows(train, lm, rng)
            M = {"dist": hgb().fit(X, y)}
            Xo, _, Yo = te.local_rows(train, lm, OFFSET_RADIUS)
            M["off"] = fit_offsets(Xo, Yo)
            Xf, df, Of = te.local_rows(train, lm, te.FINE_TRAIN_RADIUS)
            M["fdist"] = hgb().fit(Xf, df)
            M["foff"] = fit_offsets(Xf, Of)
            coarse = te.coarse_predict(M, m, lm)
            fine = te.fine_predict(M, m, lm, coarse)
            pl[lm] = np.asarray(from_canonical(fine, m["R"], m["origin"])).reshape(3)
            mean_c = np.mean([s[f"{lm}_gt"] for s in train], axis=0)
            pb[lm] = np.asarray(from_canonical(mean_c, m["R"], m["origin"])).reshape(3)
        add_derived(pl)
        add_derived(pb)
        for tag, p in (("learned", pl), ("baseline", pb)):
            preds[tag][held] = p
            err[tag][held] = {k: float(np.linalg.norm(p[k] - gt[held][int(k[1:]) - 1])) for k in ALL_POINTS}
        e = err["learned"][held]
        print(f"[{held}] {time.time() - t0:.0f}s  learned: P1 {e['P1']:.1f} P15 {e['P15']:.1f} P20 {e['P20']:.1f} "
              f"mean(all28) {np.mean(list(e.values())):.2f} | baseline mean {np.mean(list(err['baseline'][held].values())):.2f}", flush=True)

    for tag in ("learned", "baseline"):
        with open(os.path.join(HERE, f"landmark_errors_{tag}.csv"), "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["name"] + ALL_POINTS)
            for n in names:
                w.writerow([n] + [f"{err[tag][n][k]:.4f}" for k in ALL_POINTS])
        np.savez(os.path.join(HERE, f"landmark_preds_{tag}.npz"),
                 **{f"{n}/{k}": preds[tag][n][k] for n in names for k in ALL_POINTS})
    print("\nmean 3D error per landmark (mm), learned / baseline:")
    for k in ALL_POINTS:
        a = np.mean([err["learned"][n][k] for n in names])
        b = np.mean([err["baseline"][n][k] for n in names])
        print(f"  {k:4s} {a:6.2f} {b:6.2f}")


if __name__ == "__main__":
    main()
