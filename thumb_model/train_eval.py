"""Train, cross-validate and test the thumb-length landmark model (sklearn only).

Per landmark (P1 = thumb tip, P15 = thumb base):
  stage 1  per-vertex distance regression (HistGradientBoosting) over the whole
           mesh -> coarse point (weighted mean of the best vertices + offsets)
  stage 2  refinement, one of
             fine  : the same regression restricted to a 15 mm neighbourhood
             atlas : patches around the true landmark of every training hand
                     registered (similarity ICP) onto the test hand, landmark
                     transferred, weighted median over hands
             apex  : (P1 only) the mesh's thumb apex pulled back by the mean
                     apex-to-P1 offset seen in training
Thumb length = xy distance between the predicted P1 and P15.

Evaluation: leave-one-out over the 12 training hands picks the best
(P1 method, P15 method) pair, the model is refit on all 12 and applied to the
3 held-out test hands.
"""
import argparse
import csv
import glob
import os
import pickle
import sys
import time

import numpy as np
from scipy.spatial import cKDTree
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from align_landmarks import REFERENCE_THUMB_MM, sample_surface  # noqa: E402
from prep import CACHE_DIR, HEAT_SIGMA, LANDMARKS, OFFSET_RADIUS  # noqa: E402

TEST_SUBJECTS = ["20_F_0179G", "20_F_2635G", "20_M_1116G"]
SEED = 0
N_RANDOM = 6000        # random vertices per hand for stage 1
NEAR_RADIUS = 25.0     # ...plus every vertex this close to the landmark
DIST_CLIP = 40.0       # stage-1 target: distance to landmark, clipped (mm)
TOPK = 50
FINE_RADIUS = 15.0
FINE_TRAIN_RADIUS = 25.0
FINE_SIGMA = 2.0
FINE_TOPK = 20
ATLAS_RADIUS = {"P1": 20.0, "P15": 30.0}
ATLAS_MAX_PTS = 1500
ATLAS_SCALE_RANGE = (0.85, 1.15)
APEX_TUBE = 12.0
# blends: the coarse regression and the atlas tend to miss P15 in opposite
# directions, so their average cancels part of the error
BLENDS = {"P1": {"apex+atlas": ["apex", "atlas"], "mean4": ["coarse", "fine", "atlas", "apex"]},
          "P15": {"coarse+atlas": ["coarse", "atlas"], "mean3": ["coarse", "fine", "atlas"],
                  "fine+atlas": ["fine", "atlas"]}}
METHODS = {"P1": ["coarse", "fine", "atlas", "apex"] + list(BLENDS["P1"]),
           "P15": ["coarse", "fine", "atlas"] + list(BLENDS["P15"])}
def model_path(stage1="hgb"):
    return os.path.join(HERE, f"model_{stage1}.pkl")


MODEL_PATH = model_path()


# ----------------------------------------------------------------- data
def load_subject(path):
    z = np.load(path)
    m = {k: z[k] for k in z.files}
    m["name"] = os.path.basename(path)[:-4]
    return m


def surface(m):
    """Dense surface samples in the canonical frame (cached on the dict)."""
    if "surf" not in m:
        m["surf"] = sample_surface(m["Vc"], m["F"], seed=SEED)
    return m["surf"]


def stage1_rows(subjects, lm, rng):
    X, y = [], []
    for m in subjects:
        d = m[f"{lm}_dist"]
        idx = np.unique(np.concatenate([rng.choice(len(d), min(N_RANDOM, len(d)), replace=False),
                                        np.where(d < NEAR_RADIUS)[0]]))
        X.append(m["feats"][idx])
        y.append(np.minimum(d[idx], DIST_CLIP))
    return np.vstack(X), np.concatenate(y)


def local_rows(subjects, lm, radius):
    X, d, off = [], [], []
    for m in subjects:
        idx = np.where(m[f"{lm}_dist"] < radius)[0]
        X.append(m["feats"][idx])
        d.append(m[f"{lm}_dist"][idx])
        off.append(m[f"{lm}_off"][idx])
    return np.vstack(X), np.concatenate(d), np.vstack(off)


# --------------------------------------------------------------- models
def hgb():
    return HistGradientBoostingRegressor(max_iter=500, learning_rate=0.06, max_leaf_nodes=31,
                                         min_samples_leaf=20, l2_regularization=1.0,
                                         random_state=SEED)


def mlp():
    return make_pipeline(StandardScaler(),
                         MLPRegressor(hidden_layer_sizes=(128, 128, 64), max_iter=300,
                                      early_stopping=True, random_state=SEED))


STAGE1 = {"hgb": hgb, "mlp": mlp}


def fit_offsets(X, Y):
    return [hgb().fit(X, Y[:, c]) for c in range(3)]


def predict_offsets(models, X):
    return np.stack([mdl.predict(X) for mdl in models], 1)


def weighted_point(Vc, off, w):
    w = w + 1e-9
    return (w[:, None] * (Vc + off)).sum(0) / w.sum()


# ------------------------------------------------------------ geometry
def thumb_frame(p1, p15):
    d = p1 - p15
    d[2] = 0
    L = np.linalg.norm(d)
    u = d / L
    return L, u, np.array([-u[1], u[0], 0.0])


def thumb_apex(S, p1, p15):
    """Farthest surface sample along the P15->P1 axis inside a tube around it."""
    L, u, lat = thumb_frame(p1, p15)
    rel = S - p15
    t = rel @ u
    perp = np.sqrt((rel @ lat) ** 2 + (rel[:, 2] - p1[2]) ** 2)
    tube = (perp < APEX_TUBE) & (t > L - 15)
    return S[tube][np.argmax(t[tube])], u, lat


def apex_stats(subjects):
    """Mean offset (along thumb axis, lateral, vertical) from apex to true P1."""
    rows = []
    for m in subjects:
        gt1, gt15 = m["P1_gt"], m["P15_gt"]
        apex, u, lat = thumb_apex(surface(m), gt1, gt15)
        o = gt1 - apex
        rows.append([o @ u, o @ lat, o[2]])
    return np.mean(rows, 0)


def apex_predict(m, p1, p15, stats):
    apex, u, lat = thumb_apex(surface(m), p1, p15)
    return apex + stats[0] * u + stats[1] * lat + np.array([0, 0, stats[2]])


def umeyama(P, Q, scale=True):
    pc, qc = P.mean(0), Q.mean(0)
    Pc, Qc = P - pc, Q - qc
    U, S, Vt = np.linalg.svd(Pc.T @ Qc)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    D = np.diag([1.0, 1.0, d])
    R = Vt.T @ D @ U.T
    s = (S * np.diag(D)).sum() / (Pc ** 2).sum() if scale else 1.0
    return s, R, qc - s * R @ pc


def icp_similarity(src, tree, tgt, s, R, t, iters=40):
    for _ in range(iters):
        X = s * src @ R.T + t
        d, j = tree.query(X)
        keep = d <= np.quantile(d, 0.9)
        s2, R2, t2 = umeyama(X[keep], tgt[j[keep]])
        R, t = R2 @ R, s2 * R2 @ t + t2
        s = float(np.clip(s2 * s, *ATLAS_SCALE_RANGE))
    X = s * src @ R.T + t
    d, _ = tree.query(X)
    return s, R, t, np.sqrt(np.mean(d ** 2))


def build_atlas(subjects, lm, rng):
    atlas = []
    for m in subjects:
        gt = m[f"{lm}_gt"]
        S = surface(m)
        pts = S[np.linalg.norm(S - gt, axis=1) < ATLAS_RADIUS[lm]]
        if len(pts) > ATLAS_MAX_PTS:
            pts = pts[rng.choice(len(pts), ATLAS_MAX_PTS, replace=False)]
        atlas.append((pts, gt))
    return atlas


def weighted_median(C, w):
    out = np.zeros(C.shape[1])
    for c in range(C.shape[1]):
        o = np.argsort(C[:, c])
        cw = np.cumsum(w[o])
        out[c] = C[o, c][np.searchsorted(cw, 0.5 * cw[-1])]
    return out


def atlas_predict(m, lm, coarse, atlas):
    S = surface(m)
    loc = S[np.linalg.norm(S - coarse, axis=1) < ATLAS_RADIUS[lm] + 15]
    tree = cKDTree(loc)
    cands, wts = [], []
    for pts, gt in atlas:
        s, R, t, rms = icp_similarity(pts, tree, loc, 1.0, np.eye(3), coarse - gt)
        cands.append(s * R @ gt + t)
        wts.append(1.0 / (rms + 0.2))
    return weighted_median(np.array(cands), np.array(wts))


# --------------------------------------------------------- fit / predict
def fit_landmark(subjects, lm, stage1, rng):
    M = {}
    X, y = stage1_rows(subjects, lm, rng)
    M["dist"] = STAGE1[stage1]().fit(X, y)
    Xo, _, Yo = local_rows(subjects, lm, OFFSET_RADIUS)
    M["off"] = fit_offsets(Xo, Yo)
    Xf, df, Of = local_rows(subjects, lm, FINE_TRAIN_RADIUS)
    M["fdist"] = hgb().fit(Xf, df)
    M["foff"] = fit_offsets(Xf, Of)
    M["atlas"] = build_atlas(subjects, lm, rng)
    if lm == "P1":
        M["apex"] = apex_stats(subjects)
    return M


def fit_all(subjects, stage1="hgb"):
    rng = np.random.default_rng(SEED)
    return {lm: fit_landmark(subjects, lm, stage1, rng) for lm in LANDMARKS}


def coarse_predict(M, m, lm):
    pd = M["dist"].predict(m["feats"])
    top = np.argsort(pd)[:TOPK]
    w = np.exp(-pd[top] ** 2 / (2 * HEAT_SIGMA ** 2))
    return weighted_point(m["Vc"][top], predict_offsets(M["off"], m["feats"][top]), w)


def fine_predict(M, m, lm, coarse):
    cand = np.where(np.linalg.norm(m["Vc"] - coarse, axis=1) < FINE_RADIUS)[0]
    pd = M["fdist"].predict(m["feats"][cand])
    top = cand[np.argsort(pd)[:FINE_TOPK]]
    w = np.exp(-np.sort(pd)[:FINE_TOPK] ** 2 / (2 * FINE_SIGMA ** 2))
    return weighted_point(m["Vc"][top], predict_offsets(M["foff"], m["feats"][top]), w)


def predict_all(models, m):
    """Every (landmark, method) -> point in the canonical frame."""
    out = {}
    for lm in LANDMARKS:
        c = coarse_predict(models[lm], m, lm)
        out[(lm, "coarse")] = c
        out[(lm, "fine")] = fine_predict(models[lm], m, lm, c)
        out[(lm, "atlas")] = atlas_predict(m, lm, c, models[lm]["atlas"])
    out[("P1", "apex")] = apex_predict(m, out[("P1", "fine")], out[("P15", "fine")],
                                       models["P1"]["apex"])
    for lm, blends in BLENDS.items():
        for name, parts in blends.items():
            out[(lm, name)] = np.mean([out[(lm, p)] for p in parts], axis=0)
    return out


def xy_length(p1, p15):
    return float(np.linalg.norm(p1[:2] - p15[:2]))


# ------------------------------------------------------------ evaluation
def evaluate(models, m, preds, split):
    """Rows for every method pair on one hand."""
    ref = REFERENCE_THUMB_MM[m["name"][:-1]]
    gt1, gt15 = m["P1_gt"], m["P15_gt"]
    _, u, _ = thumb_frame(gt1, gt15)
    rows = []
    for m1 in METHODS["P1"]:
        for m15 in METHODS["P15"]:
            p1, p15 = preds[("P1", m1)], preds[("P15", m15)]
            L = xy_length(p1, p15)
            rows.append({"name": m["name"], "split": split, "P1_method": m1, "P15_method": m15,
                         "P1_err_mm": np.linalg.norm(p1 - gt1),
                         "P1_err_axis_mm": (p1 - gt1) @ u,
                         "P15_err_mm": np.linalg.norm(p15 - gt15),
                         "P15_err_axis_mm": (p15 - gt15) @ u,
                         "length_pred_mm": L, "length_ref_mm": ref, "length_err_mm": L - ref})
    return rows


def summarize(rows):
    table = {}
    for r in rows:
        table.setdefault((r["P1_method"], r["P15_method"]), []).append(r)
    out = []
    for (m1, m15), rs in table.items():
        e = np.array([r["length_err_mm"] for r in rs])
        out.append({"P1_method": m1, "P15_method": m15, "n": len(rs),
                    "mae_mm": np.abs(e).mean(), "bias_mm": e.mean(), "max_abs_mm": np.abs(e).max(),
                    "within_0.5": int((np.abs(e) <= 0.5).sum()),
                    "P1_err_mean": np.mean([r["P1_err_mm"] for r in rs]),
                    "P15_err_mean": np.mean([r["P15_err_mm"] for r in rs])})
    return sorted(out, key=lambda s: s["mae_mm"])


def print_summary(title, summ):
    print(f"\n{title}")
    print(f"{'P1':7s} {'P15':7s} {'MAE':>6s} {'bias':>6s} {'max':>6s} {'<=0.5':>5s} {'P1err':>6s} {'P15err':>6s}")
    for s in summ:
        print(f"{s['P1_method']:7s} {s['P15_method']:7s} {s['mae_mm']:6.2f} {s['bias_mm']:+6.2f} "
              f"{s['max_abs_mm']:6.2f} {s['within_0.5']:3d}/{s['n']:<2d} {s['P1_err_mean']:6.2f} "
              f"{s['P15_err_mean']:6.2f}")


def write_csv(path, rows):
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.4f}" if isinstance(v, float) else v) for k, v in r.items()})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage1", choices=list(STAGE1), default="hgb")
    ap.add_argument("--no-cv", action="store_true",
                    help="skip leave-one-out; reuse the method choice saved by an earlier CV run")
    ap.add_argument("--loo-all", action="store_true",
                    help="leave-one-out over all 15 hands (test hands included) instead of the 12")
    args = ap.parse_args()

    subjects = {os.path.basename(p)[:-4]: load_subject(p)
                for p in sorted(glob.glob(os.path.join(CACHE_DIR, "*.npz")))}
    train_names = [n for n in subjects if n not in TEST_SUBJECTS]
    print(f"train {len(train_names)}: {train_names}\ntest  {len(TEST_SUBJECTS)}: {TEST_SUBJECTS}")
    tag = args.stage1 + ("_all15" if args.loo_all else "")
    summary_path = os.path.join(HERE, f"results_cv_{tag}_summary.csv")

    cv_rows, cv_preds = [], {}
    if not args.no_cv:
        loo_names = list(subjects) if args.loo_all else train_names
        for held in loo_names:
            t0 = time.time()
            models = fit_all([subjects[n] for n in loo_names if n != held], args.stage1)
            preds = predict_all(models, subjects[held])
            for (lm, method), p in preds.items():
                cv_preds[f"{held}/{lm}/{method}"] = p
            cv_rows += evaluate(models, subjects[held], preds, "cv")
            best = min((r for r in cv_rows if r["name"] == held), key=lambda r: abs(r["length_err_mm"]))
            print(f"[cv] {held}: {time.time() - t0:.0f}s  best pair {best['P1_method']}/{best['P15_method']} "
                  f"len_err={best['length_err_mm']:+.2f}")
        cv_summary = summarize(cv_rows)
        print_summary(f"Leave-one-out over {len(loo_names)} hands (length error, mm)", cv_summary)
        write_csv(os.path.join(HERE, f"results_cv_{tag}.csv"), cv_rows)
        write_csv(summary_path, cv_summary)
        # held-out predictions (canonical frame) so new blends can be scored without refitting
        np.savez(os.path.join(HERE, f"cv_preds_{tag}.npz"), **cv_preds)
        choice = cv_summary[0]
    elif os.path.exists(summary_path):
        with open(summary_path) as fh:
            choice = next(csv.DictReader(fh))
        choice["bias_mm"] = float(choice["bias_mm"])
    else:
        choice = {"P1_method": "apex", "P15_method": "atlas", "bias_mm": 0.0}
    m1, m15, bias = choice["P1_method"], choice["P15_method"], choice["bias_mm"]
    print(f"\nchosen: P1={m1}  P15={m15}  bias correction={bias:+.2f} mm")

    models = fit_all([subjects[n] for n in train_names], args.stage1)
    test_rows = []
    for n in TEST_SUBJECTS:
        preds = predict_all(models, subjects[n])
        rows = evaluate(models, subjects[n], preds, "test")
        test_rows += rows
        r = next(r for r in rows if r["P1_method"] == m1 and r["P15_method"] == m15)
        print(f"[test] {n}: pred={r['length_pred_mm']:.2f} ref={r['length_ref_mm']:.1f} "
              f"err={r['length_err_mm']:+.2f} (bias-corrected {r['length_err_mm'] - bias:+.2f})  "
              f"P1err={r['P1_err_mm']:.2f} P15err={r['P15_err_mm']:.2f}")
    print_summary("Test hands, all method pairs", summarize(test_rows))
    write_csv(os.path.join(HERE, f"results_test_{args.stage1}.csv"), test_rows)

    with open(model_path(args.stage1), "wb") as fh:
        pickle.dump({"models": models, "P1_method": m1, "P15_method": m15, "length_bias_mm": bias,
                     "stage1": args.stage1, "train_subjects": train_names}, fh)
    print(f"saved {model_path(args.stage1)}")


if __name__ == "__main__":
    main()
