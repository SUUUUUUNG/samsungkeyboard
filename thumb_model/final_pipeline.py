"""Final mesh-only pipeline: P1, P15 and the xy thumb length from an obj mesh,
combining the learned model (train_eval.py) with the geometric rule
(P20 foot + 85-95 degree contour window).

  python thumb_model/final_pipeline.py --eval      15-hand held-out evaluation of every
                                                   P1 / axis / combination variant
  from final_pipeline import predict_hand          used by predict.py --mode combined

Rule side (no learning): rule_pipeline.run -> outline, P20', midline;
P1_rule = midpoint of the midline apex and the max-y point of the tip;
axis through P1_rule; lower band contour along it; P15_rule = highest contour
point among stations reached from P20' at 85-95 deg (foot +/- ~1.1 mm).
Learned side: coarse -> fine -> atlas P15 and apex P1 from model_hgb.pkl.
"""
import argparse
import csv
import glob
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)
import rule_pipeline as rp  # noqa: E402
from align_landmarks import OBJ_DIR, OUT_DIR as ALIGNED_DIR, REFERENCE_THUMB_MM, read_obj, sample_surface  # noqa: E402
from p15_foot_window_views import foot_window_estimate  # noqa: E402
from prep import CACHE_DIR, from_canonical, prepare_mesh  # noqa: E402
from thumb_views import lower_contour  # noqa: E402

TIP_RADIUS = 25.0            # thumb-tip region used for the apex / max-y search
AXES = ("learned", "midline_parallel", "midline_foot", "midline")
P1_SOURCES = ("rule_midpoint", "learned_apex")
COMBOS = ("learned", "rule", "average", "gate", "stack")


def unit(v):
    v = np.asarray(v, float)
    return v / np.linalg.norm(v)


# ------------------------------------------------------------------ rule side
def rule_parts(V, F, seed=0):
    """Everything the geometric side gives without learning (obj frame)."""
    r = rp.run(V, F, seed=seed)
    S = np.vstack([V, sample_surface(V, F, seed=seed)])
    d_mid = unit(r["P1"][:2] - r["P15_B"][:2])              # midline direction, towards the tip
    near = S[np.linalg.norm(S[:, :2] - r["P1"][:2], axis=1) < TIP_RADIUS]
    apex = near[np.argmax(near[:, :2] @ d_mid)]
    work = r["frame"].to_work(near)                          # hand frame: wrist centre -> middle tip = +y
    maxy = near[np.argmax(work[:, 1])]
    p1_rule = 0.5 * (apex + maxy)
    return {"S": S, "P20": r["P20"], "tip_outline": r["P1"], "apex": apex, "maxy": maxy, "P1_rule": p1_rule,
            "d_mid": d_mid, "P15_B": r["P15_B"], "P15_rule_B": r["P15"], "frame": r["frame"]}


def axis_direction(parts, axis, p1, p15_learned=None):
    """Unit direction from P1 towards the base for the chosen axis definition."""
    if axis == "learned":
        return unit(p15_learned[:2] - p1[:2])
    if axis == "midline_parallel":
        return -parts["d_mid"]
    if axis == "midline_foot":                                # P1_rule -> foot of P20 on the midline axis
        return unit(parts["P15_rule_B"][:2] - p1[:2])
    if axis == "midline":                                     # the midline itself (through the apex)
        return -parts["d_mid"]
    raise ValueError(axis)


def rule_p15(parts, p1, u_base):
    """P15 by the foot-window rule along the axis (p1, u_base). Returns point, t, foot t, |w20|."""
    S, p20 = parts["S"], parts["P20"]
    far = np.array([*(p1[:2] + 60.0 * u_base), p1[2]])
    ts, zs, u = lower_contour(S, p1, far)
    t_est, z_est, t20, _ = foot_window_estimate(ts, zs, u, p1, p20)
    lat = np.array([-u[1], u[0]])
    w20 = abs((p20[:2] - p1[:2]) @ lat)
    p15 = np.array([*(p1[:2] + t_est * u), z_est])
    return p15, t_est, t20, w20, (ts, zs)


def xy_length(p1, p15):
    return float(np.linalg.norm(np.asarray(p1)[:2] - np.asarray(p15)[:2]))


# ------------------------------------------------------------- learned side
def learned_parts_from_model(V, F, bundle):
    import train_eval as te
    m = prepare_mesh_from_arrays(V, F)
    preds = te.predict_all(bundle["models"], m)
    out = {k: from_canonical(preds[(lm, k)], m["R"], m["origin"]).reshape(3) for lm in ("P1", "P15") for k in ("coarse", "fine", "atlas", "apex") if (lm, k) in preds}
    return {"P1": out.get("apex", out.get("fine")), "P15": out.get("atlas")}, {f"{lm}/{k}": from_canonical(v, m["R"], m["origin"]).reshape(3) for (lm, k), v in preds.items()}


def prepare_mesh_from_arrays(V, F):
    """prep.prepare_mesh works on a path; rebuild the same dict from arrays."""
    import prep
    R, origin = prep.canonical_frame(V)
    Vc = prep.to_canonical(V, R, origin)
    N = prep.vertex_normals(V, F) @ R.T
    feats, names = prep.vertex_features(Vc, N)
    return {"V": V, "F": F, "Vc": Vc, "N": N, "feats": feats, "feat_names": names, "R": R, "origin": origin}


def learned_parts_from_npz(name, z):
    """Held-out learned predictions (canonical frame in the npz) mapped to the obj frame."""
    c = np.load(os.path.join(CACHE_DIR, f"{name}.npz"))
    R, origin = c["R"], c["origin"]
    get = lambda lm, k: from_canonical(z[f"{name}/{lm}/{k}"], R, origin).reshape(3)
    return {"P1": get("P1", "apex"), "P15": get("P15", "atlas")}


# ---------------------------------------------------------------- combining
def combine(t_learned, t_rule, t_foot, w20, method, params):
    """Along-axis position of P15 (mm from P1) for a combination method."""
    if method == "learned":
        return t_learned
    if method == "rule":
        return t_rule
    if method == "average":
        w = params.get("w", 0.5)
        return w * t_learned + (1 - w) * t_rule
    if method == "gate":                                      # learned unless it strays from the rule
        x = params.get("x", 3.0)
        return t_learned if abs(t_learned - t_rule) <= x else t_rule
    if method == "stack":
        a, b, c, d, e = params["coef"]
        return a * t_learned + b * t_rule + c * t_foot + d * w20 + e
    raise ValueError(method)


GATE_X_FIXED = None   # set by --gate-x: use this threshold instead of the nested-LOO choice


def fit_params(method, rows):
    """rows: list of (t_learned, t_rule, t_foot, w20, t_true) from the training folds."""
    A = np.array(rows)
    if method == "average":
        ws = np.linspace(0, 1, 21)
        errs = [np.abs(w * A[:, 0] + (1 - w) * A[:, 1] - A[:, 4]).mean() for w in ws]
        return {"w": float(ws[int(np.argmin(errs))])}
    if method == "gate":
        if GATE_X_FIXED is not None:
            return {"x": float(GATE_X_FIXED)}
        xs = np.arange(0.5, 8.01, 0.5)
        errs = [np.abs(np.where(np.abs(A[:, 0] - A[:, 1]) <= x, A[:, 0], A[:, 1]) - A[:, 4]).mean() for x in xs]
        return {"x": float(xs[int(np.argmin(errs))])}
    if method == "stack":
        X = np.c_[A[:, :4], np.ones(len(A))]
        coef, *_ = np.linalg.lstsq(X, A[:, 4], rcond=None)
        return {"coef": coef}
    return {}


# ------------------------------------------------------------- prediction
def predict_hand(V, F, bundle, p1_source="rule_midpoint", axis="midline_foot", method="rule", params=None):
    """Mesh-only prediction. Returns dict with P1, P15, length_mm and the parts."""
    parts = rule_parts(V, F)
    learned, all_learned = learned_parts_from_model(V, F, bundle)
    p1 = parts["P1_rule"] if p1_source == "rule_midpoint" else learned["P1"]
    u = axis_direction(parts, axis, p1, learned["P15"])
    p15_rule, t_rule, t_foot, w20, _ = rule_p15(parts, p1, u)
    t_learned = float((learned["P15"][:2] - p1[:2]) @ u)
    t = combine(t_learned, t_rule, t_foot, w20, method, params or {})
    p15 = np.array([*(p1[:2] + t * u), p15_rule[2]])
    return {"P1": p1, "P15": p15, "length_mm": xy_length(p1, p15),
            "parts": {"P20": parts["P20"], "P15_rule": p15_rule, "P15_learned": learned["P15"], "P1_learned": learned["P1"],
                      "t_learned": t_learned, "t_rule": t_rule, "t_foot": t_foot, "axis": axis, "method": method}}


# ------------------------------------------------------------- evaluation
def evaluate():
    npz_path = os.path.join(HERE, "cv_preds_hgb_all15.npz")
    if not os.path.exists(npz_path):
        sys.exit("run `python thumb_model/train_eval.py --loo-all` first (cv_preds_hgb_all15.npz missing)")
    z = np.load(npz_path)
    hands = []
    for lnd in sorted(glob.glob(os.path.join(ALIGNED_DIR, "*_aligned.lnd"))):
        name = os.path.basename(lnd)[:-12]
        A = np.loadtxt(lnd)[:, 1:]
        V, F, _ = read_obj(os.path.join(OBJ_DIR, name + ".obj"))
        parts = rule_parts(V, F)
        learned = learned_parts_from_npz(name, z)
        hands.append((name, A, parts, learned))
        print(f"[{name}] parts ready", flush=True)

    ref = {n: REFERENCE_THUMB_MM[n[:-1]] for n, *_ in hands}
    rows = []
    # per hand, per P1 source, per axis: the rule / learned along-axis quantities
    table = {}
    for name, A, parts, learned in hands:
        g1, g15 = A[0], A[14]
        u_true = unit(g1[:2] - g15[:2])
        for p1s in P1_SOURCES:
            p1 = parts["P1_rule"] if p1s == "rule_midpoint" else learned["P1"]
            for ax in AXES:
                p1_ax = parts["apex"] if ax == "midline" else p1
                u = axis_direction(parts, ax, p1_ax, learned["P15"])
                p15_rule, t_rule, t_foot, w20, _ = rule_p15(parts, p1_ax, u)
                t_learned = float((learned["P15"][:2] - p1_ax[:2]) @ u)
                t_true = float((g15[:2] - p1_ax[:2]) @ u)
                angle = float(np.degrees(np.arctan2(u_true[0] * (-u)[1] - u_true[1] * (-u)[0], u_true @ (-u))))
                table[(name, p1s, ax)] = dict(p1=p1_ax, u=u, t_learned=t_learned, t_rule=t_rule, t_foot=t_foot, w20=w20, t_true=t_true,
                                              angle=angle, p1_err=float(np.linalg.norm(p1_ax[:2] - g1[:2])),
                                              p15_rule_err=float(np.linalg.norm(p15_rule[:2] - g15[:2])),
                                              z_rule=p15_rule[2])
    names = [h[0] for h in hands]
    # combination methods with nested LOO for their parameters
    for p1s in P1_SOURCES:
        for ax in AXES:
            for method in COMBOS:
                for i, name in enumerate(names):
                    train = [(table[(n, p1s, ax)]["t_learned"], table[(n, p1s, ax)]["t_rule"], table[(n, p1s, ax)]["t_foot"],
                              table[(n, p1s, ax)]["w20"], table[(n, p1s, ax)]["t_true"]) for n in names if n != name]
                    params = fit_params(method, train)
                    e = table[(name, p1s, ax)]
                    t = combine(e["t_learned"], e["t_rule"], e["t_foot"], e["w20"], method, params)
                    p15 = np.array([*(e["p1"][:2] + t * e["u"]), e["z_rule"]])
                    g15 = hands[i][1][14]
                    length = xy_length(e["p1"], p15)
                    rows.append({"name": name, "p1_source": p1s, "axis": ax, "method": method,
                                 "p1_err_xy": e["p1_err"], "axis_angle_deg": e["angle"],
                                 "p15_err_xy": float(np.linalg.norm(p15[:2] - g15[:2])),
                                 "p15_err_along": t - e["t_true"],
                                 "length_pred": length, "length_ref": ref[name], "length_err": length - ref[name],
                                 "params": str({k: (np.round(v, 3).tolist() if hasattr(v, "__len__") else round(v, 3)) for k, v in params.items()})})
    with open(os.path.join(HERE, "final_eval.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows({k: (f"{v:.4f}" if isinstance(v, float) else v) for k, v in r.items()} for r in rows)

    # summary
    def stats(sel):
        e = np.array([r["length_err"] for r in sel]); a = np.array([r["p15_err_along"] for r in sel])
        p1 = np.array([r["p1_err_xy"] for r in sel]); ang = np.array([r["axis_angle_deg"] for r in sel])
        return dict(len_mae=np.abs(e).mean(), len_max=np.abs(e).max(), n05=int((np.abs(e) <= 0.5).sum()), n1=int((np.abs(e) <= 1).sum()),
                    n2=int((np.abs(e) <= 2).sum()), p15_sd=a.std(), p15_bias=a.mean(), p1_mae=p1.mean(), ang_mean=ang.mean(), ang_sd=ang.std())
    summ = []
    for p1s in P1_SOURCES:
        for ax in AXES:
            for method in COMBOS:
                sel = [r for r in rows if r["p1_source"] == p1s and r["axis"] == ax and r["method"] == method]
                summ.append({"p1_source": p1s, "axis": ax, "method": method, **stats(sel)})
    summ.sort(key=lambda s: s["len_mae"])
    with open(os.path.join(HERE, "final_eval_summary.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(summ[0].keys()))
        w.writeheader()
        w.writerows({k: (f"{v:.3f}" if isinstance(v, float) else v) for k, v in s.items()} for s in summ)
    print("\nP1 source / axis / method  ->  length MAE, max, <=0.5/1/2 mm | P15 along-axis bias, sd | P1 xy MAE | axis angle mean, sd")
    for s in summ:
        print(f"{s['p1_source']:14s} {s['axis']:17s} {s['method']:8s} | {s['len_mae']:5.2f} {s['len_max']:5.2f} {s['n05']:2d}/{s['n1']:2d}/{s['n2']:2d} "
              f"| {s['p15_bias']:+5.2f} {s['p15_sd']:5.2f} | {s['p1_mae']:4.2f} | {s['ang_mean']:+5.1f} {s['ang_sd']:4.1f}")
    best = summ[0]
    print(f"\nbest by length MAE: {best}")
    print("\nper-hand length error for the best setting:")
    for r in rows:
        if r["p1_source"] == best["p1_source"] and r["axis"] == best["axis"] and r["method"] == best["method"]:
            print(f"  {r['name']}: pred {r['length_pred']:.2f} ref {r['length_ref']:.1f} err {r['length_err']:+.2f} | P15 along {r['p15_err_along']:+.2f} | P1 xy {r['p1_err_xy']:.2f} | {r['params']}")
    # default configuration for predict.py: the best setting, its parameters fitted on all 15 hands
    key = (best["p1_source"], best["axis"])
    full = [(table[(n, *key)]["t_learned"], table[(n, *key)]["t_rule"], table[(n, *key)]["t_foot"], table[(n, *key)]["w20"], table[(n, *key)]["t_true"]) for n in names]
    params = {k: (v.tolist() if hasattr(v, "tolist") else v) for k, v in fit_params(best["method"], full).items()}
    note = "chosen by 15-hand held-out length MAE among 40 configurations; differences of ~0.2 mm are within noise"
    if GATE_X_FIXED is not None and best["method"] == "gate":
        note += f"; gate threshold fixed manually at {GATE_X_FIXED} mm (nested LOO picked 1.5; MAE is flat 0.75-2.0 mm)"
    config = {"p1_source": best["p1_source"], "axis": best["axis"], "method": best["method"], "params": params,
              "loo_15_hands": {k: (float(v) if isinstance(v, (float, np.floating)) else int(v)) for k, v in best.items() if k not in ("p1_source", "axis", "method")},
              "note": note}
    with open(os.path.join(HERE, "final_config.json"), "w", encoding="utf-8") as fh:
        json.dump(config, fh, ensure_ascii=False, indent=2)
    print(f"\nsaved final_config.json: {config}")
    length_figure(rows, names, best)


def length_figure(rows, names, best):
    """Per-hand length error: learned only, rule only, and the chosen combination (same P1/axis)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.family"] = "Malgun Gothic"
    plt.rcParams["axes.unicode_minus"] = False
    series = [("learned", "학습 P15만", "#eb6834"), ("rule", "규칙 P15만", "#2a78d6"), (best["method"], f"채택: {best['method']}", "#0b0b0b")]
    fig, ax = plt.subplots(figsize=(12, 4.2))
    x = np.arange(len(names))
    ax.axhspan(-0.5, 0.5, color="#f0efec", zorder=0)
    ax.axhline(0, color="#c3c2b7", lw=0.8)
    for j, (m, label, col) in enumerate(series):
        e = [next(r["length_err"] for r in rows if r["name"] == n and r["p1_source"] == best["p1_source"] and r["axis"] == best["axis"] and r["method"] == m) for n in names]
        ax.scatter(x + (j - 1) * 0.22, e, s=36, color=col, edgecolor="#fcfcfb", linewidth=1, zorder=3, label=f"{label} (MAE {np.abs(e).mean():.2f})")
    ax.set_xticks(x)
    ax.set_xticklabels([n[3:9] for n in names], fontsize=8)
    ax.set_ylabel("엄지 길이 오차 = 예측 - 기준값 (mm)")
    ax.set_title(f"메쉬만 입력, 15명 held-out  |  P1 = {best['p1_source']}, 축 = {best['axis']}", loc="left", fontsize=10)
    ax.yaxis.grid(True, color="#e1e0d9")
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.legend(frameon=False, fontsize=8.5, ncol=3, loc="upper left", bbox_to_anchor=(0, -0.14))
    fig.tight_layout()
    os.makedirs(os.path.join(HERE, "report"), exist_ok=True)
    fig.savefig(os.path.join(HERE, "report", "final_length_errors.png"), dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval", action="store_true")
    ap.add_argument("--gate-x", type=float, default=None, help="fix the gate threshold (mm) instead of choosing it by nested LOO")
    args = ap.parse_args()
    if args.gate_x is not None:
        GATE_X_FIXED = args.gate_x
    if args.eval:
        evaluate()
    else:
        ap.print_help()
