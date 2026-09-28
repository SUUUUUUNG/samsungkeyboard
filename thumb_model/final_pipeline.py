"""Final mesh-only pipeline: P1, P15 and the xy thumb length from an obj mesh.

Two families share this code:
  combined   learned model (train_eval.py) + geometric rule; adopted setting in final_config.json
  rule-only  no learning at all: P1 band rule, axis P1 -> B (thumb-width midpoint 2 cm distal
             of the P20 level, patent finger-axis rule, built with the band P1 as the tip so the
             silhouette tip point is not used), P20 foot window; adopted setting in
             final_config_rule.json. (midline_station = earlier variant with a nested-LOO station.)

  python thumb_model/final_pipeline.py --eval [--gate-x 1.0]
        15-hand held-out evaluation of every P1 / axis / combination variant
        (combination parameters and the midline station are chosen by nested LOO)
  from final_pipeline import predict_hand      used by predict.py --mode combined / --mode rule

Rule side (no learning): rule_pipeline.run -> outline, P20', midline;
P1_rule = midpoint of the midline apex and the max-y point of the tip, or the
band rule (outermost point within 1 mm of the max-y point, hand frame);
axis through P1; lower band contour along it; P15_rule = highest contour
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
BAND_Y = 1.0                 # P1 band rule: points within this y distance (hand frame) of the max-y point
STATIONS = (15.0, 20.0, 25.0, 30.0, 35.0, 40.0, 45.0, 55.0)   # midline station candidates (mm proximal of P1's projection)
STATION_AXES = tuple(f"midline_t{int(t)}" for t in STATIONS)
BASE_AXES = ("learned", "midline_parallel", "midline_foot", "midline", "p1_to_B", "p20_rotated")
AXES = BASE_AXES + ("midline_station",)   # midline_station: station chosen by nested LOO (eval) / params["t_station"] (predict)
P1_SOURCES = ("rule_midpoint", "rule_band1", "learned_apex")
COMBOS = ("learned", "rule", "average", "gate", "stack")
COMBINED_FAMILY = dict(p1=("rule_midpoint", "learned_apex"), axes=BASE_AXES[:4], methods=COMBOS)   # what final_config.json is chosen from (unchanged)
RULE_ONLY_FAMILY = dict(p1=("rule_midpoint", "rule_band1"), axes=("p1_to_B", "midline_parallel", "midline_foot", "midline", "midline_station"), methods=("rule",))
SIMPLE_FAMILY = dict(p1=("rule_band1",), axes=("p20_rotated",), methods=("rule",))   # simplest axis: P1 -> P20' rotated by the SW's constant angle -> final_config_simple.json


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
    work = r["frame"].to_work(near)                          # hand frame: wrist centre -> middle tip = +y, thumb side = +x
    maxy = near[np.argmax(work[:, 1])]
    p1_rule = 0.5 * (apex + maxy)
    top = work[work[:, 1] > work[:, 1].max() - BAND_Y]       # band rule: within 1 mm (y) of the max-y point, outermost on the thumb side
    p1_band1 = r["frame"].from_work(top[np.argmax(top[:, 0])]).reshape(3)
    B = thumb_B(S, r, p1_band1)                              # width midpoint 2 cm distal of the P20 level, built with the band P1 as the tip
    return {"S": S, "P20": r["P20"], "tip_outline": r["P1"], "apex": apex, "maxy": maxy, "P1_rule": p1_rule, "P1_band1": p1_band1,
            "B": B, "d_mid": d_mid, "P15_B": r["P15_B"], "P15_rule_B": r["P15"], "frame": r["frame"]}


def thumb_B(S, r, tip, return_parts=False):
    """B for the rule-only axis (obj frame): the patent's finger-axis point (thumb-width
    midpoint AXIS_STATION_MM = 2 cm distal of the P20 level) built from `tip` (obj frame)
    instead of the silhouette tip. The silhouette is still used for P20 and for the
    thumb's outer edge; the outline tip itself is not used."""
    fr, outl = r["frame"], r["outline"]
    Sw = fr.to_work(S)
    tips = rp.fingertips(outl)
    webs = [rp.web_between(outl, tips[i], tips[i + 1]) for i in range(4)]
    tip_w = fr.to_work(tip)[0][:2]
    d, ctr = rp.thumb_medial_axis(Sw, outl, tips[0], webs[0], tip_override=tip_w)
    web_w = outl["pts"][webs[0], :2]
    web_t = (web_w - tip_w) @ (-d)
    b = rp.width_bisector_along(Sw, tip_w, -d, web_t - rp.AXIS_STATION_MM, half_width=15.0)
    if b is None:                                             # degenerate slab: fall back to the run()'s own B
        b = fr.to_work(r["P15_B"])[0][:2]
    B = fr.from_work(np.array([b[0], b[1], 0.0]))[0]
    if return_parts:
        return B, {"Sw": Sw, "outl": outl, "tips": tips, "webs": webs, "tip_w": tip_w, "d": d, "ctr": ctr, "web_w": web_w, "web_t": web_t, "b_w": b}
    return B


def p1_from_source(parts, learned, p1_source):
    if p1_source == "rule_midpoint":
        return parts["P1_rule"]
    if p1_source == "rule_band1":
        return parts["P1_band1"]
    if p1_source == "learned_apex":
        return learned["P1"]
    raise ValueError(p1_source)


def sw_axis_angle(A):
    """Signed angle (deg, xy) from the P1->P15 axis to the P1->P20 line in a set of SW landmarks
    (28 x 3 array). In the SW's own output this is 14.6 +/- 1.1 deg over the 15 hands."""
    a = unit(A[14][:2] - A[0][:2])
    b = unit(A[19][:2] - A[0][:2])
    return float(np.degrees(np.arctan2(a[0] * b[1] - a[1] * b[0], a @ b)))


def axis_direction(parts, axis, p1, p15_learned=None, t_station=None, phi_deg=None):
    """Unit direction from P1 towards the base for the chosen axis definition."""
    if axis == "learned":
        return unit(p15_learned[:2] - p1[:2])
    if axis == "p20_rotated":                                 # P1 -> P20' line rotated back by the SW's constant angle (simplest axis)
        v = unit(parts["P20"][:2] - p1[:2])
        a = np.radians(-float(phi_deg))
        return np.array([np.cos(a) * v[0] - np.sin(a) * v[1], np.sin(a) * v[0] + np.cos(a) * v[1]])
    if axis in ("midline_parallel", "midline"):              # parallel to the midline / the midline itself (through the apex)
        return -parts["d_mid"]
    if axis == "midline_foot":                                # P1 -> foot of P20 on the midline axis
        return unit(parts["P15_rule_B"][:2] - p1[:2])
    if axis == "p1_to_B":                                     # P1 -> B: thumb-width midpoint 2 cm distal of the P20 level, built from the band P1 (patent finger-axis rule with P1 as the tip)
        return unit(parts["B"][:2] - p1[:2])
    if axis.startswith("midline_t") or axis == "midline_station":   # P1 -> midline point t_s mm proximal of P1's projection
        t_s = float(axis[9:]) if axis.startswith("midline_t") else float(t_station)
        d, b = parts["d_mid"], parts["P15_B"][:2]
        foot = b + ((p1[:2] - b) @ d) * d
        return unit(foot - t_s * d - p1[:2])
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


def needs_learning(p1_source, axis, method):
    return p1_source == "learned_apex" or axis == "learned" or method != "rule"


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
def predict_hand(V, F, bundle=None, p1_source="rule_midpoint", axis="midline_foot", method="rule", params=None):
    """Mesh-only prediction. Returns dict with P1, P15, length_mm and the parts.
    bundle (the learned model) is only needed when p1_source / axis / method use it."""
    params = params or {}
    parts = rule_parts(V, F)
    learned = {"P1": None, "P15": None}
    if needs_learning(p1_source, axis, method):
        if bundle is None:
            raise ValueError(f"configuration ({p1_source}, {axis}, {method}) needs the learned model; pass bundle")
        learned, _ = learned_parts_from_model(V, F, bundle)
    p1 = p1_from_source(parts, learned, p1_source)
    u = axis_direction(parts, axis, p1, learned["P15"], params.get("t_station"), params.get("phi_deg"))
    p15_rule, t_rule, t_foot, w20, _ = rule_p15(parts, p1, u)
    t_learned = float((learned["P15"][:2] - p1[:2]) @ u) if learned["P15"] is not None else None
    t = combine(t_learned, t_rule, t_foot, w20, method, params)
    p15 = np.array([*(p1[:2] + t * u), p15_rule[2]])
    return {"P1": p1, "P15": p15, "length_mm": xy_length(p1, p15),
            "parts": {"P20": parts["P20"], "P15_rule": p15_rule, "P15_learned": learned["P15"], "P1_learned": learned["P1"],
                      "t_learned": t_learned, "t_rule": t_rule, "t_foot": t_foot, "axis": axis, "method": method,
                      "t_station": params.get("t_station"), "phi_deg": params.get("phi_deg"), "u": u}}


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

    names = [h[0] for h in hands]
    ref = {n: REFERENCE_THUMB_MM[n[:-1]] for n in names}
    phis = {n: sw_axis_angle(A) for n, A, *_ in hands}       # SW's P1->P15 vs P1->P20 angle per hand (aligned lnd = rigid copy of lnd)
    concrete_axes = BASE_AXES + STATION_AXES
    # per hand, per P1 source, per concrete axis: the rule / learned along-axis quantities
    table = {}
    for name, A, parts, learned in hands:
        g1, g15 = A[0], A[14]
        u_true = unit(g1[:2] - g15[:2])
        phi_loo = float(np.mean([phis[m] for m in names if m != name]))   # nested: the held-out hand's own angle is not used
        for p1s in P1_SOURCES:
            p1 = p1_from_source(parts, learned, p1s)
            for ax in concrete_axes:
                p1_ax = parts["apex"] if ax == "midline" else p1
                u = axis_direction(parts, ax, p1_ax, learned["P15"], phi_deg=phi_loo)
                p15_rule, t_rule, t_foot, w20, _ = rule_p15(parts, p1_ax, u)
                t_learned = float((learned["P15"][:2] - p1_ax[:2]) @ u)
                t_true = float((g15[:2] - p1_ax[:2]) @ u)
                angle = float(np.degrees(np.arctan2(u_true[0] * (-u)[1] - u_true[1] * (-u)[0], u_true @ (-u))))
                table[(name, p1s, ax)] = dict(p1=p1_ax, u=u, t_learned=t_learned, t_rule=t_rule, t_foot=t_foot, w20=w20, t_true=t_true,
                                              angle=angle, p1_err=float(np.linalg.norm(p1_ax[:2] - g1[:2])),
                                              p15_rule_err=float(np.linalg.norm(p15_rule[:2] - g15[:2])),
                                              z_rule=p15_rule[2], g15=g15, phi_deg=phi_loo if ax == "p20_rotated" else None)
        print(f"[{name}] table ready", flush=True)

    def train_rows(p1s, ax, train_names):
        return [(table[(n, p1s, ax)]["t_learned"], table[(n, p1s, ax)]["t_rule"], table[(n, p1s, ax)]["t_foot"],
                 table[(n, p1s, ax)]["w20"], table[(n, p1s, ax)]["t_true"]) for n in train_names]

    def entry_result(e, method, params, ref_len):
        t = combine(e["t_learned"], e["t_rule"], e["t_foot"], e["w20"], method, params)
        p15 = np.array([*(e["p1"][:2] + t * e["u"]), e["z_rule"]])
        length = xy_length(e["p1"], p15)
        return length, length - ref_len, t, p15

    def fmt_params(params):
        return str({k: (np.round(v, 3).tolist() if hasattr(v, "__len__") else round(v, 3)) for k, v in params.items()})

    rows = []
    # every configuration; parameters (and the station for midline_station) chosen by nested LOO
    for p1s in P1_SOURCES:
        for ax in concrete_axes + ("midline_station",):
            for method in COMBOS:
                for name in names:
                    train_names = [n for n in names if n != name]
                    if ax == "midline_station":
                        best = (np.inf, None, None)
                        for sax in STATION_AXES:
                            params = fit_params(method, train_rows(p1s, sax, train_names))
                            mae = np.mean([abs(entry_result(table[(n, p1s, sax)], method, params, ref[n])[1]) for n in train_names])
                            if mae < best[0]:
                                best = (mae, sax, params)
                        use_ax, params = best[1], {**best[2], "t_station": float(best[1][9:])}
                    else:
                        use_ax, params = ax, fit_params(method, train_rows(p1s, ax, train_names))
                        if ax == "p20_rotated":
                            params = {**params, "phi_deg": table[(name, p1s, ax)]["phi_deg"]}
                    e = table[(name, p1s, use_ax)]
                    length, err, t, p15 = entry_result(e, method, params, ref[name])
                    rows.append({"name": name, "p1_source": p1s, "axis": ax, "method": method,
                                 "p1_err_xy": e["p1_err"], "axis_angle_deg": e["angle"],
                                 "p15_err_xy": float(np.linalg.norm(p15[:2] - e["g15"][:2])),
                                 "p15_err_along": t - e["t_true"],
                                 "length_pred": length, "length_ref": ref[name], "length_err": err,
                                 "params": fmt_params(params)})
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

    def select(p1s, ax, method):
        return [r for r in rows if r["p1_source"] == p1s and r["axis"] == ax and r["method"] == method]

    summ = []
    for p1s in P1_SOURCES:
        for ax in concrete_axes + ("midline_station",):
            for method in COMBOS:
                summ.append({"p1_source": p1s, "axis": ax, "method": method, "uses_learning": needs_learning(p1s, ax, method), **stats(select(p1s, ax, method))})
    summ.sort(key=lambda s: s["len_mae"])
    with open(os.path.join(HERE, "final_eval_summary.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(summ[0].keys()))
        w.writeheader()
        w.writerows({k: (f"{v:.3f}" if isinstance(v, float) else v) for k, v in s.items()} for s in summ)

    def print_rows(sel, title):
        print(f"\n{title}\nP1 source / axis / method  ->  length MAE, max, <=0.5/1/2 mm | P15 along-axis bias, sd | P1 xy MAE | axis angle mean, sd")
        for s in sel:
            print(f"{s['p1_source']:14s} {s['axis']:17s} {s['method']:8s} | {s['len_mae']:5.2f} {s['len_max']:5.2f} {s['n05']:2d}/{s['n1']:2d}/{s['n2']:2d} "
                  f"| {s['p15_bias']:+5.2f} {s['p15_sd']:5.2f} | {s['p1_mae']:4.2f} | {s['ang_mean']:+5.1f} {s['ang_sd']:4.1f}")

    def in_family(s, fam):
        return s["p1_source"] in fam["p1"] and s["axis"] in fam["axes"] and s["method"] in fam["methods"]

    print_rows(summ[:20], f"top 20 of {len(summ)} configurations (nested LOO for parameters / station)")
    rule_only = [s for s in summ if not s["uses_learning"]]
    print_rows(rule_only, "no-learning configurations (P1 rule, mesh axis, rule P15); midline_tNN = fixed station (tuned on all 15), midline_station = nested LOO, p20_rotated = SW angle by nested LOO")

    def per_hand(best):
        print(f"\nper-hand length error for {best['p1_source']} / {best['axis']} / {best['method']}:")
        for r in select(best["p1_source"], best["axis"], best["method"]):
            print(f"  {r['name']}: pred {r['length_pred']:.2f} ref {r['length_ref']:.1f} err {r['length_err']:+.2f} | P15 along {r['p15_err_along']:+.2f} | P1 xy {r['p1_err_xy']:.2f} | {r['params']}")

    def full_params(best):
        """parameters fitted on all 15 hands for the deployable config (station: lowest 15-hand MAE)."""
        ax = best["axis"]
        if ax == "midline_station":
            cands = []
            for sax in STATION_AXES:
                params = fit_params(best["method"], train_rows(best["p1_source"], sax, names))
                mae = np.mean([abs(entry_result(table[(n, best["p1_source"], sax)], best["method"], params, ref[n])[1]) for n in names])
                cands.append((mae, sax, params))
            mae, sax, params = min(cands, key=lambda c: c[0])
            params = {**params, "t_station": float(sax[9:])}
        else:
            params = fit_params(best["method"], train_rows(best["p1_source"], ax, names))
            if ax == "p20_rotated":
                params = {**params, "phi_deg": float(np.mean(list(phis.values())))}
        return {k: (v.tolist() if hasattr(v, "tolist") else v) for k, v in params.items()}

    def save_config(best, path, note):
        config = {"p1_source": best["p1_source"], "axis": best["axis"], "method": best["method"], "params": full_params(best),
                  "loo_15_hands": {k: (float(v) if isinstance(v, (float, np.floating)) else int(v)) for k, v in best.items() if k not in ("p1_source", "axis", "method", "uses_learning")},
                  "note": note}
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(config, fh, ensure_ascii=False, indent=2)
        print(f"\nsaved {os.path.basename(path)}: {config}")
        return config

    # combined family (unchanged selection rule) -> final_config.json
    best_c = next(s for s in summ if in_family(s, COMBINED_FAMILY))
    per_hand(best_c)
    note = "chosen by 15-hand held-out length MAE among the combined family (2 P1 x 4 axes x 5 methods); differences of ~0.2 mm are within noise"
    if GATE_X_FIXED is not None and best_c["method"] == "gate":
        note += f"; gate threshold fixed manually at {GATE_X_FIXED} mm (nested LOO picked 1.5; MAE is flat 0.75-2.0 mm)"
    save_config(best_c, os.path.join(HERE, "final_config.json"), note)
    # rule-only family (no learning; station by nested LOO) -> final_config_rule.json
    best_r = next(s for s in summ if in_family(s, RULE_ONLY_FAMILY))
    per_hand(best_r)
    note_r = "no learning: chosen by 15-hand held-out length MAE among the rule-only family (2 P1 x 5 axes)"
    if best_r["axis"] == "midline_station":
        note_r += "; t_station here is the station with the lowest 15-hand MAE (the held-out number used nested LOO per fold)"
    elif best_r["axis"] == "p1_to_B":
        note_r += "; axis = band P1 -> B (thumb-width midpoint 2 cm distal of the P20 level: patent finger-axis rule, no fitted constant)"
    save_config(best_r, os.path.join(HERE, "final_config_rule.json"), note_r)
    # simplest axis (band P1 -> P20' rotated by the SW's constant angle) -> final_config_simple.json
    best_s = next(s for s in summ if in_family(s, SIMPLE_FAMILY))
    per_hand(best_s)
    ph = np.array(list(phis.values()))
    save_config(best_s, os.path.join(HERE, "final_config_simple.json"),
                f"no learning, simplest axis: P1 (band rule) -> P20' line rotated by phi_deg towards the thumb's outer side; "
                f"phi_deg = mean SW angle(P1->P15 vs P1->P20) over the 15 hands ({ph.mean():.1f} +/- {ph.std():.1f} deg; "
                f"the held-out numbers used the leave-one-out mean per fold)")
    length_figure(rows, names, [(best_c, f"학습+규칙 채택 ({best_c['method']})", "#0b0b0b"),
                                (best_r, f"규칙만 채택: P1 {best_r['p1_source'][5:]}, 축 {best_r['axis']}", "#1a9850"),
                                (best_s, "최단순 축: P1→P20′ 선을 14.6° 회전", "#7a3fbf")],
                  "최단순 축(P1→P20′ 회전) vs 규칙 전용 vs 결합 (15명 held-out)", "final_length_errors_simple.png")
    ref_r = {"p1_source": "rule_band1", "axis": "midline_station", "method": "rule"}
    length_figure(rows, names, [(dict(p1_source=best_c["p1_source"], axis=best_c["axis"], method="learned"), "학습 P15만", "#eb6834"),
                                (dict(p1_source=best_c["p1_source"], axis=best_c["axis"], method="rule"), "규칙 P15만", "#2a78d6"),
                                (best_c, f"채택: {best_c['method']}", "#0b0b0b")],
                  f"메쉬만 입력, 15명 held-out  |  P1 = {best_c['p1_source']}, 축 = {best_c['axis']}", "final_length_errors.png")
    length_figure(rows, names, [(best_c, f"학습+규칙 채택 ({best_c['method']})", "#0b0b0b"),
                                (ref_r, "규칙만 이전안: P1 띠, 축 중심선 20mm 지점(LOO)", "#2a78d6"),
                                (best_r, f"규칙만 채택: P1 {best_r['p1_source'][5:]}, 축 {best_r['axis']}", "#1a9850")],
                  "학습 없는 규칙 전용 파이프라인 vs 채택된 결합 파이프라인 (15명 held-out)", "final_length_errors_rule_only.png")


def length_figure(rows, names, series, title, fname):
    """Per-hand length error for a few (selector, label, colour) series."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.family"] = "Malgun Gothic"
    plt.rcParams["axes.unicode_minus"] = False
    fig, ax = plt.subplots(figsize=(12, 4.2))
    x = np.arange(len(names))
    ax.axhspan(-0.5, 0.5, color="#f0efec", zorder=0)
    ax.axhline(0, color="#c3c2b7", lw=0.8)
    for j, (sel, label, col) in enumerate(series):
        e = [next(r["length_err"] for r in rows if r["name"] == n and r["p1_source"] == sel["p1_source"] and r["axis"] == sel["axis"] and r["method"] == sel["method"]) for n in names]
        ax.scatter(x + (j - 1) * 0.22, e, s=36, color=col, edgecolor="#fcfcfb", linewidth=1, zorder=3, label=f"{label} (MAE {np.abs(e).mean():.2f})")
    ax.set_xticks(x)
    ax.set_xticklabels([n[3:9] for n in names], fontsize=8)
    ax.set_ylabel("엄지 길이 오차 = 예측 - 기준값 (mm)")
    ax.set_title(title, loc="left", fontsize=10)
    ax.yaxis.grid(True, color="#e1e0d9")
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.legend(frameon=False, fontsize=8.5, ncol=3, loc="upper left", bbox_to_anchor=(0, -0.14))
    fig.tight_layout()
    os.makedirs(os.path.join(HERE, "report"), exist_ok=True)
    fig.savefig(os.path.join(HERE, "report", fname), dpi=160)
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
