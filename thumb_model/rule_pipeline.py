"""Silhouette-rule landmark pipeline after patent KR 10-1217207 (hand auto-measurement).

Everything is computed from the mesh alone, no learning:
  wrist points -> wrist centre -> radial outline -> fingertips -> web points
  -> finger axes -> finger bases (P15..P19)
Reconstructed points are compared with the SW landmarks in aligned/*.lnd.

Work frame (patent convention): origin = wrist centre, +y = wrist centre -> middle
fingertip, thumb on +x, z unchanged. Results are mapped back to the obj frame.
"""
import csv
import glob
import os
import sys

import numpy as np
from scipy.signal import find_peaks

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from align_landmarks import (OBJ_DIR, OUT_DIR as ALIGNED_DIR, REFERENCE_THUMB_MM,  # noqa: E402
                             hand_axis, read_obj, sample_surface)

OUTLINE_STEP_DEG = 0.25
TIP_PROMINENCE_MM = 8.0       # a fingertip must stick out this much beyond the neighbouring notches
WRIST_TANGENT_FRAC = 0.40     # patent: tangent search limited to 40 % of the hand height
WRIST_MIN_ABOVE_CUT_MM = 5.0  # below this the concavity search has degenerated to the cut corner
WRIST_FALLBACK_MM = 21.0      # mean height of the SW's wrist points above the cut (15 hands)
AXIS_STATION_MM = 20.0        # patent: finger width bisected 2 cm above the web point
SLAB_MM = 0.75                # half-thickness of the slabs used to read widths off the samples
LND_IDX = {f"P{i}": i - 1 for i in range(1, 29)}
ALL_POINTS = [f"P{i}" for i in range(1, 29)]
# first entry is the default: its landmarks go to rule_landmarks.csv and the 28-point error columns
THUMB_AXIS_VARIANTS = ("B_along_2cm", "A_horizontal_2cm", "C_multi_station")


# ----------------------------------------------------------------- frames
class Frame:
    """p_work = S @ Rz(theta) @ (p - c); S flips x when the thumb is on -x."""

    def __init__(self, c, theta, sx):
        self.c, self.theta, self.sx = np.asarray(c, float), float(theta), float(sx)
        ct, st = np.cos(theta), np.sin(theta)
        self.R = np.array([[ct, -st, 0], [st, ct, 0], [0, 0, 1.0]])
        self.S = np.diag([sx, 1.0, 1.0])

    def to_work(self, P):
        return (np.atleast_2d(P) - self.c) @ (self.S @ self.R).T

    def from_work(self, P):
        return np.atleast_2d(P) @ (self.S @ self.R) + self.c

    @staticmethod
    def from_axis(origin, tip, sx):
        d = np.asarray(tip[:2]) - np.asarray(origin[:2])
        theta = np.pi / 2 - np.arctan2(d[1], d[0])  # rotate d onto +y
        return Frame([origin[0], origin[1], 0.0], theta, sx)


# --------------------------------------------------------------- outline
def radial_outline(S, center, step=OUTLINE_STEP_DEG, ang_range=(0.0, 180.0)):
    """Farthest surface sample per angular bin around `center` (xy). Returns
    dict of arrays sorted by angle: ang (deg), r, pts (3D)."""
    rel = S[:, :2] - np.asarray(center[:2])
    ang = np.degrees(np.arctan2(rel[:, 1], rel[:, 0])) % 360.0
    r = np.linalg.norm(rel, axis=1)
    lo, hi = ang_range
    m = (ang >= lo) & (ang <= hi)
    nb = int(round((hi - lo) / step))
    b = np.minimum(((ang[m] - lo) / step).astype(int), nb - 1)
    best = np.full(nb, -1)
    best_r = np.full(nb, -np.inf)
    idx = np.where(m)[0]
    for i, bi, ri in zip(idx, b, r[m]):
        if ri > best_r[bi]:
            best_r[bi], best[bi] = ri, i
    ok = best >= 0
    return {"ang": lo + (np.arange(nb)[ok] + 0.5) * step, "r": best_r[ok], "pts": S[best[ok]]}


def local_maxima(r, prominence):
    """Indices of local maxima of r with topographic prominence >= `prominence`."""
    peaks, _ = find_peaks(r, prominence=prominence)
    return list(peaks)


def fingertips(outline, k=5):
    """Five most prominent outline maxima in angular order (thumb first when the
    sweep starts on the thumb side)."""
    r = outline["r"]
    cands = local_maxima(r, TIP_PROMINENCE_MM)
    if len(cands) < k:
        raise RuntimeError(f"only {len(cands)} fingertip candidates found")
    cands = sorted(cands, key=lambda i: -r[i])[:k]
    return sorted(cands)


def web_between(outline, i_tip_a, i_tip_b):
    """Outline point between two tips closest to the centre (patent b7)."""
    seg = slice(i_tip_a, i_tip_b + 1)
    j = i_tip_a + int(np.argmin(outline["r"][seg]))
    return j


# ------------------------------------------------------------ wrist points
def wrist_points(S, V):
    """Patent 3D (b1)-(b3) in the work frame (thumb +x). Returns outer (thumb
    side), inner, centre — 3D points with silhouette z."""
    y_cut = V[:, 1].min()
    y_top = V[:, 1].max()
    cut = V[V[:, 1] < y_cut + 3.0]
    p_xmax = cut[np.argmax(cut[:, 0])]
    # thumb-side silhouette per 1 mm y-slab (max x), up to 40 % of the hand height
    ys = np.arange(y_cut, y_cut + WRIST_TANGENT_FRAC * (y_top - y_cut), 1.0)
    side = []
    for y0 in ys:
        s = S[(S[:, 1] >= y0) & (S[:, 1] < y0 + 1.0)]
        if len(s):
            side.append(s[np.argmax(s[:, 0])])
    side = np.array(side)
    d = side[:, :2] - p_xmax[:2]
    ok = d[:, 1] > 2.0
    angs = np.arctan2(d[ok, 0], d[ok, 1])  # angle from +y towards +x
    p_tan = side[ok][np.argmax(angs)]      # tangent point: outermost direction
    # deepest point of the outline between P_Xmax and P_tangent (distance from the chord)
    seg = side[(side[:, 1] > p_xmax[1]) & (side[:, 1] < p_tan[1])]
    if len(seg) == 0:
        seg = side
    chord = p_tan[:2] - p_xmax[:2]
    chord /= np.linalg.norm(chord)
    nrm = np.array([-chord[1], chord[0]])
    dist = (seg[:, :2] - p_xmax[:2]) @ nrm
    outer = seg[np.argmin(dist)] if dist.min() < 0 else seg[np.argmax(np.abs(dist))]
    # Our meshes are cut at the wrist, so the concavity the patent looks for is
    # often not inside the mesh and the search degenerates to the cut corner.
    # The SW's wrist points sit on the silhouette 14-30 mm above the cut
    # (mean 21 mm); fall back to that height when the search degenerates.
    if outer[1] - y_cut < WRIST_MIN_ABOVE_CUT_MM:
        s = S[np.abs(S[:, 1] - (y_cut + WRIST_FALLBACK_MM)) < 1.0]
        outer = s[np.argmax(s[:, 0])]
    # inner point: same height, minimum x (little-finger side)
    s = S[np.abs(S[:, 1] - outer[1]) < 1.0]
    inner = s[np.argmin(s[:, 0])]
    centre = 0.5 * (outer + inner)
    return outer, inner, centre


# ----------------------------------------------------- finger axes / bases
def width_bisector_at_y(S, y0, x_lo, x_hi):
    """Midpoint of the finger's x-extent in the slab y = y0 restricted to x in
    (x_lo, x_hi). Returns None if the slab is empty."""
    s = S[(np.abs(S[:, 1] - y0) < SLAB_MM) & (S[:, 0] > x_lo) & (S[:, 0] < x_hi)]
    if len(s) < 4:
        return None
    return np.array([0.5 * (s[:, 0].min() + s[:, 0].max()), y0])


def width_bisector_along(S, origin, direction, station, half_width=25.0):
    """Midpoint of the thumb's lateral extent at `station` mm from `origin`
    along `direction` (2D unit)."""
    lat = np.array([-direction[1], direction[0]])
    rel = S[:, :2] - origin
    t = rel @ direction
    w = rel @ lat
    s = (np.abs(t - station) < SLAB_MM) & (np.abs(w) < half_width)
    if s.sum() < 4:
        return None
    return origin + station * direction + 0.5 * (w[s].min() + w[s].max()) * lat


def palm_z(S, xy, radius=2.0):
    d = np.linalg.norm(S[:, :2] - xy, axis=1)
    near = S[d < radius]
    return float(np.percentile(near[:, 2], 10)) if len(near) else 0.0


def foot(a, b, p):
    """Foot of the perpendicular from p onto line ab (2D)."""
    d = b - a
    t = (p - a) @ d / (d @ d)
    return a + t * d


def finger_base(S, tip, web, bisector):
    """Patent (b8): axis = tip -> bisector; base = foot of the web on the axis,
    on the palm surface."""
    axis_a, axis_b = tip[:2], bisector[:2]
    base_xy = foot(axis_a, axis_b, web[:2])
    return np.array([base_xy[0], base_xy[1], palm_z(S, base_xy)])


def thumb_medial_axis(S, outline, i_tip, i_web, stations=(6.0, 10.0, 14.0, 18.0, 22.0, 26.0), outer_after_tip=False):
    """Medial axis of an outer digit (thumb, or little finger with
    outer_after_tip=True) between the tip and the web level. The radial sweep
    only shows the digit's inner edge near the tip (further down the rays reach
    the neighbouring finger), so the edges are read from the surface samples
    station by station: midpoint of the lateral extent within +/-15 mm of a
    preliminary axis. Returns (unit direction base->tip, point on the axis)."""
    P = outline["pts"][:, :2]
    tip = P[i_tip]
    web = P[i_web]
    radial = P[i_tip + 1:] if outer_after_tip else P[:i_tip]   # the digit's outer (free) edge
    if len(radial) < 5:
        return None
    # preliminary direction: tip -> midpoint of the web and the radial-edge point equally far from the tip
    d_web = np.linalg.norm(web - tip)
    r_match = radial[np.argmin(np.abs(np.linalg.norm(radial - tip, axis=1) - d_web))]
    d0 = tip - 0.5 * (web + r_match)
    d0 /= np.linalg.norm(d0)
    mids = [width_bisector_along(S, tip, -d0, s, half_width=15.0) for s in stations if s < 0.8 * d_web]
    mids = np.array([m for m in mids if m is not None])
    if len(mids) < 3:
        return None
    ctr = mids.mean(0)
    _, _, vt = np.linalg.svd(mids - ctr, full_matrices=False)
    d = vt[0] if vt[0] @ d0 > 0 else -vt[0]
    return d, ctr


def thumb_bisector(S, tip, web, variant, x_lo, outline=None, i_tip=None, i_web=None):
    """Point B that, together with the tip, defines the thumb axis (work frame,
    thumb on +x). Variants:
      A_horizontal_2cm  patent wording: bisector of the horizontal width 2 cm above the web
      B_along_2cm       bisector of the width perpendicular to the medial axis, 2 cm from the web level
      C_multi_station   medial axis through the tip (mean of edge midpoints at several stations)"""
    if variant == "A_horizontal_2cm":
        return width_bisector_at_y(S, web[1] + AXIS_STATION_MM, x_lo, np.inf)
    axis = thumb_medial_axis(S, outline, i_tip, i_web)
    if axis is None:
        return None
    d, ctr = axis
    if variant == "B_along_2cm":
        web_t = (web[:2] - tip[:2]) @ (-d)
        b = width_bisector_along(S, tip[:2], -d, web_t - AXIS_STATION_MM, half_width=15.0)
        return b
    # C: the medial-axis line (through its centroid), evaluated 20 mm from the tip
    return ctr + ((tip[:2] - ctr) @ d - 20.0) * d


# --------------------------------------------------------------- pipeline
def run(V, F, thumb_variant="B_along_2cm", seed=0, frame_override=None):
    """frame_override = (centre_xyz, middle_tip_xyz) in the obj frame skips the
    patent wrist rule and uses that centre/axis (diagnostic: SW's own P28/P3)."""
    S = np.vstack([V, sample_surface(V, F, seed=seed)])
    # pass 0: initial frame from the wrist cut and the longest finger, thumb side from geometry
    origin0, axis0 = hand_axis(V)
    tip0 = origin0 + 150 * axis0
    fr0 = Frame.from_axis(origin0, tip0, +1.0)
    Sw = fr0.to_work(S)
    full = radial_outline(Sw, [0, 0], ang_range=(0.0, 180.0))
    tips0 = fingertips(full)
    pts0 = full["pts"][tips0]
    mid = pts0[np.argmax(pts0[:, 1])]
    far = pts0[np.argmax(np.linalg.norm(pts0[:, :2] - mid[:2], axis=1))]
    sx = 1.0 if far[0] > mid[0] else -1.0
    if sx < 0:
        fr0 = Frame.from_axis(origin0, tip0, -1.0)
        Sw = fr0.to_work(S)
    Vw = fr0.to_work(V)
    # pass 1: patent wrist points and centre, then the final frame (centre -> middle tip = +y)
    if frame_override is None:
        outer, inner, centre = wrist_points(Sw, Vw)
        outl = radial_outline(Sw, centre, ang_range=(0.0, 180.0))
        tips = fingertips(outl)
        mid_tip = outl["pts"][tips][np.argmax(outl["pts"][tips][:, 1])]
        fr = Frame.from_axis(fr0.from_work(centre)[0], fr0.from_work(mid_tip)[0], sx)
        Sw, Vw = fr.to_work(S), fr.to_work(V)
        outer, inner, centre = wrist_points(Sw, Vw)
    else:
        c_obj, tip_obj = frame_override
        fr = Frame.from_axis(np.asarray(c_obj), np.asarray(tip_obj), sx)
        Sw, Vw = fr.to_work(S), fr.to_work(V)
        centre = fr.to_work(np.asarray(c_obj))[0]
        outer = inner = centre
    centre_xy = centre[:2]
    outl = radial_outline(Sw, centre_xy, ang_range=(0.0, 180.0))
    tips = fingertips(outl)
    webs = [web_between(outl, tips[i], tips[i + 1]) for i in range(4)]
    T = outl["pts"][tips]      # thumb, index, middle, ring, little (angular order from +x)
    W = outl["pts"][webs]      # thumb-index, index-middle, middle-ring, ring-little
    out = {"P1": T[0], "P2": T[1], "P3": T[2], "P4": T[3], "P5": T[4],
           "P20": W[0], "P21": W[1], "P22": W[2], "P23": W[3],
           "P26": outer, "P27": inner, "P28": centre}
    # fingers 2-4: axis = tip -> horizontal width bisector 2 cm above the finger's web
    # (patent uses the web on the finger's own side)
    finger_webs = {"P16": (1, W[1], W[0][0], W[1][0]), "P17": (2, W[2], W[1][0], W[2][0]),
                   "P18": (3, W[3], W[2][0], W[3][0])}
    for name, (ti, web, x_hi, x_lo) in finger_webs.items():
        lo, hi = sorted([x_lo, x_hi])
        b = width_bisector_at_y(Sw, web[1] + AXIS_STATION_MM, lo, hi)
        if b is None:
            b = T[ti][:2]
        out[name] = finger_base(Sw, T[ti], web, b)
        out[name + "_B"] = np.array([b[0], b[1], 0.0])
    # little finger: short and splayed, so a horizontal slab 2 cm above its web often
    # misses it; use the medial axis between its free edge and the ring-little web
    axis = thumb_medial_axis(Sw, outl, tips[4], webs[3], outer_after_tip=True)
    b = T[4][:2] - 20.0 * axis[0] if axis is not None else T[4][:2]
    out["P19"] = finger_base(Sw, T[4], W[3], b)
    out["P19_B"] = np.array([b[0], b[1], 0.0])
    # thumb
    b = thumb_bisector(Sw, T[0], W[0], thumb_variant, x_lo=W[0][0], outline=outl, i_tip=tips[0], i_web=webs[0])
    if b is None:
        b = T[0][:2]
    out["P15"] = finger_base(Sw, T[0], W[0], b)
    out["P15_B"] = np.array([b[0], b[1], 0.0])
    out["thumb_variant"] = thumb_variant
    # P6 (thumb IP): not in the patent; the SW's P6 lies on the P1-P15 line at
    # 0.506 +/- 0.029 of the way from P15, so use the midpoint (approximation)
    mid = 0.5 * (out["P1"][:2] + out["P15"][:2])
    out["P6"] = np.array([mid[0], mid[1], palm_z(Sw, mid)])
    # P25 (patent 손가쪽점): outline below the index finger at y between the
    # thumb-index web and the index base, 4.5:5.5 (SW's P25 sits at ~0.55)
    y25 = W[0][1] + 0.55 * (out["P16"][1] - W[0][1])
    seg = outl["pts"][webs[0]:tips[1] + 1]
    out["P25"] = seg[np.argmin(np.abs(seg[:, 1] - y25))]
    # P24 (patent 손안쪽점): outline below the little finger, 1.5 cm below the little base
    seg = outl["pts"][tips[4]:]
    out["P24"] = seg[np.argmin(np.abs(seg[:, 1] - (out["P19"][1] - 15.0)))]
    # derived points, same rules as the SW: thirds of tip->base (z = 0), wrist midpoint (z = 1)
    for tip_k, base_k, a, b_ in (("P2", "P16", "P7", "P11"), ("P3", "P17", "P8", "P12"),
                                 ("P4", "P18", "P9", "P13"), ("P5", "P19", "P10", "P14")):
        t, bpt = out[tip_k], out[base_k]
        out[a] = np.array([*(t[:2] + (bpt[:2] - t[:2]) / 3.0), 0.0])
        out[b_] = np.array([*(t[:2] + 2.0 * (bpt[:2] - t[:2]) / 3.0), 0.0])
    out["P28"] = np.array([*(0.5 * (out["P26"][:2] + out["P27"][:2])), 1.0])
    # back to the obj frame
    res = {k: fr.from_work(v)[0] for k, v in out.items() if k != "thumb_variant"}
    res["frame"] = fr
    res["outline"] = outl
    res["thumb_variant"] = thumb_variant
    return res


def axis_component(pred, gt, tip, base):
    u = (tip[:2] - base[:2])
    u /= np.linalg.norm(u)
    return float((pred[:2] - gt[:2]) @ u)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--frame", choices=("patent", "sw"), default="patent",
                    help="patent: wrist rule + own frame; sw: diagnostic, use the SW's P28/P3 as centre/axis")
    args = ap.parse_args()
    variants = THUMB_AXIS_VARIANTS
    rows, land = [], []
    for obj_path in sorted(glob.glob(os.path.join(OBJ_DIR, "*.obj"))):
        name = os.path.splitext(os.path.basename(obj_path))[0]
        V, F, _ = read_obj(obj_path)
        A = np.loadtxt(os.path.join(ALIGNED_DIR, f"{name}_aligned.lnd"))[:, 1:]
        ref = REFERENCE_THUMB_MM[name[:-1]]
        row = {"name": name}
        override = (A[LND_IDX["P28"]], A[LND_IDX["P3"]]) if args.frame == "sw" else None
        for vi, var in enumerate(variants):
            r = run(V, F, thumb_variant=var, frame_override=override)
            if vi == 0:
                for k in ("P1", "P2", "P3", "P4", "P5", "P20", "P21", "P22", "P23", "P26", "P27", "P28",
                          "P16", "P17", "P18", "P19"):
                    g = A[LND_IDX[k]]
                    row[f"{k}_err"] = float(np.linalg.norm(r[k] - g))
                    row[f"{k}_err_xy"] = float(np.linalg.norm(r[k][:2] - g[:2]))
                for k, tip, base in (("P16", "P2", "P16"), ("P17", "P3", "P17"), ("P18", "P4", "P18"), ("P19", "P5", "P19")):
                    row[f"{k}_axis"] = axis_component(r[k], A[LND_IDX[k]], A[LND_IDX[tip]], A[LND_IDX[base]])
                land.append({"name": name, **{f"{k}_{c}": float(r[k][i]) for k in
                             ALL_POINTS + ["P15_B"] for i, c in enumerate("xyz")}})
                for k in ALL_POINTS:
                    row[f"{k}_err3d"] = float(np.linalg.norm(r[k] - A[LND_IDX[k]]))
                    row[f"{k}_errxy"] = float(np.linalg.norm(r[k][:2] - A[LND_IDX[k]][:2]))
            g15 = A[LND_IDX["P15"]]
            row[f"P15_err[{var}]"] = float(np.linalg.norm(r["P15"] - g15))
            row[f"P15_axis[{var}]"] = axis_component(r["P15"], g15, A[LND_IDX["P1"]], g15)
            length = float(np.linalg.norm(r["P1"][:2] - r["P15"][:2]))
            row[f"len[{var}]"] = length
            row[f"len_err[{var}]"] = length - ref
        rows.append(row)
        print(f"{name}: P28 {row['P28_err']:.1f}  P1 {row['P1_err']:.1f}  P20 {row['P20_err']:.1f}  "
              f"P21-23 {row['P21_err']:.1f}/{row['P22_err']:.1f}/{row['P23_err']:.1f}  "
              f"P16-19 axis {row['P16_axis']:+.1f}/{row['P17_axis']:+.1f}/{row['P18_axis']:+.1f}/{row['P19_axis']:+.1f}  "
              + "  ".join(f"P15[{v[0]}] {row[f'P15_axis[{v}]']:+.2f} len_err {row[f'len_err[{v}]']:+.2f}" for v in variants))

    def stats(key):
        v = np.array([r[key] for r in rows])
        return f"mean {v.mean():+.2f} sd {v.std():.2f} max|.| {np.abs(v).max():.2f}"

    print("\nReconstructed vs SW landmarks (3D distance, mm):")
    for k in ("P28", "P26", "P27", "P1", "P2", "P3", "P4", "P5", "P20", "P21", "P22", "P23", "P16", "P17", "P18", "P19"):
        print(f"  {k:4s} {stats(f'{k}_err')}")
    print("\nFinger bases, along-axis component (mm):")
    for k in ("P16", "P17", "P18", "P19"):
        print(f"  {k:4s} {stats(f'{k}_axis')}")
    print("\nThumb base P15 by axis variant (along-axis error, mm) and thumb length error vs reference:")
    for var in variants:
        e = np.array([r[f"len_err[{var}]"] for r in rows])
        print(f"  {var:18s} P15 axis {stats(f'P15_axis[{var}]')} | length MAE {np.abs(e).mean():.2f} max {np.abs(e).max():.2f} <=0.5: {(np.abs(e) <= 0.5).sum()}/15")

    suffix = "" if args.frame == "patent" else "_swframe"
    with open(os.path.join(HERE, f"rule_eval{suffix}.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows({k: (f"{v:.4f}" if isinstance(v, float) else v) for k, v in r.items()} for r in rows)
    with open(os.path.join(HERE, f"rule_landmarks{suffix}.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(land[0].keys()))
        w.writeheader()
        w.writerows({k: (f"{v:.4f}" if isinstance(v, float) else v) for k, v in r.items()} for r in land)


if __name__ == "__main__":
    main()
