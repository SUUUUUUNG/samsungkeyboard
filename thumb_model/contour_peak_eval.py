"""Thumb length from the palmar contour extrema (the thumb_views approach): P15 = lift-off peak of the
band contour. Per-hand length errors and the share of hands within 0.5 / 1 / 2 mm, for
  A  hypothesis test: true P1 + true axis (as drawn in report/thumb_views), P15 = peak
  B  original mesh-only chain: apex P1 -> P6 = lowest point -> axis P1P6 -> peak (band 8 mm)
  C  mesh-only with the current best inputs: band P1 + simple axis (P1->P20' rotated 14.6 deg) -> peak
  D  extrema distance: length ~ a * |peak - contact min| + b, LOO regression (true axis)
  E  reference: same inputs as C but the adopted window rule (P20' foot 85-95 deg)
Writes thumb_model/report/contour_peak_eval.csv.

    python thumb_model/contour_peak_eval.py
"""
import csv
import glob
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)
from align_landmarks import OBJ_DIR, OUT_DIR as ALIGNED_DIR, REFERENCE_THUMB_MM, read_obj  # noqa: E402
from final_pipeline import axis_direction, rule_p15, rule_parts, sw_axis_angle, unit, xy_length  # noqa: E402
from thumb_views import contact_min_of, lower_contour, peak_of  # noqa: E402
from train_eval import thumb_apex  # noqa: E402


def contour_along(S, p1, u, band):
    far = np.array([*(p1[:2] + 60.0 * u), p1[2]])
    return lower_contour(S, p1, far, band=band)


def p6_lowest(S, p1, d):
    lat = np.array([-d[1], d[0]])
    rel = S[:, :2] - p1[:2]
    t, w = rel @ d, rel @ lat
    T = S[(t > 5.0) & (t < 40.0) & (np.abs(w) < 10.0)]
    return T[np.argmin(T[:, 2])]


def main():
    hands = []
    for f in sorted(glob.glob(os.path.join(ALIGNED_DIR, "*_aligned.lnd"))):
        n = os.path.basename(f)[:-12]
        A = np.loadtxt(f)[:, 1:]
        V, F, _ = read_obj(os.path.join(OBJ_DIR, n + ".obj"))
        hands.append((n, A, rule_parts(V, F)))
        print(f"[{n}] ready", flush=True)
    phis = np.array([sw_axis_angle(A) for _, A, _ in hands])
    rows = []
    for i, (n, A, parts) in enumerate(hands):
        g1, g6, g15 = A[0], A[5], A[14]
        S = parts["S"]
        ref = REFERENCE_THUMB_MM[n[:-1]]
        # A: true P1 + true axis, band 12 (as in thumb_views)
        ts, zs, u_t = lower_contour(S, g1, g15)
        tp, zp = peak_of(ts, zs)
        tm, zm = contact_min_of(ts, zs)
        t15 = float((g15[:2] - g1[:2]) @ u_t)
        t6 = float((g6[:2] - g1[:2]) @ u_t)
        errA = tp - ref
        # B: original chain (apex -> P6 lowest -> axis -> peak, band 8)
        apex, _, _ = thumb_apex(S, parts["tip_outline"], parts["P15_rule_B"])
        d0 = unit(parts["P15_rule_B"][:2] - apex[:2])
        p6 = p6_lowest(S, apex, d0)
        u_b = unit(p6[:2] - apex[:2])
        tsb, zsb, _ = contour_along(S, apex, u_b, 8.0)
        tpb, _ = peak_of(tsb, zsb)
        errB = tpb - ref
        # C: band P1 + simple axis -> peak (band 12)
        p1 = parts["P1_band1"]
        phi = float(np.mean(np.delete(phis, i)))
        u_c = axis_direction(parts, "p20_rotated", p1, phi_deg=phi)
        tsc, zsc, _ = contour_along(S, p1, u_c, 12.0)
        tpc, _ = peak_of(tsc, zsc)
        errC = tpc - ref
        # E: adopted window rule with the same inputs as C
        p15e, t_e, _, _, _ = rule_p15(parts, p1, u_c)
        errE = xy_length(p1, p15e) - ref
        rows.append(dict(name=n, ref=ref, t15=t15, t6=t6, peak_t=tp, peak_z=zp, cmin_t=tm, p15_minus_peak=t15 - tp, p6_minus_cmin=t6 - tm,
                         extrema_dist=tp - tm, errA=errA, errB=errB, errC=errC, errE=errE))
    # D: extrema-distance regression, LOO
    d = np.array([r["extrema_dist"] for r in rows]); L = np.array([r["ref"] for r in rows])
    for i, r in enumerate(rows):
        m = np.ones(len(rows), bool); m[i] = False
        a, b = np.polyfit(d[m], L[m], 1)
        r["errD"] = a * d[i] + b - L[i]
    keys = ["errA", "errB", "errC", "errD", "errE"]
    labels = {"errA": "A 정답 P1·정답 축 + 봉우리", "errB": "B 메쉬 체인: apex→P6 최저점→축→봉우리", "errC": "C 메쉬: 띠 P1 + 최단순 축 + 봉우리",
              "errD": "D 극점 간격 회귀 (LOO)", "errE": "E 참고: C와 같은 입력, 채택 창 규칙"}
    print(f"\n{'hand':8s} " + " ".join(f"{k:>7s}" for k in keys) + " | P15-peak  P6-cmin  peak z")
    for r in rows:
        print(f"{r['name'][3:9]:8s} " + " ".join(f"{r[k]:+7.2f}" for k in keys) + f" | {r['p15_minus_peak']:+7.1f} {r['p6_minus_cmin']:+8.1f} {r['peak_z']:6.1f}")
    print(f"\n{'variant':44s} | MAE   max  | bias    sd  | <=0.5   <=1     <=2")
    for k in keys:
        e = np.array([r[k] for r in rows]); n05, n1, n2 = (np.abs(e) <= 0.5).sum(), (np.abs(e) <= 1).sum(), (np.abs(e) <= 2).sum()
        print(f"{labels[k]:44s} | {np.abs(e).mean():5.2f} {np.abs(e).max():5.1f} | {e.mean():+5.2f} {e.std():5.2f} | {n05:2d} ({100*n05/15:3.0f}%) {n1:2d} ({100*n1/15:3.0f}%) {n2:2d} ({100*n2/15:3.0f}%)")
    v = np.array([r["p15_minus_peak"] for r in rows]); w = np.array([r["p6_minus_cmin"] for r in rows])
    print(f"\nP15 - peak: {v.mean():+.2f} +/- {v.std():.2f};  P6 - contact min: {w.mean():+.2f} +/- {w.std():.2f};  extrema distance {d.mean():.1f} +/- {d.std():.1f} = {np.mean(d / L):.2f} +/- {np.std(d / L):.2f} of length, corr {np.corrcoef(d, L)[0,1]:.2f}")
    out = os.path.join(HERE, "report", "contour_peak_eval.csv")
    with open(out, "w", newline="", encoding="utf-8") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); wr.writeheader()
        wr.writerows({k: (f"{v:.3f}" if isinstance(v, float) else v) for k, v in r.items()} for r in rows)
    print("wrote", out)


if __name__ == "__main__":
    main()
