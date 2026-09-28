"""Three-view (XY / YZ / ZX) plots of the thumb region for every hand, in the lnd
frame, with P1 / P6 / P15 / P20 marked. The YZ view also carries the palmar
lower contour along the thumb axis (min z in a +/-8 mm band) and its lift-off
peak, which is what the "peak vs slope" grouping of P15 refers to.

Output: thumb_model/report/thumb_views/<name>.png
"""
import glob
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from align_landmarks import OUT_DIR as ALIGNED_DIR, OBJ_DIR, read_obj, sample_surface  # noqa: E402

OUT = os.path.join(HERE, "report", "thumb_views")
MARGIN = 15.0
MARKS = {1: "#ff2b2b", 6: "#ff9f1c", 15: "#39ff14", 20: "#00e5ff"}
BAND = 12.0


def lower_contour(S, p1, p15, band=BAND):
    """min z per 1 mm station along the P1->P15 axis; returns stations (mm from P1), z, unit axis."""
    u = p15[:2] - p1[:2]
    u /= np.linalg.norm(u)
    lat = np.array([-u[1], u[0]])
    rel = S[:, :2] - p1[:2]
    t, w = rel @ u, rel @ lat
    ts = np.arange(-5.0, 80.0, 1.0)
    z = np.full(len(ts), np.nan)
    for i, t0 in enumerate(ts):
        m = (np.abs(t - t0) < 0.75) & (np.abs(w) < band)
        if m.sum() >= 3:
            z[i] = S[m, 2].min()
    zs = np.copy(z)
    for i in range(len(z)):  # light smoothing
        seg = z[max(0, i - 2):i + 3]
        seg = seg[~np.isnan(seg)]
        zs[i] = seg.mean() if len(seg) else np.nan
    return ts, zs, u


def peak_of(ts, zs):
    """lift-off bump maximum, t in (25, 62] mm from P1"""
    seg = np.where((ts > 25) & (ts <= 62) & ~np.isnan(zs))[0]
    i = seg[np.argmax(zs[seg])]
    return ts[i], zs[i]


def contact_min_of(ts, zs):
    """proximal-phalanx contact minimum, t in (5, 40) mm from P1"""
    seg = np.where((ts > 5) & (ts < 40) & ~np.isnan(zs))[0]
    i = seg[np.argmin(zs[seg])]
    return ts[i], zs[i]


def group_label(t15, tp, zp):
    if zp < 2.0:
        return "봉우리 약함"
    d = t15 - tp
    return "꼭대기" if abs(d) <= 2.5 else ("손목쪽 비탈 %+.1fmm" % d if d > 0 else "끝쪽 비탈 %+.1fmm" % d)


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.family"] = "Malgun Gothic"
    plt.rcParams["axes.unicode_minus"] = False
    os.makedirs(OUT, exist_ok=True)
    for lnd_path in sorted(glob.glob(os.path.join(ALIGNED_DIR, "*_aligned.lnd"))):
        name = os.path.basename(lnd_path)[:-12]
        L = np.loadtxt(os.path.join(os.path.dirname(ALIGNED_DIR), "lnd", name + ".lnd"))[:, 1:]  # lnd frame
        T = np.loadtxt(os.path.join(ALIGNED_DIR, f"{name}_transform.txt"), comments="#")
        Ti = np.linalg.inv(T)
        V, F, _ = read_obj(os.path.join(OBJ_DIR, name + ".obj"))
        Vl = (np.c_[V, np.ones(len(V))] @ Ti.T)[:, :3]
        S = np.vstack([V, sample_surface(V, F)])
        Sl = (np.c_[S, np.ones(len(S))] @ Ti.T)[:, :3]
        P = {k: L[k - 1] for k in MARKS}
        pts = np.array(list(P.values()))
        lo, hi = pts.min(0) - MARGIN, pts.max(0) + MARGIN
        crop = Vl[(Vl[:, 0] > lo[0]) & (Vl[:, 0] < hi[0]) & (Vl[:, 1] > lo[1]) & (Vl[:, 1] < hi[1])]
        ts, zs, u = lower_contour(Sl, P[1], P[15])
        tp, zp = peak_of(ts, zs)
        tm, zm = contact_min_of(ts, zs)
        t15 = (P[15][:2] - P[1][:2]) @ u
        t6 = (P[6][:2] - P[1][:2]) @ u
        label = group_label(t15, tp, zp)

        fig, axes = plt.subplots(1, 3, figsize=(20, 6.6))
        views = [("XY (top)", 0, 1, "X", "Y"), ("YZ (side)", 1, 2, "Y", "Z"), ("ZX (front)", 2, 0, "Z", "X")]
        for ax, (title, a, b, la, lb) in zip(axes, views):
            ax.scatter(crop[:, a], crop[:, b], s=1.5, color="#cfcfcf", zorder=1)
            for k, col in MARKS.items():
                ax.scatter(P[k][a], P[k][b], s=110, color=col, edgecolor="k", zorder=4, label=f"P{k}")
                ax.annotate(str(k), (P[k][a], P[k][b]), xytext=(6, 5), textcoords="offset points",
                            fontsize=10, fontweight="bold", zorder=5)
            ax.set_title(title)
            ax.set_xlabel(la)
            ax.set_ylabel(lb)
            ax.set_aspect("equal")
        # palmar contour along the thumb axis, drawn in the YZ view (y = P1.y + t*u_y)
        ax = axes[1]
        ok = ~np.isnan(zs)
        ax.plot(P[1][1] + ts[ok] * u[1], zs[ok], color="#333333", lw=1.2, zorder=3,
                label=f"엄지 축 띠(±{BAND:.0f}mm) 하단 컨투어")
        ax.scatter([P[1][1] + tp * u[1]], [zp], marker="^", s=80, color="k", zorder=6,
                   label=f"▲ 들림 봉우리 꼭대기 (t={tp:.0f}, P15 t={t15:.1f})")
        ax.scatter([P[1][1] + tm * u[1]], [zm], marker="v", s=80, color="k", zorder=6,
                   label=f"▼ 접촉 최저점 (t={tm:.0f}, P6 t={t6:.1f})")
        axes[0].legend(loc="upper left", fontsize=9)
        axes[1].legend(loc="upper right", fontsize=8)
        fig.suptitle(f"{name}: thumb region (P1/P6/P15/P20 bbox + {MARGIN:.0f}mm)   |   P15 위치: {label}   "
                     f"|   P15 - ▲ = {t15 - tp:+.1f}mm,  P6 - ▼ = {t6 - tm:+.1f}mm", fontsize=12)
        fig.tight_layout(rect=(0, 0, 1, 0.95))
        fig.savefig(os.path.join(OUT, f"{name}.png"), dpi=110)
        plt.close(fig)
        print(f"{name}: {label}  peak t={tp:.0f} z={zp:.1f} (P15 t={t15:.1f}, diff {t15 - tp:+.1f}) | "
              f"contact min t={tm:.0f} z={zm:.1f} (P6 t={t6:.1f}, diff {t6 - tm:+.1f})")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
