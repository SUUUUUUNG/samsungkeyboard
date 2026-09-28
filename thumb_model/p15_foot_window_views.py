"""Per-hand views of the "P20 foot +/- window" P15 estimate: the highest point of
the palmar band contour among axis stations reached from P20 at 85-95 degrees
(= perpendicular foot +/- ~1.1 mm). Uses the SW's own thumb axis and P20, so
this shows the rule's ceiling, not a mesh-only prediction.

Output: thumb_model/report/p15_foot85_95/<name>.png
"""
import glob
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)
from align_landmarks import OUT_DIR as ALIGNED_DIR, OBJ_DIR, read_obj, sample_surface  # noqa: E402
from thumb_views import BAND, MARGIN, MARKS, lower_contour  # noqa: E402

OUT = os.path.join(HERE, "report", "p15_foot85_95")
ANGLES = (85.0, 95.0)


def foot_window_estimate(ts, zs, u, p1, p20, angles=ANGLES):
    """Axis station of the highest contour point among stations hit from P20
    at angles in `angles` (deg, measured from the axis). Returns (t_est, z_est, t_foot, window)."""
    lat = np.array([-u[1], u[0]])
    rel = p20[:2] - p1[:2]
    t20, w20 = rel @ u, abs(rel @ lat)
    th = np.radians(np.linspace(angles[0], angles[1], 201))
    tt = t20 + w20 / np.tan(th)
    lo, hi = tt.min(), tt.max()
    seg = np.where((ts >= lo - 0.5) & (ts <= hi + 0.5) & ~np.isnan(zs))[0]
    i = seg[np.argmax(zs[seg])]
    return ts[i], zs[i], t20, (lo, hi)


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.family"] = "Malgun Gothic"
    plt.rcParams["axes.unicode_minus"] = False
    os.makedirs(OUT, exist_ok=True)
    errs = []
    for lnd_path in sorted(glob.glob(os.path.join(ALIGNED_DIR, "*_aligned.lnd"))):
        name = os.path.basename(lnd_path)[:-12]
        L = np.loadtxt(os.path.join(os.path.dirname(ALIGNED_DIR), "lnd", name + ".lnd"))[:, 1:]
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
        t_est, z_est, t20, (wlo, whi) = foot_window_estimate(ts, zs, u, P[1], P[20])
        t15 = (P[15][:2] - P[1][:2]) @ u
        est3 = np.array([*(P[1][:2] + t_est * u), z_est])
        foot3 = np.array([*(P[1][:2] + t20 * u), zs[np.argmin(np.abs(ts - t20))]])
        win = np.array([[*(P[1][:2] + w * u), zs[np.argmin(np.abs(ts - w))]] for w in (wlo, whi)])
        err = t15 - t_est
        errs.append(err)

        fig, axes = plt.subplots(1, 3, figsize=(20, 6.6))
        views = [("XY (top)", 0, 1, "X", "Y"), ("YZ (side)", 1, 2, "Y", "Z"), ("ZX (front)", 2, 0, "Z", "X")]
        for ax, (title, a, b, la, lb) in zip(axes, views):
            ax.scatter(crop[:, a], crop[:, b], s=1.5, color="#cfcfcf", zorder=1)
            # axis line P1 -> P15 extended
            seg = np.array([P[1][:2] - 5 * u, P[1][:2] + 75 * u])
            seg3 = np.c_[seg, np.zeros(2)]
            if a != 2 and b != 2:
                ax.plot(seg3[:, a], seg3[:, b], color="#777", lw=0.8, zorder=2, label="엄지 축 (P1→P15)")
            # perpendicular from P20 to its foot
            ax.plot([P[20][a], foot3[a]], [P[20][b], foot3[b]], color="#00b8d4", lw=1.0, ls="--", zorder=3, label="P20 → 축 수선")
            ax.plot(win[:, a], win[:, b], color="#d81b60", lw=4, alpha=0.6, zorder=3, label=f"85~95° 창 (발 ±{(whi - wlo) / 2:.1f}mm)")
            for k, col in MARKS.items():
                ax.scatter(P[k][a], P[k][b], s=110, color=col, edgecolor="k", zorder=4)
                ax.annotate(str(k), (P[k][a], P[k][b]), xytext=(6, 5), textcoords="offset points", fontsize=10, fontweight="bold", zorder=5)
            ax.scatter(foot3[a], foot3[b], marker="x", s=90, color="#00838f", zorder=6, linewidths=2, label=f"× 수직 발 (t={t20:.1f})")
            ax.scatter(est3[a], est3[b], marker="*", s=260, color="#d81b60", edgecolor="k", zorder=7, label=f"★ 추정 P15 (t={t_est:.0f}), 정답 P15 t={t15:.1f}")
            ax.set_title(title)
            ax.set_xlabel(la)
            ax.set_ylabel(lb)
            ax.set_aspect("equal")
        ok = ~np.isnan(zs)
        axes[1].plot(P[1][1] + ts[ok] * u[1], zs[ok], color="#333333", lw=1.2, zorder=3, label=f"엄지 축 띠(±{BAND:.0f}mm) 하단 컨투어")
        axes[0].legend(loc="upper left", fontsize=8)
        axes[1].legend(loc="upper right", fontsize=8)
        fig.suptitle(f"{name}: P15 추정 = P20 수직 발 85~95° 창 안의 컨투어 최고점   |   정답 P15 - 추정 = {err:+.1f}mm (축 방향)   "
                     f"|   정답 축·정답 P20 사용 (규칙의 상한)", fontsize=12)
        fig.tight_layout(rect=(0, 0, 1, 0.95))
        fig.savefig(os.path.join(OUT, f"{name}.png"), dpi=110)
        plt.close(fig)
        print(f"{name}: foot t={t20:.1f} window [{wlo:.1f},{whi:.1f}] est t={t_est:.0f}  P15 t={t15:.1f}  err {err:+.1f}")
    e = np.array(errs)
    print(f"\n15 hands: mean {e.mean():+.2f} sd {e.std():.2f} max|.| {np.abs(e).max():.2f}  <=1mm {(np.abs(e) <= 1).sum()}/15  <=2mm {(np.abs(e) <= 2).sum()}/15")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
