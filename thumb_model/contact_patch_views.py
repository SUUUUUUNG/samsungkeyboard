"""Contact patch of the thumb: points within 0.3 / 0.5 / 1.0 mm of the thumb's own
lowest point (local floor, not the palm heel), top view with P1 / P6 / P15 / P20
(lnd frame). Prints where P6 sits relative to the patch (extent along the axis,
centroid, proximal-most point, min-y / max-x candidates).

Output: thumb_model/report/contact_patch/<name>.png
"""
import glob
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from align_landmarks import OUT_DIR as ALIGNED_DIR, OBJ_DIR, read_obj, sample_surface  # noqa: E402

OUT = os.path.join(HERE, "report", "contact_patch")
MARGIN = 15.0
MARKS = {1: "#ff2b2b", 6: "#ff9f1c", 15: "#39ff14", 20: "#00e5ff"}
LAYERS = ((0.3, "#08306b"), (0.5, "#2171b5"), (1.0, "#9ecae1"))   # z above local floor, colour
BAND = 12.0     # lateral half-width around the thumb axis (mm)
T_RANGE = (3.0, 45.0)   # along-axis window from P1 in which the thumb-tip patch is searched


def unit(v):
    v = np.asarray(v, float)
    return v / np.linalg.norm(v)


def thumb_patch(S, p1, u, dz, band=BAND, t_range=T_RANGE):
    """points of the distal contact patch: within dz of the thumb's local floor, inside the band, first run along t (gap > 3 mm ends it)."""
    lat = np.array([-u[1], u[0]])
    rel = S[:, :2] - p1[:2]
    t, w = rel @ u, rel @ lat
    inband = (np.abs(w) < band) & (t > t_range[0]) & (t < t_range[1])
    zfloor = S[inband, 2].min()
    m = inband & (S[:, 2] < zfloor + dz)
    idx = np.where(m)[0]
    if len(idx) < 5:
        return None, zfloor
    order = idx[np.argsort(t[idx])]
    tt = t[order]
    gaps = np.where(np.diff(tt) > 3.0)[0]
    keep = order[: gaps[0] + 1] if len(gaps) else order
    return {"idx": keep, "t": t[keep], "w": w[keep], "P": S[keep]}, zfloor


def candidates(patch):
    """xy candidate points for P6 from the patch."""
    P, t, w = patch["P"], patch["t"], patch["w"]
    c = {
        "centroid": P[:, :2].mean(0),
        "max-t (proximal-most)": P[np.argmax(t), :2],
        "min-y": P[np.argmin(P[:, 1]), :2],
        "max-x": P[np.argmax(P[:, 0]), :2],
    }
    # proximal edge on the axis: among the 20 % most proximal patch points, the one closest to the axis
    k = t >= np.quantile(t, 0.8)
    sub = np.where(k)[0]
    c["proximal edge, on axis"] = P[sub[np.argmin(np.abs(w[sub]))], :2]
    return c


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.family"] = "Malgun Gothic"
    plt.rcParams["axes.unicode_minus"] = False
    os.makedirs(OUT, exist_ok=True)
    rows = []
    for lnd_path in sorted(glob.glob(os.path.join(ALIGNED_DIR, "*_aligned.lnd"))):
        name = os.path.basename(lnd_path)[:-12]
        L = np.loadtxt(os.path.join(os.path.dirname(ALIGNED_DIR), "lnd", name + ".lnd"))[:, 1:]
        T = np.loadtxt(os.path.join(ALIGNED_DIR, f"{name}_transform.txt"), comments="#")
        Ti = np.linalg.inv(T)
        V, F, _ = read_obj(os.path.join(OBJ_DIR, name + ".obj"))
        S = np.vstack([V, sample_surface(V, F)])
        S = (np.c_[S, np.ones(len(S))] @ Ti.T)[:, :3]
        P = {k: L[k - 1] for k in MARKS}
        u = unit(P[15][:2] - P[1][:2])
        lat = np.array([-u[1], u[0]])
        t6, t15 = (P[6][:2] - P[1][:2]) @ u, (P[15][:2] - P[1][:2]) @ u
        w6 = (P[6][:2] - P[1][:2]) @ lat
        patches = {dz: thumb_patch(S, P[1], u, dz)[0] for dz, _ in LAYERS}
        zfloor = thumb_patch(S, P[1], u, 1.0)[1]
        rows.append((name, t6, w6, t15, patches, u, lat, P[1]))

        pts = np.array(list(P.values()))
        lo, hi = pts.min(0) - MARGIN, pts.max(0) + MARGIN
        crop = S[(S[:, 0] > lo[0]) & (S[:, 0] < hi[0]) & (S[:, 1] > lo[1]) & (S[:, 1] < hi[1])]
        fig, ax = plt.subplots(figsize=(9, 8.5))
        ax.scatter(crop[:, 0], crop[:, 1], s=1.2, color="#dedede", zorder=1, label="엄지 영역 점")
        for dz, col in LAYERS[::-1]:
            m = crop[:, 2] < zfloor + dz
            ax.scatter(crop[m, 0], crop[m, 1], s=3, color=col, zorder=2)
        for dz, col in LAYERS:
            ax.scatter([], [], s=20, color=col, label=f"엄지 최저점에서 {dz:.1f}mm 이내 (z<{zfloor + dz:.1f})")
        pc = patches[1.0]
        if pc is not None:
            ax.scatter(pc["P"][:, 0], pc["P"][:, 1], s=6, facecolor="none", edgecolor="#e41a1c", lw=0.4, zorder=3, label="1.0mm 패치 (축 ±12mm, 첫 덩어리)")
            for lab, q in candidates(pc).items():
                ax.scatter(q[0], q[1], marker="x", s=60, color="k", zorder=6)
                ax.annotate(lab, q, xytext=(6, 4), textcoords="offset points", fontsize=7.5, zorder=6)
        for k, col in MARKS.items():
            ax.scatter(P[k][0], P[k][1], s=130, color=col, edgecolor="k", zorder=5)
            ax.annotate(str(k), (P[k][0], P[k][1]), xytext=(7, 6), textcoords="offset points", fontsize=11, fontweight="bold", zorder=6)
        a, b = P[1][:2] - 5 * u, P[1][:2] + 70 * u
        ax.plot([a[0], b[0]], [a[1], b[1]], color="#555", lw=0.8, zorder=3)
        ax.set_aspect("equal")
        ax.set_xlabel("X (lnd 좌표계)")
        ax.set_ylabel("Y")
        title = f"{name}: 엄지 접촉 패치 (엄지 자체 최저점 z={zfloor:.2f} 기준)   |   P6 t={t6:.1f}, P15 t={t15:.1f}"
        if pc is not None:
            title += f"   |   1mm 패치 t {pc['t'].min():.0f}~{pc['t'].max():.0f}"
        ax.set_title(title, fontsize=9.5, loc="left")
        ax.legend(loc="upper right", fontsize=8)
        fig.tight_layout()
        fig.savefig(os.path.join(OUT, f"{name}.png"), dpi=110)
        plt.close(fig)

    # ---- numbers: P6 relative to the patch, per layer -------------------------------------------------
    for dz, _ in LAYERS:
        print(f"\n=== layer z < local floor + {dz:.1f} mm ===")
        print(f"{'hand':8s} {'t6':>5s} {'w6':>5s} | patch t: start  end  ctr   n | P6 - end | P6 - ctr |  P6 xy error of candidates: " + "  ".join(f"{k[:10]:>10s}" for k in ("centroid", "max-t", "min-y", "max-x", "prox-edge")))
        stats = {"P6 - patch end (along)": [], "P6 - patch centroid (along)": []}
        cand_err = {}
        for name, t6, w6, t15, patches, u, lat, p1 in rows:
            pc = patches[dz]
            if pc is None:
                print(f"{name[3:9]:8s} {t6:5.1f} {w6:+5.1f} |  (no patch)")
                continue
            tt = pc["t"]
            stats["P6 - patch end (along)"].append(t6 - tt.max())
            stats["P6 - patch centroid (along)"].append(t6 - tt.mean())
            c = candidates(pc)
            p6xy = p1[:2] + t6 * u + w6 * lat
            errs = []
            for k, q in c.items():
                d = p6xy - q
                cand_err.setdefault(k, []).append((d @ u, d @ lat))
                errs.append(np.linalg.norm(d))
            print(f"{name[3:9]:8s} {t6:5.1f} {w6:+5.1f} | {tt.min():5.1f} {tt.max():5.1f} {tt.mean():5.1f} {len(tt):4d} | {t6 - tt.max():+7.2f} | {t6 - tt.mean():+7.2f} |  " + "  ".join(f"{e:10.2f}" for e in errs))
        for k, v in stats.items():
            v = np.array(v)
            print(f"  {k:30s} mean {v.mean():+6.2f} sd {v.std():5.2f} (n={len(v)})")
        print("  P6 - candidate, along-axis mean±sd / lateral mean±sd / xy mean:")
        for k, v in cand_err.items():
            v = np.array(v)
            print(f"    {k:26s} along {v[:, 0].mean():+6.2f} ± {v[:, 0].std():4.2f}   lateral {v[:, 1].mean():+6.2f} ± {v[:, 1].std():4.2f}   xy {np.linalg.norm(v, axis=1).mean():5.2f}")
    print("\nwrote", OUT)


if __name__ == "__main__":
    main()
