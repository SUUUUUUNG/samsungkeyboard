"""Explainer figure: how the thumb-axis band lower contour is built (one hand).

Output: thumb_model/report/thumb_views/_how_contour_is_built.png
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)
from align_landmarks import OUT_DIR as ALIGNED_DIR, OBJ_DIR, read_obj, sample_surface  # noqa: E402
from thumb_views import BAND, OUT, contact_min_of, lower_contour, peak_of  # noqa: E402

HAND = "20_F_0004G"
T_EXAMPLE = 44.0


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.family"] = "Malgun Gothic"
    plt.rcParams["axes.unicode_minus"] = False
    os.makedirs(OUT, exist_ok=True)
    L = np.loadtxt(os.path.join(os.path.dirname(ALIGNED_DIR), "lnd", HAND + ".lnd"))[:, 1:]
    T = np.loadtxt(os.path.join(ALIGNED_DIR, f"{HAND}_transform.txt"), comments="#")
    Ti = np.linalg.inv(T)
    V, F, _ = read_obj(os.path.join(OBJ_DIR, HAND + ".obj"))
    S = np.vstack([V, sample_surface(V, F)])
    S = (np.c_[S, np.ones(len(S))] @ Ti.T)[:, :3]
    P = {k: L[k - 1] for k in (1, 6, 15, 20)}
    u = P[15][:2] - P[1][:2]
    u /= np.linalg.norm(u)
    lat = np.array([-u[1], u[0]])
    rel = S[:, :2] - P[1][:2]
    t, w = rel @ u, rel @ lat
    band = (np.abs(w) < BAND) & (t > -5) & (t < 80)
    ts, zs, _ = lower_contour(S, P[1], P[15])
    tp, zp = peak_of(ts, zs)
    tm, zm = contact_min_of(ts, zs)
    ex = band & (np.abs(t - T_EXAMPLE) < 0.75)

    fig, ax = plt.subplots(1, 2, figsize=(18, 7.5))
    crop = (S[:, 0] > -5) & (S[:, 0] < 62) & (S[:, 1] > 40) & (S[:, 1] < 118)
    ax[0].scatter(S[crop & ~band, 0], S[crop & ~band, 1], s=1, color="#d9d9d9", label="띠 밖의 점 (사용 안 함)")
    ax[0].scatter(S[band, 0], S[band, 1], s=1.5, color="#8fb8e8", label=f"띠 안의 점 (|w| < {BAND:.0f}mm)")
    ax[0].scatter(S[ex, 0], S[ex, 1], s=14, color="#d62728", zorder=5,
                  label=f"예: t = {T_EXAMPLE:.0f}mm 판(두께 1.5mm)의 점 {ex.sum()}개")
    for tt in np.arange(0, 80, 5):
        c = P[1][:2] + tt * u
        a, b = c + BAND * lat, c - BAND * lat
        ax[0].plot([a[0], b[0]], [a[1], b[1]], color="#555", lw=0.6)
        if tt % 10 == 0:
            ax[0].text(*(c - (BAND + 3) * lat), f"t={tt:.0f}", fontsize=7, color="#555", ha="center")
    for s in (-BAND, BAND):
        a = P[1][:2] - 5 * u + s * lat
        b = P[1][:2] + 80 * u + s * lat
        ax[0].plot([a[0], b[0]], [a[1], b[1]], "k--", lw=0.8)
    a, b = P[1][:2] - 5 * u, P[1][:2] + 80 * u
    ax[0].plot([a[0], b[0]], [a[1], b[1]], "k-", lw=1.2, label="엄지 축 (P1→P15 직선)")
    for k, c in ((1, "#ff2b2b"), (6, "#ff9f1c"), (15, "#39ff14"), (20, "#00e5ff")):
        ax[0].scatter(P[k][0], P[k][1], s=110, color=c, edgecolor="k", zorder=6)
        ax[0].annotate(str(k), (P[k][0], P[k][1]), xytext=(6, 5), textcoords="offset points", fontweight="bold")
    ax[0].set_aspect("equal")
    ax[0].set_xlabel("X")
    ax[0].set_ylabel("Y")
    ax[0].set_title("① 위에서 본 모습: 축·띠·판(스테이션)")
    ax[0].legend(loc="upper right", fontsize=8)

    ax[1].scatter(t[band], S[band, 2], s=1, color="#8fb8e8", label="띠 안의 모든 점 (t, z)")
    ax[1].scatter(t[ex], S[ex, 2], s=14, color="#d62728", zorder=5, label=f"t = {T_EXAMPLE:.0f} 판의 점들")
    ax[1].scatter([T_EXAMPLE], [S[ex, 2].min()], s=90, marker="s", color="#d62728", edgecolor="k", zorder=6,
                  label="그 판의 최저 z = 곡선의 한 점")
    ax[1].plot(ts, zs, "k-", lw=1.6, zorder=4, label="하단 컨투어 = 판마다 최저 z를 이은 선")
    ax[1].scatter([tp], [zp], marker="^", s=110, color="k", zorder=7, label=f"▲ 들림 봉우리 꼭대기 (t={tp:.0f})")
    ax[1].scatter([tm], [zm], marker="v", s=110, color="k", zorder=7, label=f"▼ 접촉 최저점 (t={tm:.0f})")
    for k, c in ((1, "#ff2b2b"), (6, "#ff9f1c"), (15, "#39ff14")):
        tk = (P[k][:2] - P[1][:2]) @ u
        ax[1].axvline(tk, color=c, lw=1.2)
        ax[1].text(tk, 33, f"P{k}", color="k", ha="center", fontsize=9, fontweight="bold")
    ax[1].set_xlim(-5, 80)
    ax[1].set_ylim(-1, 35)
    ax[1].set_xlabel("t = 축을 따라 P1에서 잰 거리 (mm, 손목 쪽 +)")
    ax[1].set_ylabel("Z (mm, 유리판 = 0)")
    ax[1].set_title("② 축 방향으로 펼친 모습: 띠 안 점들의 (t, z)와 하단 컨투어")
    ax[1].legend(loc="upper right", fontsize=8)
    fig.suptitle(f"{HAND}: 엄지 축 띠(±{BAND:.0f}mm) 하단 컨투어를 만드는 과정", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(os.path.join(OUT, "_how_contour_is_built.png"), dpi=110)
    print(f"band points {band.sum()}, example slab points {ex.sum()}, slab min z {S[ex, 2].min():.2f}; "
          f"peak t={tp:.0f} z={zp:.1f}, contact min t={tm:.0f} z={zm:.1f}")


if __name__ == "__main__":
    main()
