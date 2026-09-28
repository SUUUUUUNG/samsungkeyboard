"""Figures for the participant self-measurement guide: the paper version of the final pipeline
(band P1 -> axis = P1-V line rotated 15 deg -> S = foot of V on the axis). One scanned hand (0004) in
the hand frame (middle finger = up) stands in for a photocopy print.
Writes thumb_model/report/self_guide/fig1..fig5.png and the printable 15-degree template (fig_template.png).

    python thumb_model/self_guide_figs.py
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)
import rule_pipeline as rp  # noqa: E402
from align_landmarks import OBJ_DIR, read_obj, sample_surface  # noqa: E402
from final_pipeline import BAND_Y, TIP_RADIUS  # noqa: E402

OUT = os.path.join(HERE, "report", "self_guide")
NAME = "20_F_0004G"
PHI = 15.0
INK = "#1f2a44"
SKIN = "#d3cec4"
ORANGE = "#e2571c"
GREEN = "#0f8a6f"
PURPLE = "#6d3fc0"
GOLD = "#c98a00"
BLUE = "#2b6fd6"


def unit(v):
    v = np.asarray(v, float)
    return v / np.linalg.norm(v)


def rot(v, deg):
    c, s = np.cos(np.radians(deg)), np.sin(np.radians(deg))
    return np.array([c * v[0] - s * v[1], s * v[0] + c * v[1]])


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon, Rectangle
    plt.rcParams["font.family"] = "Malgun Gothic"
    plt.rcParams["axes.unicode_minus"] = False
    os.makedirs(OUT, exist_ok=True)

    V, F, _ = read_obj(os.path.join(OBJ_DIR, NAME + ".obj"))
    r = rp.run(V, F)
    fr = r["frame"]
    S = np.vstack([V, sample_surface(V, F)])
    Sw = fr.to_work(S)[:, :2]
    tip0 = fr.to_work(r["P1"])[0][:2]
    Vn = fr.to_work(r["P20"])[0][:2]
    P26, P27 = fr.to_work(r["P26"])[0][:2], fr.to_work(r["P27"])[0][:2]
    P3 = fr.to_work(r["P3"])[0][:2]
    # band P1 on paper: topmost point of the thumb tip, points within 1 mm below it, outermost on the thumb side (+x here)
    near = Sw[np.linalg.norm(Sw - tip0, axis=1) < TIP_RADIUS]
    ytop = near[:, 1].max()
    band = near[near[:, 1] > ytop - BAND_Y]
    P1 = band[np.argmax(band[:, 0])]
    topmost = band[np.argmax(band[:, 1])]
    # axis: P1 -> V rotated by 15 deg away from the index (towards +x side here); choose the sign that moves the line away from V's side
    v = unit(Vn - P1)
    cand = [rot(v, PHI), rot(v, -PHI)]
    lat = np.array([-v[1], v[0]])
    outer_sign = np.sign(np.mean(near[:, :2] @ lat) - P1 @ lat)      # which side of the P1-V line the thumb body lies on... use tip region centroid
    u = cand[0] if np.sign(cand[0] @ lat) == outer_sign else cand[1]
    Spt = P1 + ((Vn - P1) @ u) * u
    length = np.linalg.norm(Spt - P1)
    thumb_deg = np.degrees(np.arctan2(abs(u[0]), abs(u[1])))
    print(f"{NAME}: P1=({P1[0]:.1f},{P1[1]:.1f}) V=({Vn[0]:.1f},{Vn[1]:.1f}) S=({Spt[0]:.1f},{Spt[1]:.1f}) length {length:.1f} mm, thumb axis {thumb_deg:.0f} deg from vertical")
    lo, hi = Sw.min(0), Sw.max(0)

    def paper(ax, extra_bottom=70, extra_top=14):
        ax.scatter(Sw[:, 0], Sw[:, 1], s=1.6, color=SKIN, zorder=1)
        ax.set_aspect("equal")
        ax.set_xlim(lo[0] - 22, hi[0] + 22)
        ax.set_ylim(lo[1] - extra_bottom, hi[1] + extra_top)
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_color("#9a9891")

    def dot(ax, p, col, label, dx=4, dy=4, size=140, fs=13, ha="left"):
        ax.scatter(*p, s=size, color=col, edgecolor="white", linewidth=1.5, zorder=8)
        ax.annotate(label, p, xytext=(dx, dy), textcoords="offset points", fontsize=fs, fontweight="bold", color=col, zorder=9, ha=ha)

    # ---------------- fig1: placement
    fig, ax = plt.subplots(figsize=(6.4, 8.2))
    paper(ax)
    ax.plot([P3[0], P3[0]], [lo[1] - 60, hi[1] + 8], "--", color=BLUE, lw=1.3, zorder=3)
    ax.annotate("① 가운데손가락을 종이 긴 변과 나란히(세로)", (P3[0], hi[1] + 9), ha="center", fontsize=10, color=BLUE)
    a1 = P1 + 62 * u
    ax.plot([P1[0], a1[0]], [P1[1], a1[1]], color=GOLD, lw=1.6, zorder=4)
    ax.plot([P1[0], P1[0]], [P1[1], P1[1] - 62], ":", color=GOLD, lw=1.4, zorder=4)
    th = np.linspace(-np.pi / 2, np.arctan2(u[1], u[0]), 30)
    ax.plot(P1[0] + 26 * np.cos(th), P1[1] + 26 * np.sin(th), color=GOLD, lw=1.4, zorder=4)
    ax.annotate(f"③ 엄지는 검지와 자연스럽게 벌리기\n   세로선에서 30~45° (이 예시 {thumb_deg:.0f}°)", (P1[0] - 2, P1[1] - 42), ha="right", fontsize=10, color=GOLD)
    ax.annotate("② 손가락 사이가 모두\n   보이게 벌리기", (lo[0] - 4, hi[1] - 30), ha="right", fontsize=10, color=INK)
    ax.plot([P26[0], P27[0]], [P26[1], P27[1]], "--", color=INK, lw=1.2, zorder=4)
    ax.annotate("④ 손목까지 포함", (0.5 * (P26[0] + P27[0]), P26[1] - 6), ha="center", va="top", fontsize=10, color=INK)
    card = Rectangle((lo[0] + 4, lo[1] - 66), 85.6, 54.0, fill=False, ec=GREEN, lw=1.8, zorder=5)
    ax.add_patch(card)
    ax.annotate("⑤ 신용카드를 손 옆에 같이 올리기\n   (출력물에서 가로 85.6mm인지 확인 = 배율 검사)", (lo[0] + 4 + 42.8, lo[1] - 66 + 27), ha="center", va="center", fontsize=9.5, color=GREEN)
    ax.set_title("손바닥을 유리에 평평하게 눌러 붙인 상태 (출력물 예시)", loc="left", fontsize=11, color=INK)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig1_placement.png"), dpi=150)
    plt.close(fig)

    # ---------------- fig2: P1 (band rule) and V
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.6), gridspec_kw=dict(width_ratios=[1, 1]))
    ax = axes[0]
    ax.scatter(Sw[:, 0], Sw[:, 1], s=3, color=SKIN, zorder=1)
    ax.set_aspect("equal")
    ax.set_xlim(tip0[0] - 16, tip0[0] + 12)
    ax.set_ylim(tip0[1] - 18, tip0[1] + 8)
    ax.set_xticks([]); ax.set_yticks([])
    ax.axhline(ytop, color=BLUE, lw=1.4, zorder=4)
    ax.axhline(ytop - BAND_Y, color=BLUE, lw=1.0, ls="--", zorder=4)
    ax.scatter(band[:, 0], band[:, 1], s=14, color=BLUE, zorder=5)
    ax.annotate("① 엄지 끝의 가장 위 점에 닿게 가로선", (tip0[0] - 15.5, ytop + 1.2), fontsize=9.5, color=BLUE)
    ax.annotate("② 그 선에서 1mm 아래까지의 윤곽 (파란 구간)", (tip0[0] - 15.5, ytop - 4.2), fontsize=9.5, color=BLUE)
    dot(ax, P1, ORANGE, "③ P1 = 파란 구간의\n    엄지 바깥쪽 끝점", dx=-2, dy=-30, ha="right", fs=11)
    ax.scatter(*topmost, s=60, marker="^", color="#898781", edgecolor="k", zorder=6)
    ax.annotate("가장 위 점", topmost, xytext=(4, 4), textcoords="offset points", fontsize=9, color="#898781")
    ax.set_title("P1 (엄지 끝) 찍기 — 엄지 끝 부분 확대", loc="left", fontsize=11, color=INK)
    ax = axes[1]
    ax.scatter(Sw[:, 0], Sw[:, 1], s=2, color=SKIN, zorder=1)
    ax.set_aspect("equal")
    ax.set_xlim(Vn[0] - 34, Vn[0] + 30)
    ax.set_ylim(Vn[1] - 26, Vn[1] + 40)
    ax.set_xticks([]); ax.set_yticks([])
    dot(ax, Vn, GREEN, "V = 엄지와 검지 사이\n골짜기의 가장 깊은 점", dx=8, dy=-24)
    ax.set_title("V (골점) 찍기", loc="left", fontsize=11, color=INK)
    for a_ in axes:
        for s in a_.spines.values():
            s.set_color("#9a9891")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig2_points.png"), dpi=150)
    plt.close(fig)

    # ---------------- fig3: axis by rotation + S
    fig, ax = plt.subplots(figsize=(6.6, 7.2))
    ax.scatter(Sw[:, 0], Sw[:, 1], s=2.2, color=SKIN, zorder=1)
    ax.set_aspect("equal")
    ax.set_xlim(min(P1[0], Spt[0]) - 30, max(P1[0], Vn[0]) + 26)
    ax.set_ylim(Spt[1] - 24, P1[1] + 14)
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_color("#9a9891")
    ax.plot([P1[0], Vn[0]], [P1[1], Vn[1]], "--", color=GREEN, lw=1.8, zorder=4)
    ax.annotate("① P1과 V를 잇는 선(파선)", (0.35 * P1[0] + 0.65 * Vn[0] - 4, 0.35 * P1[1] + 0.65 * Vn[1] + 8), fontsize=10, color=GREEN, ha="right")
    # template wedge at P1
    L_t = 60.0
    w0 = P1 + L_t * v
    w1 = P1 + L_t * u
    wedge = Polygon([P1, w0, w1], closed=True, fc="#fff3b0", ec=GOLD, lw=1.4, alpha=0.75, zorder=3)
    ax.add_patch(wedge)
    th0, th1 = np.arctan2(v[1], v[0]), np.arctan2(u[1], u[0])
    tt = np.linspace(th0, th1, 20)
    ax.plot(P1[0] + 16 * np.cos(tt), P1[1] + 16 * np.sin(tt), color=GOLD, lw=1.4, zorder=6)
    mid = 0.5 * (th0 + th1)
    ax.annotate("② 15° 각도판의 꼭짓점을 P1에,\n   한 변을 P1–V 선에 맞춤", (P1[0] + 22 * np.cos(mid), P1[1] + 22 * np.sin(mid)), fontsize=10, color=GOLD)
    a1 = P1 + 66 * u
    ax.plot([P1[0], a1[0]], [P1[1], a1[1]], color=INK, lw=2.4, zorder=5)
    ax.annotate("③ 다른 변을 따라 그은 선 = 엄지 축\n   (엄지 바깥쪽으로 15°)", (P1[0] + 40 * u[0] + 4, P1[1] + 40 * u[1] - 2), fontsize=10, color=INK)
    ax.plot([Vn[0], Spt[0]], [Vn[1], Spt[1]], color=PURPLE, lw=2, zorder=5)
    latu = np.array([-u[1], u[0]])
    sgn = np.sign((Vn - Spt) @ latu)
    c0 = Spt + 4 * u; c1 = c0 + sgn * 4 * latu; c2 = Spt + sgn * 4 * latu
    ax.plot([c0[0], c1[0], c2[0]], [c0[1], c1[1], c2[1]], color=PURPLE, lw=1.2, zorder=6)
    ax.annotate("④ V에서 엄지 축에 직각으로\n    내린 점 = S", (Spt[0] + 6, Spt[1] - 12), fontsize=10, color=PURPLE, ha="left")
    dot(ax, P1, ORANGE, "P1", dx=-16, dy=6)
    dot(ax, Vn, GREEN, "V", dx=8, dy=-14)
    dot(ax, Spt, PURPLE, "S", dx=-16, dy=-6)
    ax.set_title("엄지 축과 시작점 S 만들기", loc="left", fontsize=11, color=INK)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig3_construction.png"), dpi=150)
    plt.close(fig)

    # ---------------- fig4: measurement
    fig, ax = plt.subplots(figsize=(6.4, 5.6))
    ax.scatter(Sw[:, 0], Sw[:, 1], s=2.2, color=SKIN, zorder=1)
    ax.set_aspect("equal")
    ax.set_xlim(min(P1[0], Spt[0]) - 30, max(P1[0], Vn[0]) + 30)
    ax.set_ylim(Spt[1] - 18, P1[1] + 12)
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_color("#9a9891")
    n_tick = int(length) + 6
    base = P1 - 3 * u
    rw = 8.0
    e0, e1 = base, base + n_tick * u
    poly = np.array([e0, e1, e1 - rw * latu, e0 - rw * latu])
    ax.fill(poly[:, 0], poly[:, 1], color="#fbeaa6", alpha=0.75, zorder=5, ec="#7a5c00", lw=0.8)
    for i in range(n_tick + 1):
        p = base + i * u
        h = 4.5 if i % 10 == 0 else (3.0 if i % 5 == 0 else 1.8)
        q = p - h * latu
        ax.plot([p[0], q[0]], [p[1], q[1]], color="#7a5c00", lw=0.9 if i % 5 == 0 else 0.6, zorder=6)
        if i % 10 == 0:
            ax.annotate(str(i // 10), (q[0], q[1]), xytext=(0, -9), textcoords="offset points", fontsize=8, color="#7a5c00", ha="center", zorder=7)
    ax.plot([P1[0], Spt[0]], [P1[1], Spt[1]], color=ORANGE, lw=2.4, zorder=8)
    dot(ax, P1, ORANGE, "P1", dx=-16, dy=8)
    dot(ax, Spt, PURPLE, "S", dx=6, dy=-14)
    ax.annotate(f"P1–S 직선거리 = 엄지 길이\n(이 예시 {length:.1f} mm)", (0.5 * (P1[0] + Spt[0]) + 14, 0.5 * (P1[1] + Spt[1]) + 10), fontsize=11, color=ORANGE, fontweight="bold")
    ax.set_title("자로 재기: P1에서 S까지, 0.5mm 단위", loc="left", fontsize=11, color=INK)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig4_measure.png"), dpi=150)
    plt.close(fig)

    # ---------------- template: 15-degree wedge, printed at 100 % (angle is scale-free anyway)
    fig = plt.figure(figsize=(160 / 25.4, 70 / 25.4))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 160); ax.set_ylim(0, 70); ax.set_aspect("equal"); ax.axis("off")
    apex = np.array([12.0, 12.0])
    e1 = apex + 140.0 * np.array([1.0, 0.0])
    e2 = apex + 140.0 * np.array([np.cos(np.radians(PHI)), np.sin(np.radians(PHI))])
    ax.add_patch(Polygon([apex, e1, e2], closed=True, fc="#fff3b0", ec=INK, lw=1.6))
    ax.scatter(*apex, s=40, color=INK, zorder=5)
    tt = np.linspace(0, np.radians(PHI), 20)
    ax.plot(apex[0] + 30 * np.cos(tt), apex[1] + 30 * np.sin(tt), color=INK, lw=1)
    ax.text(apex[0] + 34, apex[1] + 4.5, "15°", fontsize=14, color=INK, fontweight="bold")
    ax.text(apex[0] - 2, apex[1] - 7, "꼭짓점 → P1에 맞춤", fontsize=10, color=INK)
    ax.text(apex[0] + 60, apex[1] - 4.5, "이 변 → P1–V 선에 맞춤", fontsize=10, color=INK)
    ax.text(apex[0] + 52, apex[1] + 24, "이 변을 따라 선 긋기 = 엄지 축", fontsize=10, color=INK, rotation=PHI)
    ax.text(2, 64, "15° 각도판 — 선을 따라 오려서 사용 (각도는 인쇄 배율과 무관)", fontsize=10.5, color=INK)
    fig.savefig(os.path.join(OUT, "fig_template.png"), dpi=200)
    plt.close(fig)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
