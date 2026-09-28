"""Three-panel explainer of the rule-only pipeline on one hand (obj frame):
(1) P1 band rule at the thumb tip, (2) axis P1 -> B (thumb-width midpoint 2 cm
distal of the P20 level) vs the SW axis, (3) lower contour along that axis with
the P20 foot window and the chosen P15.

    python thumb_model/rule_only_explainer.py [name]   -> thumb_model/report/rule_only_explainer.png
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)
from align_landmarks import OBJ_DIR, OUT_DIR as ALIGNED_DIR, read_obj  # noqa: E402
from final_pipeline import BAND_Y, TIP_RADIUS, axis_direction, rule_parts, rule_p15, unit  # noqa: E402
from p15_foot_window_views import foot_window_estimate  # noqa: E402
from thumb_views import lower_contour  # noqa: E402


def main(name="20_F_0004G"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.family"] = "Malgun Gothic"
    plt.rcParams["axes.unicode_minus"] = False

    V, F, _ = read_obj(os.path.join(OBJ_DIR, name + ".obj"))
    A = np.loadtxt(os.path.join(ALIGNED_DIR, f"{name}_aligned.lnd"))[:, 1:]
    g1, g15, g20 = A[0], A[14], A[19]
    parts = rule_parts(V, F)
    S, fr, b, p20 = parts["S"], parts["frame"], parts["B"][:2], parts["P20"]
    p1 = parts["P1_band1"]
    u = axis_direction(parts, "p1_to_B", p1)                  # axis = band P1 -> B (width midpoint 2 cm distal of the P20 level)
    u_true = unit(g15[:2] - g1[:2])
    ang = np.degrees(np.arctan2(u_true[0] * u[1] - u_true[1] * u[0], u_true @ u))
    p15, t_est, t20, w20, (ts, zs) = rule_p15(parts, p1, u)
    _, _, _, (lo, hi) = foot_window_estimate(ts, zs, u, p1, p20)
    t_true = float((g15[:2] - p1[:2]) @ u)

    fig, axes = plt.subplots(1, 3, figsize=(19, 6.6))
    # ---- panel 1: P1 band rule (tip region)
    ax = axes[0]
    near = S[np.linalg.norm(S[:, :2] - parts["tip_outline"][:2], axis=1) < TIP_RADIUS]
    work = fr.to_work(near)
    band = work[:, 1] > work[:, 1].max() - BAND_Y
    ax.scatter(near[~band, 0], near[~band, 1], s=2, color="#d5d4cf", label="엄지 끝 주변 점")
    ax.scatter(near[band, 0], near[band, 1], s=14, color="#2a78d6", label="후보: 손 좌표계 y최대점에서 1mm 아래까지")
    ax.scatter(*parts["apex"][:2], marker="s", s=70, color="#898781", edgecolor="k", zorder=5, label="중심선 끝(apex)")
    ax.scatter(*parts["maxy"][:2], marker="^", s=80, color="#898781", edgecolor="k", zorder=5, label="y최대점")
    ax.scatter(*parts["P1_rule"][:2], marker="o", s=70, color="#eb6834", edgecolor="k", zorder=5, label="중점 규칙 P1 (결합 파이프라인)")
    ax.scatter(*p1[:2], marker="*", s=260, color="#ff2b2b", edgecolor="k", zorder=6, label="띠 규칙 P1 = 후보 중 엄지 바깥쪽 끝점")
    ax.scatter(*g1[:2], marker="x", s=120, color="k", lw=2.2, zorder=7, label="SW 정답 P1")
    # hand-frame y direction
    ydir = fr.from_work(np.array([[0, 1, 0.0]]))[0][:2] - fr.from_work(np.array([[0, 0, 0.0]]))[0][:2]
    ydir = unit(ydir)
    o = parts["tip_outline"][:2] + np.array([-16, -14.0])
    ax.annotate("", xy=o + 10 * ydir, xytext=o, arrowprops=dict(arrowstyle="->", lw=1.4, color="#52514e"))
    ax.text(*(o + 10.8 * ydir), "손 좌표계 +y\n(손목 중심→중지 끝)", fontsize=8.5, color="#52514e")
    ax.set_aspect("equal")
    c = parts["tip_outline"][:2]
    ax.set_xlim(c[0] - 20, c[0] + 20)
    ax.set_ylim(c[1] - 24, c[1] + 12)
    ax.set_title(f"① P1 띠 규칙  |  {name}", loc="left", fontsize=11)
    ax.legend(fontsize=8, loc="lower left")
    ax.set_xlabel("x (obj 좌표계, mm)")
    ax.set_ylabel("y")

    # ---- panel 2: axis construction
    ax = axes[1]
    pts = np.array([g1[:2], g15[:2], p20[:2]])
    lo_xy, hi_xy = pts.min(0) - 18, pts.max(0) + 18
    crop = S[(S[:, 0] > lo_xy[0]) & (S[:, 0] < hi_xy[0]) & (S[:, 1] > lo_xy[1]) & (S[:, 1] < hi_xy[1])]
    ax.scatter(crop[:, 0], crop[:, 1], s=1.2, color="#dedede")
    lat_u = np.array([-u[1], u[0]])
    ax.plot([b[0] - 14 * lat_u[0], b[0] + 14 * lat_u[0]], [b[1] - 14 * lat_u[1], b[1] + 14 * lat_u[1]], color="#1a1a19", lw=1.4, label="P20′ 높이보다 2cm 위에서 잰 엄지 폭")
    ax.scatter(*b, marker="s", s=110, color="#ffd54a", edgecolor="k", zorder=6, label="B = 그 폭의 중앙점")
    a1 = p1[:2] + 62 * u
    ax.plot([p1[0], a1[0]], [p1[1], a1[1]], color="#1a9850", lw=2.2, label="규칙 축 = P1 → B")
    b1 = g1[:2] + 62 * u_true
    ax.plot([g1[0], b1[0]], [g1[1], b1[1]], color="k", lw=1.4, label=f"SW 축 (P1→P15), 각도 차 {ang:+.1f}°")
    ax.scatter(*p1[:2], marker="*", s=220, color="#ff2b2b", edgecolor="k", zorder=7, label="띠 규칙 P1")
    ax.scatter(*p20[:2], s=90, color="#00e5ff", edgecolor="k", zorder=6, label="P20′ (엄지–검지 오목점, 규칙)")
    footp20 = p1[:2] + t20 * u
    ax.plot([p20[0], footp20[0]], [p20[1], footp20[1]], ":", color="#00a0b0", lw=1.2)
    ax.scatter(*footp20, marker="x", s=70, color="#00a0b0", lw=2, zorder=6, label="P20′의 수직 발")
    ax.scatter(*g15[:2], marker="x", s=120, color="k", lw=2.2, zorder=7, label="SW 정답 P15")
    ax.scatter(*p15[:2], marker="*", s=220, color="#39ff14", edgecolor="k", zorder=7, label="규칙 P15 (③에서 결정)")
    ax.set_aspect("equal")
    ax.set_xlim(lo_xy[0], hi_xy[0])
    ax.set_ylim(lo_xy[1], hi_xy[1])
    ax.set_title("② 엄지 축: P1 → B (P20′ 높이 2cm 위 엄지 폭의 중앙점)", loc="left", fontsize=11)
    ax.legend(fontsize=7.6, loc="lower left")
    ax.set_xlabel("x (mm)")

    # ---- panel 3: contour along the axis with the foot window
    ax = axes[2]
    ax.plot(ts, zs, color="#2a78d6", lw=1.6, label="축 띠(±12mm) 밑면 컨투어: 1mm마다 최저 z")
    ax.axvspan(lo, hi, color="#f8c8d8", alpha=0.6, label=f"P20′ 발 기준 85~95° 창 (t {lo:.1f}~{hi:.1f})")
    ax.axvline(t20, color="#00a0b0", ls=":", lw=1.4, label=f"P20′ 수직 발 t={t20:.1f}")
    ax.scatter([t_est], [zs[np.argmin(np.abs(ts - t_est))]], marker="*", s=260, color="#39ff14", edgecolor="k", zorder=6, label=f"창 안 최고점 = 규칙 P15, t={t_est:.1f}")
    ax.axvline(t_true, color="k", lw=1.4, label=f"SW 정답 P15의 축 위 위치 t={t_true:.1f}")
    ax.set_xlim(0, 62)
    ax.set_xlabel("축 방향 거리 t (P1에서 손목 쪽으로, mm)")
    ax.set_ylabel("밑면 z (mm, 유리 = 0)")
    ax.set_title("③ 규칙 P15: 창 안 컨투어 최고점", loc="left", fontsize=11)
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(color="#e1e0d9")
    L = float(np.linalg.norm(p1[:2] - p15[:2]))
    Lt = float(np.linalg.norm(g1[:2] - g15[:2]))
    fig.suptitle(f"규칙 전용 파이프라인 한눈에 보기 ({name}): 길이 예측 {L:.1f}mm, SW 기준 {Lt:.1f}mm, 오차 {L - Lt:+.1f}mm", x=0.01, ha="left", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out = os.path.join(HERE, "report", "rule_only_explainer.png")
    fig.savefig(out, dpi=140)
    print("wrote", out)


if __name__ == "__main__":
    main(*sys.argv[1:2])
