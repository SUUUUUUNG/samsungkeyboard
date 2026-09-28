"""Figure set for the simplest rule-only pipeline (axis = band P1 -> P20' line rotated by the
SW's constant angle). Writes to thumb_model/report/simple/:

  sw_angle_consistency.png   the SW's own angle(P1->P15 vs P1->P20) per hand: 14.6 +/- 1.1 deg
  axis_explainer.png         one hand: P1, P20', the P1->P20' line, the rotated axis, the SW axis
  pipeline_explainer.png     one hand, three panels: P1 band rule / axis / contour window -> P15
  <name>.png                 per hand (15): top view with axis, window, rule P15 and the SW's P15

    python thumb_model/simple_explainer.py
"""
import glob
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)
from align_landmarks import OBJ_DIR, OUT_DIR as ALIGNED_DIR, REFERENCE_THUMB_MM, read_obj  # noqa: E402
from final_pipeline import BAND_Y, TIP_RADIUS, axis_direction, rule_p15, rule_parts, sw_axis_angle, unit  # noqa: E402
from p15_foot_window_views import foot_window_estimate  # noqa: E402

OUT = os.path.join(HERE, "report", "simple")
EXAMPLE = "20_F_0004G"


def signed_angle(a, b):
    return float(np.degrees(np.arctan2(a[0] * b[1] - a[1] * b[0], a @ b)))


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.family"] = "Malgun Gothic"
    plt.rcParams["axes.unicode_minus"] = False
    os.makedirs(OUT, exist_ok=True)

    names = [os.path.basename(f)[:-12] for f in sorted(glob.glob(os.path.join(ALIGNED_DIR, "*_aligned.lnd")))]
    A_all = {n: np.loadtxt(os.path.join(ALIGNED_DIR, f"{n}_aligned.lnd"))[:, 1:] for n in names}
    phis = np.array([sw_axis_angle(A_all[n]) for n in names])
    phi = float(phis.mean())

    # ---- per-hand figures (and the numbers for the consistency plot)
    rows = []
    for n in names:
        A = A_all[n]
        g1, g15, g20 = A[0], A[14], A[19]
        V, F, _ = read_obj(os.path.join(OBJ_DIR, n + ".obj"))
        parts = rule_parts(V, F)
        S, p1, p20 = parts["S"], parts["P1_band1"], parts["P20"]
        u = axis_direction(parts, "p20_rotated", p1, phi_deg=phi)
        p15, t_est, t20, w20, (ts, zs) = rule_p15(parts, p1, u)
        _, _, _, (lo, hi) = foot_window_estimate(ts, zs, u, p1, p20)
        u_true = unit(g15[:2] - g1[:2])
        ang = signed_angle(u_true, u)
        L, Lref = float(np.linalg.norm(p1[:2] - p15[:2])), REFERENCE_THUMB_MM[n[:-1]]
        rows.append((n, ang, L - Lref))
        if n == EXAMPLE:
            example = dict(parts=parts, p1=p1, p20=p20, u=u, p15=p15, ts=ts, zs=zs, t_est=t_est, t20=t20, lo=lo, hi=hi, g1=g1, g15=g15, g20=g20, ang=ang, L=L, Lref=Lref)

        fig, ax = plt.subplots(figsize=(8.5, 8))
        pts = np.array([g1[:2], g15[:2], p20[:2]])
        lo_xy, hi_xy = pts.min(0) - 18, pts.max(0) + 16
        crop = S[(S[:, 0] > lo_xy[0]) & (S[:, 0] < hi_xy[0]) & (S[:, 1] > lo_xy[1]) & (S[:, 1] < hi_xy[1])]
        ax.scatter(crop[:, 0], crop[:, 1], s=1.2, color="#e3e2dd", zorder=0)
        v = unit(p20[:2] - p1[:2])
        ax.plot([p1[0], p20[0]], [p1[1], p20[1]], color="#898781", lw=1.2, ls="--", zorder=2, label="P1 → P20′ 선")
        a1 = p1[:2] + 62 * u
        ax.plot([p1[0], a1[0]], [p1[1], a1[1]], color="#7a3fbf", lw=2.4, zorder=4, label=f"엄지 축 = P1→P20′ 선을 {phi:.1f}° 회전")
        b1 = g1[:2] + 62 * u_true
        ax.plot([g1[0], b1[0]], [g1[1], b1[1]], color="k", lw=1.3, zorder=4, label=f"SW 축 (정답 P1→P15), 각도 차 {ang:+.1f}°")
        wa, wb = p1[:2] + lo * u, p1[:2] + hi * u
        ax.plot([wa[0], wb[0]], [wa[1], wb[1]], color="#f06292", lw=7, alpha=0.6, zorder=3, label="P20′ 발 기준 85~95° 창")
        foot = p1[:2] + t20 * u
        ax.plot([p20[0], foot[0]], [p20[1], foot[1]], ":", color="#00a0b0", lw=1.2, zorder=3)
        ax.scatter(*p1[:2], marker="*", s=240, color="#ff2b2b", edgecolor="k", zorder=7, label="띠 규칙 P1")
        ax.scatter(*p20[:2], s=90, color="#00e5ff", edgecolor="k", zorder=6, label="P20′ (규칙)")
        ax.scatter(*p15[:2], marker="*", s=240, color="#39ff14", edgecolor="k", zorder=7, label="규칙 P15 (창 안 컨투어 최고점)")
        ax.scatter(*g1[:2], marker="x", s=110, color="k", lw=2, zorder=8, label="SW 정답 P1 / P15")
        ax.scatter(*g15[:2], marker="x", s=110, color="k", lw=2, zorder=8)
        ax.set_aspect("equal")
        ax.set_xlim(lo_xy[0], hi_xy[0])
        ax.set_ylim(lo_xy[1], hi_xy[1])
        ax.set_xlabel("x (obj 좌표계, mm)")
        ax.set_ylabel("y")
        ax.set_title(f"{n}: 최단순 축   |   길이 예측 {L:.1f}, SW 기준 {Lref:.1f}, 오차 {L - Lref:+.1f}mm", loc="left", fontsize=10.5)
        ax.legend(fontsize=8, loc="lower left", framealpha=0.95)
        fig.tight_layout()
        fig.savefig(os.path.join(OUT, f"{n}.png"), dpi=120)
        plt.close(fig)
        print(f"[{n}] angle {ang:+.1f} deg, length err {L - Lref:+.2f}", flush=True)

    # ---- SW angle consistency
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.4))
    x = np.arange(len(names))
    ax = axes[0]
    ax.axhspan(phi - phis.std(), phi + phis.std(), color="#f0efec", zorder=0)
    ax.axhline(phi, color="#c3c2b7", lw=1)
    ax.scatter(x, phis, s=46, color="#0b0b0b", zorder=3)
    ax.set_xticks(x)
    ax.set_xticklabels([n[3:9] for n in names], fontsize=8)
    ax.set_ylabel("각도 (°)")
    ax.set_title(f"SW 정답 점에서 잰 각도: P1→P20 선과 P1→P15 축 사이  =  {phi:.1f} ± {phis.std():.1f}°", loc="left", fontsize=10)
    ax.yaxis.grid(True, color="#e1e0d9")
    ax.set_axisbelow(True)
    ax = axes[1]
    ang = np.array([r[1] for r in rows])
    ax.axhspan(-1, 1, color="#f0efec", zorder=0)
    ax.axhline(0, color="#c3c2b7", lw=1)
    ax.scatter(x, ang, s=46, color="#7a3fbf", zorder=3)
    ax.set_xticks(x)
    ax.set_xticklabels([n[3:9] for n in names], fontsize=8)
    ax.set_ylabel("각도 오차 (°)")
    ax.set_title(f"메쉬에서 만든 축(띠 P1→P20′ 선을 {phi:.1f}° 회전) vs SW 축  =  {ang.mean():+.1f} ± {ang.std():.1f}°", loc="left", fontsize=10)
    ax.yaxis.grid(True, color="#e1e0d9")
    ax.set_axisbelow(True)
    for a_ in axes:
        for s in ("top", "right"):
            a_.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "sw_angle_consistency.png"), dpi=150)
    plt.close(fig)

    # ---- axis explainer (one hand)
    e = example
    parts, p1, p20, u = e["parts"], e["p1"], e["p20"], e["u"]
    S = parts["S"]
    fig, ax = plt.subplots(figsize=(9.5, 9))
    pts = np.array([e["g1"][:2], e["g15"][:2], p20[:2]])
    lo_xy, hi_xy = pts.min(0) - 20, pts.max(0) + 16
    crop = S[(S[:, 0] > lo_xy[0]) & (S[:, 0] < hi_xy[0]) & (S[:, 1] > lo_xy[1]) & (S[:, 1] < hi_xy[1])]
    ax.scatter(crop[:, 0], crop[:, 1], s=1.2, color="#e3e2dd", zorder=0)
    v = unit(p20[:2] - p1[:2])
    ax.plot([p1[0], p20[0]], [p1[1], p20[1]], "--", color="#898781", lw=1.6, zorder=2, label="① P1 → P20′ 선")
    # angle arc
    th0 = np.arctan2(v[1], v[0]); th1 = np.arctan2(u[1], u[0])
    tt = np.linspace(th0, th1, 30); r = 22.0
    ax.plot(p1[0] + r * np.cos(tt), p1[1] + r * np.sin(tt), color="#7a3fbf", lw=1.4, zorder=3)
    mid = 0.5 * (th0 + th1)
    ax.annotate(f"② {phi:.1f}° 회전\n(SW 정답 점들에서 잰 상수)", (p1[0] + (r + 3) * np.cos(mid), p1[1] + (r + 3) * np.sin(mid)), fontsize=9.5, color="#7a3fbf")
    a1 = p1[:2] + 62 * u
    ax.plot([p1[0], a1[0]], [p1[1], a1[1]], color="#7a3fbf", lw=2.6, zorder=4, label="② 엄지 축 = 회전한 선")
    u_true = unit(e["g15"][:2] - e["g1"][:2])
    b1 = e["g1"][:2] + 62 * u_true
    ax.plot([e["g1"][0], b1[0]], [e["g1"][1], b1[1]], color="k", lw=1.3, zorder=4, label=f"SW 축 (정답 P1→P15), 각도 차 {e['ang']:+.1f}°")
    foot = p1[:2] + e["t20"] * u
    ax.plot([p20[0], foot[0]], [p20[1], foot[1]], ":", color="#00a0b0", lw=1.4, zorder=3)
    ax.scatter(*foot, marker="x", s=80, color="#00a0b0", lw=2, zorder=6, label="③ P20′의 수직 발")
    wa, wb = p1[:2] + e["lo"] * u, p1[:2] + e["hi"] * u
    ax.plot([wa[0], wb[0]], [wa[1], wb[1]], color="#f06292", lw=8, alpha=0.6, zorder=3, label="③ 발 기준 85~95° 창")
    ax.scatter(*p1[:2], marker="*", s=280, color="#ff2b2b", edgecolor="k", zorder=7, label="띠 규칙 P1")
    ax.scatter(*p20[:2], s=100, color="#00e5ff", edgecolor="k", zorder=6, label="P20′ (엄지–검지 오목점)")
    ax.scatter(*e["p15"][:2], marker="*", s=280, color="#39ff14", edgecolor="k", zorder=7, label="④ 규칙 P15 = 창 안 컨투어 최고점")
    ax.scatter(*e["g15"][:2], marker="x", s=120, color="k", lw=2.2, zorder=8, label="SW 정답 P15")
    ax.set_aspect("equal")
    ax.set_xlim(lo_xy[0], hi_xy[0])
    ax.set_ylim(lo_xy[1], hi_xy[1])
    ax.set_xlabel("x (obj 좌표계, mm)")
    ax.set_ylabel("y")
    ax.set_title(f"최단순 엄지 축 ({EXAMPLE}): P1과 P20′만으로   |   길이 오차 {e['L'] - e['Lref']:+.1f}mm", loc="left", fontsize=11)
    ax.legend(fontsize=8.5, loc="lower left", framealpha=0.95)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "axis_explainer.png"), dpi=140)
    plt.close(fig)

    # ---- pipeline explainer (three panels)
    fig, axes = plt.subplots(1, 3, figsize=(19, 6.6))
    ax = axes[0]
    fr = parts["frame"]
    near = S[np.linalg.norm(S[:, :2] - parts["tip_outline"][:2], axis=1) < TIP_RADIUS]
    work = fr.to_work(near)
    band = work[:, 1] > work[:, 1].max() - BAND_Y
    ax.scatter(near[~band, 0], near[~band, 1], s=2, color="#d5d4cf", label="엄지 끝 주변 점")
    ax.scatter(near[band, 0], near[band, 1], s=14, color="#2a78d6", label="후보: 손 좌표계 y최대점에서 1mm 아래까지")
    ax.scatter(*parts["maxy"][:2], marker="^", s=80, color="#898781", edgecolor="k", zorder=5, label="y최대점")
    ax.scatter(*p1[:2], marker="*", s=260, color="#ff2b2b", edgecolor="k", zorder=6, label="띠 규칙 P1 = 후보 중 엄지 바깥쪽 끝점")
    ax.scatter(*e["g1"][:2], marker="x", s=120, color="k", lw=2.2, zorder=7, label="SW 정답 P1")
    c = parts["tip_outline"][:2]
    ax.set_aspect("equal")
    ax.set_xlim(c[0] - 20, c[0] + 20)
    ax.set_ylim(c[1] - 24, c[1] + 12)
    ax.set_title(f"① P1 띠 규칙  |  {EXAMPLE}", loc="left", fontsize=11)
    ax.legend(fontsize=8, loc="lower left")
    ax.set_xlabel("x (obj 좌표계, mm)")
    ax.set_ylabel("y")
    ax = axes[1]
    ax.scatter(crop[:, 0], crop[:, 1], s=1.2, color="#dedede")
    ax.plot([p1[0], p20[0]], [p1[1], p20[1]], "--", color="#898781", lw=1.4, label="P1 → P20′ 선")
    ax.plot([p1[0], a1[0]], [p1[1], a1[1]], color="#7a3fbf", lw=2.4, label=f"축 = 그 선을 {phi:.1f}° 회전")
    ax.plot([e["g1"][0], b1[0]], [e["g1"][1], b1[1]], color="k", lw=1.3, label=f"SW 축, 각도 차 {e['ang']:+.1f}°")
    ax.scatter(*p1[:2], marker="*", s=220, color="#ff2b2b", edgecolor="k", zorder=7, label="띠 규칙 P1")
    ax.scatter(*p20[:2], s=90, color="#00e5ff", edgecolor="k", zorder=6, label="P20′")
    ax.scatter(*e["g15"][:2], marker="x", s=120, color="k", lw=2.2, zorder=7, label="SW 정답 P15")
    ax.scatter(*e["p15"][:2], marker="*", s=220, color="#39ff14", edgecolor="k", zorder=7, label="규칙 P15 (③에서 결정)")
    ax.set_aspect("equal")
    ax.set_xlim(lo_xy[0], hi_xy[0])
    ax.set_ylim(lo_xy[1], hi_xy[1])
    ax.set_title("② 엄지 축: P1 → P20′ 선을 회전", loc="left", fontsize=11)
    ax.legend(fontsize=8, loc="lower left")
    ax.set_xlabel("x (mm)")
    ax = axes[2]
    ts, zs = e["ts"], e["zs"]
    t_true = float((e["g15"][:2] - p1[:2]) @ u)
    ax.plot(ts, zs, color="#2a78d6", lw=1.6, label="축 띠(±12mm) 밑면 컨투어: 1mm마다 최저 z")
    ax.axvspan(e["lo"], e["hi"], color="#f8c8d8", alpha=0.6, label=f"P20′ 발 기준 85~95° 창 (t {e['lo']:.1f}~{e['hi']:.1f})")
    ax.axvline(e["t20"], color="#00a0b0", ls=":", lw=1.4, label=f"P20′ 수직 발 t={e['t20']:.1f}")
    ax.scatter([e["t_est"]], [zs[np.argmin(np.abs(ts - e["t_est"]))]], marker="*", s=260, color="#39ff14", edgecolor="k", zorder=6, label=f"창 안 최고점 = 규칙 P15, t={e['t_est']:.1f}")
    ax.axvline(t_true, color="k", lw=1.4, label=f"SW 정답 P15의 축 위 위치 t={t_true:.1f}")
    ax.set_xlim(0, 62)
    ax.set_xlabel("축 방향 거리 t (P1에서 손목 쪽으로, mm)")
    ax.set_ylabel("밑면 z (mm, 유리 = 0)")
    ax.set_title("③ 규칙 P15: 창 안 컨투어 최고점", loc="left", fontsize=11)
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(color="#e1e0d9")
    fig.suptitle(f"최단순 규칙 파이프라인 한눈에 보기 ({EXAMPLE}): 길이 예측 {e['L']:.1f}mm, SW 기준 {e['Lref']:.1f}mm, 오차 {e['L'] - e['Lref']:+.1f}mm", x=0.01, ha="left", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(os.path.join(OUT, "pipeline_explainer.png"), dpi=140)
    plt.close(fig)
    print(f"phi = {phi:.2f} +/- {phis.std():.2f}; mesh axis vs SW {ang.mean():+.2f} +/- {ang.std():.2f}; wrote {OUT}")


if __name__ == "__main__":
    main()
