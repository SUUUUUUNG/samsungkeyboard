"""How the rule-only thumb axis (band P1 -> B) is built, on one hand, obj frame:
band P1 and P20' -> preliminary direction -> width midpoints at 6..26 mm from P1
-> fitted medial axis -> B = width midpoint 2 cm distal of the P20' level -> axis P1 -> B.
The silhouette tip point is not part of the construction (it only seeds the tip region).

    python thumb_model/axis_explainer.py [name]   -> thumb_model/report/axis_explainer.png
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)
import rule_pipeline as rp  # noqa: E402
from align_landmarks import OBJ_DIR, OUT_DIR as ALIGNED_DIR, read_obj  # noqa: E402
from final_pipeline import axis_direction, rule_parts, thumb_B, unit  # noqa: E402


def main(name="20_F_0004G"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.family"] = "Malgun Gothic"
    plt.rcParams["axes.unicode_minus"] = False

    V, F, _ = read_obj(os.path.join(OBJ_DIR, name + ".obj"))
    A = np.loadtxt(os.path.join(ALIGNED_DIR, f"{name}_aligned.lnd"))[:, 1:]
    g1, g15 = A[0], A[14]
    parts = rule_parts(V, F)
    r = rp.run(V, F)
    fr = r["frame"]
    p1 = parts["P1_band1"]
    B, q = thumb_B(parts["S"], r, p1, return_parts=True)
    Sw, outl, tip, d, web = q["Sw"], q["outl"], q["tip_w"], q["d"], q["web_w"]
    P = outl["pts"][:, :2]
    radial = P[:q["tips"][0]]
    d_web = np.linalg.norm(web - tip)
    r_match = radial[np.argmin(np.abs(np.linalg.norm(radial - tip, axis=1) - d_web))]
    d0 = unit(tip - 0.5 * (web + r_match))
    lat0 = np.array([-d0[1], d0[0]])
    rel = Sw[:, :2] - tip
    t0, w0 = rel @ (-d0), rel @ lat0
    slabs = []
    for s in (6.0, 10.0, 14.0, 18.0, 22.0, 26.0):
        if s >= 0.8 * d_web:
            continue
        m = (np.abs(t0 - s) < rp.SLAB_MM) & (np.abs(w0) < 15.0)
        if m.sum() < 4:
            continue
        wl, wr = w0[m].min(), w0[m].max()
        base = tip + s * (-d0)
        slabs.append((base + wl * lat0, base + wr * lat0, base + 0.5 * (wl + wr) * lat0))
    lat = np.array([-d[1], d[0]])
    sB = q["web_t"] - rp.AXIS_STATION_MM
    m = (np.abs(rel @ (-d) - sB) < rp.SLAB_MM) & (np.abs(rel @ lat) < 15.0)
    wl, wr = (rel @ lat)[m].min(), (rel @ lat)[m].max()
    B_edges = (tip + sB * (-d) + wl * lat, tip + sB * (-d) + wr * lat)
    web_level = (web - 32 * lat, web + 32 * lat)
    to_obj = lambda z: fr.from_work(np.array([[z[0], z[1], 0.0]]))[0][:2]

    u = axis_direction(parts, "p1_to_B", p1)
    u_true = unit(g15[:2] - g1[:2])
    ang_rule = np.degrees(np.arctan2(u_true[0] * u[1] - u_true[1] * u[0], u_true @ u))

    fig, ax = plt.subplots(figsize=(10.5, 10))
    pts = np.array([g1[:2], g15[:2], parts["P20"][:2]])
    lo, hi = pts.min(0) - 20, pts.max(0) + 16
    S = parts["S"]
    crop = S[(S[:, 0] > lo[0]) & (S[:, 0] < hi[0]) & (S[:, 1] > lo[1]) & (S[:, 1] < hi[1])]
    ax.scatter(crop[:, 0], crop[:, 1], s=1.2, color="#e3e2dd", zorder=0)
    ol = np.array([to_obj(z) for z in P[max(0, q["tips"][0] - 260):q["webs"][0] + 40]])
    ax.plot(ol[:, 0], ol[:, 1], color="#b5b3ab", lw=1, zorder=1, label="실루엣(외곽선): P20′과 엄지 바깥 가장자리에만 사용")
    W_, R_ = to_obj(web), to_obj(r_match)
    ax.scatter(*p1[:2], marker="*", s=260, color="#ff2b2b", edgecolor="k", zorder=8, label="① 띠 규칙 P1 (3단계에서 구함)")
    ax.scatter(*W_, s=90, color="#00e5ff", edgecolor="k", zorder=6, label="① 엄지–검지 오목점 P20′")
    ax.scatter(*R_, marker="<", s=80, color="#b5b3ab", edgecolor="k", zorder=6, label="① 바깥 가장자리에서 P1까지 P20′와 같은 거리인 점")
    pre0, pre1 = to_obj(tip + 3 * d0), to_obj(tip - 45 * d0)
    ax.plot([pre0[0], pre1[0]], [pre0[1], pre1[1]], ":", color="#b5b3ab", lw=1.2, zorder=2, label="② 예비 방향: P1 → (P20′와 바깥점의 중점)")
    for i, (e0, e1, mid) in enumerate(slabs):
        a0, a1, am = to_obj(e0), to_obj(e1), to_obj(mid)
        ax.plot([a0[0], a1[0]], [a0[1], a1[1]], color="#eb6834", lw=1.1, zorder=3, label="③ P1에서 6~26mm 지점마다 예비 방향에 수직으로 잰 엄지 폭" if i == 0 else None)
        ax.scatter(*am, s=42, color="#eb6834", edgecolor="k", zorder=6, label="③ 그 폭의 중앙점" if i == 0 else None)
    f0, f1 = to_obj(q["ctr"] + 8 * d), to_obj(q["ctr"] - 40 * d)
    ax.plot([f0[0], f1[0]], [f0[1], f1[1]], "--", color="#eb6834", lw=1.3, zorder=3, label="④ 중앙점들에 맞춘 직선(내측 축): 폭을 재는 방향만 정함")
    wl0, wl1 = to_obj(web_level[0]), to_obj(web_level[1])
    ax.plot([wl0[0], wl1[0]], [wl0[1], wl1[1]], "-.", color="#00a0b0", lw=1, zorder=3, label="⑤ P20′ 높이 (P20′을 지나고 내측 축에 수직)")
    be0, be1 = to_obj(B_edges[0]), to_obj(B_edges[1])
    ax.plot([be0[0], be1[0]], [be0[1], be1[1]], color="#1a1a19", lw=1.6, zorder=4, label="⑤ P20′ 높이에서 끝 쪽으로 20mm 위의 엄지 폭")
    ax.scatter(*B[:2], marker="s", s=120, color="#ffd54a", edgecolor="k", zorder=7, label="⑤ 그 폭의 중앙점 B")
    a1 = p1[:2] + 60 * u
    ax.plot([p1[0], a1[0]], [p1[1], a1[1]], color="#1a9850", lw=2.4, zorder=6, label="⑥ 규칙 전용 엄지 축 = P1 → B")
    b1 = g1[:2] + 60 * u_true
    ax.plot([g1[0], b1[0]], [g1[1], b1[1]], color="#7a3fbf", lw=1.4, zorder=6, label="SW 축 (정답 P1 → P15)")
    ax.set_aspect("equal")
    ax.set_xlim(lo[0], hi[0])
    ax.set_ylim(lo[1], hi[1])
    ax.set_xlabel("x (obj 좌표계, mm)")
    ax.set_ylabel("y")
    ax.set_title(f"규칙 전용 엄지 축은 어떻게 만드나 ({name})   |   SW 축과의 각도 {ang_rule:+.1f}°", loc="left", fontsize=11)
    ax.legend(fontsize=8, loc="lower left", framealpha=0.95)
    fig.tight_layout()
    out = os.path.join(HERE, "report", "axis_explainer.png")
    fig.savefig(out, dpi=140)
    print("wrote", out)


if __name__ == "__main__":
    main(*sys.argv[1:2])
