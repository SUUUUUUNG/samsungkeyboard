"""Per-landmark (P1..P28) error report across approaches, for sharing with the team.

Inputs (thumb_model/):
  rule_eval.csv                         rule pipeline, P{k}_err3d columns (28 points)
  landmark_errors_learned.csv           learned model, 15-fold LOO, 15 x 28
  landmark_errors_baseline.csv          mean-position baseline, LOO, 15 x 28
  rule_landmarks.csv / landmark_preds_learned.npz   predictions for the hand maps
  results_cv_hgb.csv + results_test_hgb.csv         apex/atlas held-out P1/P15 (thumb-length panel)
Outputs (thumb_model/report/):
  landmark_errors_summary.csv, fig1..fig4 png, landmark_report.html (self-contained)
"""
import base64
import csv
import io
import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from align_landmarks import OBJ_DIR, OUT_DIR as ALIGNED_DIR, REFERENCE_THUMB_MM, read_obj  # noqa: E402

REPORT = os.path.join(HERE, "report")
POINTS = [f"P{i}" for i in range(1, 29)]
CATEGORIES = [("손가락 끝", ["P1", "P2", "P3", "P4", "P5"]),
              ("엄지 IP·기저", ["P6", "P15"]),
              ("손가락 기저", ["P16", "P17", "P18", "P19"]),
              ("web·측면 (실루엣)", ["P20", "P21", "P22", "P23", "P24", "P25"]),
              ("손목", ["P26", "P27", "P28"]),
              ("파생점 (3등분)", ["P7", "P8", "P9", "P10", "P11", "P12", "P13", "P14"])]
ORDER = [p for _, ps in CATEGORIES for p in ps]
APPROACHES = [("rule", "규칙 파이프라인 (특허 재구성, 학습 없음)"),
              ("learned", "학습 모델 (정점 특징 + 부스팅, 19점, 15-fold LOO)"),
              ("baseline", "기준선 (공통 좌표계 평균 위치, LOO)")]
COLORS = {"rule": "#2a78d6", "learned": "#eb6834", "baseline": "#1baf7a", "best": "#eda100"}
FLAGGED = {"20_F_0097G", "20_M_1113G", "20_M_1573G"}
EXAMPLE_HANDS = ["20_M_1559G", "20_M_1113G"]
SEQ = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7", "#3987e5", "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b"]


def short(name):
    return name[3:9]


def load_errors():
    rule = pd.read_csv(os.path.join(HERE, "rule_eval.csv")).set_index("name")
    err = {"rule": rule[[f"{p}_err3d" for p in POINTS]].rename(columns=lambda c: c[:-6])}
    for tag in ("learned", "baseline"):
        err[tag] = pd.read_csv(os.path.join(HERE, f"landmark_errors_{tag}.csv")).set_index("name")[POINTS]
    hands = list(err["rule"].index)
    for tag in err:
        err[tag] = err[tag].loc[hands]
    return err, hands, rule


def thumb_lengths(hands, rule):
    ref = np.array([REFERENCE_THUMB_MM[h[:-1]] for h in hands])
    out = {"rule": rule.loc[hands, "len[B_along_2cm]"].values - ref}
    for tag in ("learned", "baseline"):
        z = np.load(os.path.join(HERE, f"landmark_preds_{tag}.npz"))
        out[tag] = np.array([np.linalg.norm(z[f"{h}/P1"][:2] - z[f"{h}/P15"][:2]) for h in hands]) - ref
    cv = pd.read_csv(os.path.join(HERE, "results_cv_hgb.csv"))
    te = pd.read_csv(os.path.join(HERE, "results_test_hgb.csv"))
    d = pd.concat([cv, te])
    d = d[(d.P1_method == "apex") & (d.P15_method == "atlas")].set_index("name")
    out["best"] = d.loc[hands, "length_err_mm"].values
    return out, ref


def summary_table(err):
    rows = []
    for p in ORDER:
        cat = next(c for c, ps in CATEGORIES if p in ps)
        row = {"point": p, "category": cat}
        for tag, _ in APPROACHES:
            v = err[tag][p].values
            row[f"{tag}_mean"] = v.mean()
            row[f"{tag}_sd"] = v.std()
            row[f"{tag}_max"] = v.max()
        rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- figures
def setup_mpl():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    for f in ("Malgun Gothic", "NanumGothic", "AppleGothic"):
        try:
            matplotlib.font_manager.findfont(f, fallback_to_default=False)
            plt.rcParams["font.family"] = f
            break
        except Exception:
            continue
    plt.rcParams.update({"axes.unicode_minus": False, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.edgecolor": "#c3c2b7", "axes.labelcolor": "#52514e", "xtick.color": "#898781",
                         "ytick.color": "#898781", "grid.color": "#e1e0d9", "grid.linewidth": 0.8,
                         "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb", "font.size": 9})
    return plt


def fig_bars(plt, summ):
    fig, ax = plt.subplots(figsize=(13, 4.6))
    x = np.arange(len(ORDER))
    w = 0.26
    for i, (tag, label) in enumerate(APPROACHES):
        ax.bar(x + (i - 1) * w, summ[f"{tag}_mean"], w * 0.92, color=COLORS[tag], label=label, zorder=3)
        ax.errorbar(x + (i - 1) * w, summ[f"{tag}_mean"], yerr=summ[f"{tag}_sd"], fmt="none", ecolor="#52514e",
                    elinewidth=0.8, capsize=0, zorder=4)
    for y, lab in ((0.5, "0.5 mm 목표"), (2.0, "2 mm")):
        ax.axhline(y, color="#898781", lw=0.8, zorder=2)
        ax.text(len(ORDER) - 0.3, y, lab, va="center", ha="left", fontsize=8, color="#52514e", clip_on=False)
    # category bands
    start = 0
    for ci, (cat, ps) in enumerate(CATEGORIES):
        n = len(ps)
        if ci % 2 == 1:
            ax.axvspan(start - 0.5, start + n - 0.5, color="#f0efec", zorder=0)
        ax.text(start + n / 2 - 0.5, ax.get_ylim()[1] * 0.98 if False else 0, "", ha="center")
        ax.annotate(cat, xy=(start + n / 2 - 0.5, 1.0), xycoords=("data", "axes fraction"), ha="center", va="bottom",
                    fontsize=8.5, color="#52514e")
        start += n
    ax.set_xticks(x)
    ax.set_xticklabels(ORDER, rotation=0, fontsize=8)
    ax.set_xlim(-0.6, len(ORDER) - 0.4)
    ax.set_ylabel("SW 랜드마크와의 3D 거리 (mm)\n15명 평균, 선은 ± 표준편차")
    fig.subplots_adjust(right=0.93)
    ax.yaxis.grid(True, zorder=1)
    ax.set_axisbelow(True)
    ax.legend(loc="upper left", frameon=False, fontsize=8.5, ncol=3, bbox_to_anchor=(0, -0.12))
    fig.tight_layout()
    fig.savefig(os.path.join(REPORT, "fig1_landmark_errors.png"), dpi=160)
    plt.close(fig)


def fig_heatmaps(plt, err, hands):
    from matplotlib.colors import LinearSegmentedColormap, Normalize
    cmap = LinearSegmentedColormap.from_list("seqblue", SEQ)
    norm = Normalize(0, 8)
    fig, axes = plt.subplots(3, 1, figsize=(13, 10.5), sharex=True)
    for ax, (tag, label) in zip(axes, APPROACHES):
        M = err[tag][ORDER].values
        im = ax.imshow(M, cmap=cmap, norm=norm, aspect="auto")
        ax.set_yticks(range(len(hands)))
        ax.set_yticklabels([short(h) + ("  ▲" if h in FLAGGED else "") for h in hands], fontsize=8)
        ax.set_xticks(range(len(ORDER)))
        ax.set_xticklabels(ORDER, fontsize=8)
        ax.set_title(label, loc="left", fontsize=10, color="#0b0b0b")
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.tick_params(length=0)
        start = 0
        for cat, ps in CATEGORIES:
            start += len(ps)
            ax.axvline(start - 0.5, color="#fcfcfb", lw=2)
    cb = fig.colorbar(im, ax=axes, orientation="vertical", fraction=0.02, pad=0.01, extend="max")
    cb.set_label("3D 오차 (mm)")
    cb.outline.set_visible(False)
    fig.text(0.01, 0.005, "▲ = 모든 접근에서 P15가 같은 방향으로 틀리는 세 명 (0097, 1113, 1573)", fontsize=8, color="#52514e")
    fig.savefig(os.path.join(REPORT, "fig2_heatmaps.png"), dpi=160, bbox_inches="tight")
    plt.close(fig)


def fig_thumb(plt, tl, hands):
    fig, ax = plt.subplots(figsize=(13, 4))
    x = np.arange(len(hands))
    ax.axhspan(-0.5, 0.5, color="#f0efec", zorder=0)
    ax.axhline(0, color="#c3c2b7", lw=0.8, zorder=1)
    series = [("rule", "규칙 파이프라인"), ("learned", "학습 모델 (19점 확장, coarse+fine)"),
              ("best", "학습 모델 (P1·P15 전용 apex/atlas)"), ("baseline", "기준선")]
    for i, (tag, label) in enumerate(series):
        ax.scatter(x + (i - 1.5) * 0.17, tl[tag], s=34, color=COLORS[tag], label=label, zorder=3,
                   edgecolor="#fcfcfb", linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels([short(h) for h in hands], fontsize=8)
    ax.set_ylabel("엄지 길이 오차 = 예측 - 기준값 (mm)")
    ax.text(-0.45, 0.5, "±0.5 mm 목표 띠", ha="left", va="bottom", fontsize=8, color="#52514e")
    ax.yaxis.grid(True, zorder=1)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=8.5, ncol=4, loc="upper left", bbox_to_anchor=(0, -0.12))
    fig.tight_layout()
    fig.savefig(os.path.join(REPORT, "fig3_thumb_length.png"), dpi=160)
    plt.close(fig)


def fig_hands(plt, hands):
    rl = pd.read_csv(os.path.join(HERE, "rule_landmarks.csv")).set_index("name")
    lz = np.load(os.path.join(HERE, "landmark_preds_learned.npz"))
    fig, axes = plt.subplots(1, 2, figsize=(13, 7.2))
    for ax, h in zip(axes, EXAMPLE_HANDS):
        V, _, _ = read_obj(os.path.join(OBJ_DIR, h + ".obj"))
        A = np.loadtxt(os.path.join(ALIGNED_DIR, f"{h}_aligned.lnd"))[:, 1:]
        ax.scatter(V[::3, 0], V[::3, 1], s=0.4, color="#e1e0d9", zorder=0)
        for p in POINTS:
            g = A[int(p[1:]) - 1]
            r = np.array([rl.loc[h, f"{p}_x"], rl.loc[h, f"{p}_y"]])
            l = lz[f"{h}/{p}"][:2]
            ax.annotate("", xy=r, xytext=g[:2], arrowprops=dict(arrowstyle="-", color=COLORS["rule"], lw=1.2), zorder=2)
            ax.annotate("", xy=l, xytext=g[:2], arrowprops=dict(arrowstyle="-", color=COLORS["learned"], lw=1.2), zorder=2)
            ax.scatter(*r, s=16, color=COLORS["rule"], zorder=3, edgecolor="#fcfcfb", linewidth=0.6)
            ax.scatter(*l, s=16, color=COLORS["learned"], zorder=3, edgecolor="#fcfcfb", linewidth=0.6)
            ax.scatter(g[0], g[1], s=22, color="#0b0b0b", zorder=4)
            ax.text(g[0] + 1.5, g[1] + 1.5, p[1:], fontsize=6.5, color="#52514e", zorder=5)
        ax.set_aspect("equal")
        ax.set_title(f"{short(h)}  ({'잘 맞는 손' if h == EXAMPLE_HANDS[0] else '안 맞는 손'})", loc="left", fontsize=10)
        ax.set_xlabel("x (mm, obj 좌표계)")
        ax.set_ylabel("y (mm)")
        ax.grid(True)
        ax.set_axisbelow(True)
    handles = [plt.Line2D([], [], marker="o", ls="", color="#0b0b0b", label="SW 랜드마크 (정답)"),
               plt.Line2D([], [], marker="o", ls="", color=COLORS["rule"], label="규칙 파이프라인"),
               plt.Line2D([], [], marker="o", ls="", color=COLORS["learned"], label="학습 모델")]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, fontsize=9)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(os.path.join(REPORT, "fig4_hand_maps.png"), dpi=160)
    plt.close(fig)


# ------------------------------------------------------------------- html
def build_html(err, hands, summ, tl, ref):
    with open(os.path.join(REPORT, "fig4_hand_maps.png"), "rb") as fh:
        fig4 = base64.b64encode(fh.read()).decode()
    data = {
        "hands": hands, "short": [short(h) for h in hands], "flagged": [h in FLAGGED for h in hands],
        "points": ORDER, "categories": [{"name": c, "points": ps} for c, ps in CATEGORIES],
        "approaches": [{"key": k, "label": l, "color": COLORS[k]} for k, l in APPROACHES],
        "errors": {k: {h: [round(float(err[k].loc[h, p]), 2) for p in ORDER] for h in hands} for k, _ in APPROACHES},
        "summary": {k: {"mean": [round(float(v), 2) for v in summ[f"{k}_mean"]],
                        "sd": [round(float(v), 2) for v in summ[f"{k}_sd"]],
                        "max": [round(float(v), 2) for v in summ[f"{k}_max"]]} for k, _ in APPROACHES},
        "thumb": {k: [round(float(v), 2) for v in tl[k]] for k in ("rule", "learned", "best", "baseline")},
        "thumb_ref": [float(r) for r in ref],
    }
    overall = {k: float(err[k][ORDER].values.mean()) for k, _ in APPROACHES}
    p15 = {k: float(err[k]["P15"].mean()) for k, _ in APPROACHES}
    thumb_mae = {k: float(np.abs(tl[k]).mean()) for k in ("rule", "learned", "best", "baseline")}
    tiles = "".join(f"""
      <div class="tile"><div class="tile-label">{l}</div>
        <div class="tile-value">{overall[k]:.2f}<span class="unit">mm</span></div>
        <div class="tile-sub">28점 평균 3D 오차 · P15 {p15[k]:.2f} mm · 엄지 길이 MAE {thumb_mae[k]:.2f} mm</div></div>"""
                    for k, l in APPROACHES)
    table_rows = "".join(
        f"<tr><td class='pt'>{r['point']}</td><td class='cat'>{r['category']}</td>" +
        "".join(f"<td>{r[f'{k}_mean']:.2f} <span class='sd'>± {r[f'{k}_sd']:.2f}</span></td><td class='mx'>{r[f'{k}_max']:.1f}</td>" for k, _ in APPROACHES) +
        "</tr>" for _, r in summ.iterrows())
    html = HTML_TEMPLATE
    for key, val in (("__DATA__", json.dumps(data, ensure_ascii=False)), ("__TILES__", tiles), ("__TABLE__", table_rows),
                     ("__FIG4__", fig4), ("__N_FLAG__", str(len(FLAGGED))),
                     ("__BEST_MAE__", f"{thumb_mae['best']:.2f}"), ("__RULE_MAE__", f"{thumb_mae['rule']:.2f}")):
        html = html.replace(key, val)
    with open(os.path.join(REPORT, "landmark_report.html"), "w", encoding="utf-8") as fh:
        fh.write(html)


HTML_TEMPLATE = r"""<title>랜드마크 재현 오차</title>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans+KR:wght@400;500;600&display=swap">
<style>
:root{
  color-scheme:light;
  --page:#f9f9f7; --surface:#fcfcfb; --ink:#0b0b0b; --ink-2:#52514e; --muted:#898781; --grid:#e1e0d9; --axis:#c3c2b7;
  --band:#f0efec; --ring:rgba(11,11,11,.10);
  --rule:#2a78d6; --learned:#eb6834; --baseline:#1baf7a; --best:#eda100;
  --seq0:#cde2fb; --seq1:#9ec5f4; --seq2:#6da7ec; --seq3:#3987e5; --seq4:#256abf; --seq5:#184f95; --seq6:#0d366b;
}
@media (prefers-color-scheme: dark){ :root:not([data-theme="light"]){
  color-scheme:dark;
  --page:#0d0d0d; --surface:#1a1a19; --ink:#ffffff; --ink-2:#c3c2b7; --muted:#898781; --grid:#2c2c2a; --axis:#383835;
  --band:#232322; --ring:rgba(255,255,255,.10);
  --rule:#3987e5; --learned:#d95926; --baseline:#199e70; --best:#c98500;
}}
:root[data-theme="dark"]{
  color-scheme:dark;
  --page:#0d0d0d; --surface:#1a1a19; --ink:#ffffff; --ink-2:#c3c2b7; --muted:#898781; --grid:#2c2c2a; --axis:#383835;
  --band:#232322; --ring:rgba(255,255,255,.10);
  --rule:#3987e5; --learned:#d95926; --baseline:#199e70; --best:#c98500;
}
body{background:var(--page);color:var(--ink);font-family:"IBM Plex Sans KR",system-ui,-apple-system,"Segoe UI","Malgun Gothic",sans-serif;font-size:14px;line-height:1.55}
.wrap{max-width:1120px;margin:0 auto;padding-block:28px 48px;padding-inline:20px}
h1{font-size:26px;font-weight:600;margin:0 0 6px;letter-spacing:-.01em;text-wrap:balance}
h2{font-size:17px;font-weight:600;margin:40px 0 6px}
p{max-width:72ch;margin:6px 0;color:var(--ink-2)}
p.lead{color:var(--ink);font-size:15px}
.eyebrow{font-size:11px;letter-spacing:.08em;text-transform:uppercase;color:var(--muted);margin-bottom:8px}
.tiles{display:grid;grid-template-columns:repeat(auto-fit,minmax(240px,1fr));gap:12px;margin:22px 0 8px}
.tile{background:var(--surface);border:1px solid var(--ring);border-radius:8px;padding:14px 16px}
.tile-label{font-size:12px;color:var(--ink-2);min-height:2.6em}
.tile-value{font-size:30px;font-weight:600;margin-top:4px}
.tile-value .unit{font-size:14px;font-weight:400;color:var(--muted);margin-left:4px}
.tile-sub{font-size:12px;color:var(--muted);margin-top:4px}
.card{background:var(--surface);border:1px solid var(--ring);border-radius:8px;padding:16px 16px 12px;margin-top:12px}
.legend{display:flex;flex-wrap:wrap;gap:6px 18px;font-size:12.5px;color:var(--ink-2);margin:6px 0 10px}
.legend span::before{content:"";display:inline-block;width:10px;height:10px;border-radius:2px;margin-right:6px;vertical-align:-1px;background:var(--c)}
svg{display:block;width:100%;height:auto}
svg text{font-family:inherit;fill:var(--muted);font-size:11px}
svg .axis{stroke:var(--axis);stroke-width:1}
svg .grid{stroke:var(--grid);stroke-width:1}
svg .ref{stroke:var(--muted);stroke-width:1}
svg .cat{fill:var(--ink-2);font-size:11.5px}
svg .band{fill:var(--band)}
.tip{position:fixed;pointer-events:none;background:var(--ink);color:var(--page);padding:6px 9px;border-radius:6px;font-size:12px;line-height:1.4;opacity:0;transition:opacity .08s;z-index:10;max-width:260px}
.tip b{font-weight:600}
.heat{overflow-x:auto}
.heat-grid{display:grid;gap:2px;font-size:11px;font-variant-numeric:tabular-nums;min-width:820px}
.heat-grid .h{color:var(--muted);text-align:center;padding:2px 0}
.heat-grid .r{color:var(--ink-2);padding-right:8px;text-align:right;white-space:nowrap}
.heat-grid .r.flag{color:var(--ink);font-weight:600}
.cell{height:22px;border-radius:2px;cursor:default}
.cell:focus-visible{outline:2px solid var(--ink);outline-offset:1px}
.scale{display:flex;align-items:center;gap:8px;font-size:12px;color:var(--ink-2);margin-top:10px}
.scale .ramp{display:flex;gap:2px}.scale .ramp i{display:block;width:26px;height:12px;border-radius:2px}
.tabs{display:flex;gap:6px;flex-wrap:wrap;margin-bottom:10px}
.tabs button{font:inherit;font-size:12.5px;padding:6px 12px;border-radius:999px;border:1px solid var(--ring);background:var(--surface);color:var(--ink-2);cursor:pointer}
.tabs button[aria-selected="true"]{background:var(--ink);color:var(--page);border-color:var(--ink)}
.tabs button:focus-visible{outline:2px solid var(--ink);outline-offset:2px}
table{border-collapse:collapse;width:100%;font-size:12.5px;font-variant-numeric:tabular-nums}
.tbl{overflow-x:auto}
th{font-weight:500;color:var(--ink-2);text-align:right;padding:8px 8px;border-bottom:1px solid var(--axis);white-space:nowrap}
th.l,td.pt,td.cat{text-align:left}
td{padding:6px 8px;border-bottom:1px solid var(--grid);text-align:right;white-space:nowrap}
td.pt{font-weight:600}td.cat{color:var(--muted)}td .sd{color:var(--muted)}td.mx{color:var(--muted)}
th.g{text-align:center;border-bottom:none;padding-bottom:0}
ul{max-width:80ch;color:var(--ink-2);padding-left:20px}li{margin:4px 0}
img{max-width:100%;border-radius:6px}
.note{font-size:12.5px;color:var(--muted)}
@media (max-width:640px){h1{font-size:22px}.wrap{padding-inline:16px}}
</style>
<div class="wrap">
  <div class="eyebrow">samsungkeyboard · 브랜치 hy1 · 2026-09-24</div>
  <h1>랜드마크 재현 오차 비교</h1>
  <p class="lead">손 3D 메쉬(obj)만으로 비공개 SW의 랜드마크 28개를 얼마나 재현하는지, 접근별로 15명 전원을 학습에 없는 상태에서 측정한 결과입니다. 오차는 SW 랜드마크와의 3D 거리(mm)입니다.</p>
  <p>28개 점을 실제로 산출하는 접근은 두 가지(규칙 파이프라인, 학습 모델)이며, 기준선은 "모델링 없이 얻는 수준"의 참조입니다. P15 하나만 다룬 분석(고전 ML 손 치수 회귀, 다른 랜드마크→P15 회귀, 기하 구성 전수 탐색)은 전체 점 오차가 정의되지 않아 여기 포함하지 않았습니다.</p>

  <div class="tiles">__TILES__</div>
  <p class="note">엄지 길이(P1–P15 xy 거리) 기준값과의 MAE: 학습 모델의 P1·P15 전용 정밀화(apex/atlas)는 __BEST_MAE__ mm로 가장 낮고, 규칙 파이프라인은 __RULE_MAE__ mm. 목표 0.5 mm에는 어느 접근도 도달하지 못했습니다.</p>

  <h2>1. 점별 평균 오차</h2>
  <p>막대는 15명 평균, 가는 선은 ±표준편차. 가로선은 0.5 mm 목표와 2 mm. 점은 부류별로 묶었습니다.</p>
  <div class="card">
    <div class="legend" id="legend1"></div>
    <div id="bars"></div>
  </div>

  <h2>2. 사람 × 점 히트맵</h2>
  <p>같은 색 척도(0–8 mm)로 접근을 나란히 봅니다. ▲는 모든 접근에서 P15가 같은 방향으로 틀리는 세 명(0097, 1113, 1573). 셀에 마우스를 올리면 값이 보입니다.</p>
  <div class="card">
    <div class="tabs" role="tablist" id="heat-tabs"></div>
    <div class="heat"><div class="heat-grid" id="heat"></div></div>
    <div class="scale"><span>0 mm</span><div class="ramp" id="ramp"></div><span>8 mm 이상</span></div>
  </div>

  <h2>3. 엄지 길이 오차 (사람별)</h2>
  <p>예측 길이 − 기준값. 회색 띠가 ±0.5 mm 목표입니다.</p>
  <div class="card">
    <div class="legend" id="legend3"></div>
    <div id="thumb"></div>
  </div>

  <h2>4. 예시 손 지도</h2>
  <p>잘 맞는 손(1559)과 안 맞는 손(1113)의 xy 평면. 검정이 SW 랜드마크, 파랑이 규칙 파이프라인, 주황이 학습 모델이며 선은 오차 벡터입니다. 회색 점은 메쉬 정점.</p>
  <div class="card"><img src="data:image/png;base64,__FIG4__" alt="예시 손 두 명의 xy 평면 랜드마크 지도: SW 랜드마크와 두 접근의 예측점, 오차 벡터"></div>

  <h2>5. 표 (평균 ± 표준편차 / 최대, mm)</h2>
  <div class="card tbl"><table>
    <thead>
      <tr><th class="l"></th><th class="l"></th><th class="g" colspan="2">규칙 파이프라인</th><th class="g" colspan="2">학습 모델</th><th class="g" colspan="2">기준선</th></tr>
      <tr><th class="l">점</th><th class="l">부류</th><th>평균 ± sd</th><th>최대</th><th>평균 ± sd</th><th>최대</th><th>평균 ± sd</th><th>최대</th></tr>
    </thead>
    <tbody>__TABLE__</tbody>
  </table></div>

  <h2>6. 방법과 읽을 때 주의할 점</h2>
  <ul>
    <li><b>규칙 파이프라인</b>: 특허 KR 10-1217207(손 자동계측 방법)의 실루엣 규칙을 메쉬 위에 재구성. 손목점 → 방사 외곽선 → 손가락 끝점 → web 점 → 손가락 축 → 기저점. 학습이 없어 15명 모두 동등한 조건. 손목 규칙은 메쉬가 손목에서 절단돼 있어 절단면+21 mm로 대체(근사). P6은 P1–P15 중점(근사), P24·P25는 특허 규칙 그대로.</li>
    <li><b>학습 모델</b>: 정점별 기하 특징 32개로 각 점까지의 거리와 오프셋을 부스팅 회귀(coarse+fine), 19개 실측점 각각 학습. 15-fold LOO(14명 학습 → 1명 예측). P1·P15는 별도의 정밀화(apex/atlas)가 더 좋으며 3번 그림에 함께 표시.</li>
    <li><b>기준선</b>: 손목 절단면 중심·손 축으로 만든 공통 좌표계에서 각 점의 평균 위치(다른 14명 평균). 손 크기 차이를 보정하지 않은 가장 단순한 추정.</li>
    <li><b>파생점 P7–P14, P28</b>은 SW가 다른 점에서 계산하는 점(손가락 끝→기저 3등분, 손목 중점)이라 예측한 부모점의 오차를 그대로 물려받습니다. 파생점 z는 SW 규칙대로 0(3등분)·1(P28)로 두었습니다.</li>
    <li><b>실루엣 점(P20–P27)</b>은 수직 모서리 위의 점이라 z가 불확정입니다. 3D 오차에는 z 차이가 들어가므로 xy만 보면 더 작습니다(예: 규칙 파이프라인 P20 3D 5.5 mm, xy 1.5 mm).</li>
    <li>표본이 15명이라 표준편차·최대값은 흔들립니다. 한 사람이 바뀌면 평균이 0.2–0.3 mm 움직일 수 있습니다.</li>
  </ul>
</div>
<div class="tip" id="tip" role="status" aria-live="polite"></div>
<script id="data" type="application/json">__DATA__</script>
<script>
(function(){
  const D = JSON.parse(document.getElementById('data').textContent);
  const tip = document.getElementById('tip');
  const cssColor = k => getComputedStyle(document.documentElement).getPropertyValue('--'+k).trim();
  function showTip(html, ev){ tip.innerHTML = html; tip.style.opacity = 1; moveTip(ev); }
  function moveTip(ev){ const x = ev.clientX + 14, y = ev.clientY + 14; tip.style.left = Math.min(x, window.innerWidth - 270) + 'px'; tip.style.top = y + 'px'; }
  function hideTip(){ tip.style.opacity = 0; }
  const NS = 'http://www.w3.org/2000/svg';
  const el = (n, a, parent) => { const e = document.createElementNS(NS, n); for (const k in a) e.setAttribute(k, a[k]); if (parent) parent.appendChild(e); return e; };

  // legends
  function legend(id, items){ const L = document.getElementById(id); L.innerHTML=''; for (const it of items){ const s = document.createElement('span'); s.textContent = it.label; s.style.setProperty('--c', 'var(--'+it.key+')'); L.appendChild(s);} }
  legend('legend1', D.approaches);
  legend('legend3', [{key:'rule',label:'규칙 파이프라인'},{key:'learned',label:'학습 모델 (19점 확장)'},{key:'best',label:'학습 모델 (P1·P15 전용 apex/atlas)'},{key:'baseline',label:'기준선'}]);

  // 1. grouped bars
  (function(){
    const W = 1080, H = 360, m = {l:44, r:12, t:30, b:34};
    const pw = W - m.l - m.r, ph = H - m.t - m.b;
    const n = D.points.length, gw = pw / n, bw = gw * 0.26;
    const ymax = 10;
    const y = v => m.t + ph - Math.min(v, ymax) / ymax * ph;
    const svg = el('svg', {viewBox:`0 0 ${W} ${H}`, role:'img', 'aria-label':'점별 평균 3D 오차, 접근별 막대'}, document.getElementById('bars'));
    // category bands + labels
    let start = 0;
    D.categories.forEach((c, i) => {
      const x0 = m.l + start * gw, x1 = m.l + (start + c.points.length) * gw;
      if (i % 2 === 1) el('rect', {x:x0, y:m.t, width:x1-x0, height:ph, class:'band'}, svg);
      const t = el('text', {x:(x0+x1)/2, y:m.t-10, 'text-anchor':'middle', class:'cat'}, svg); t.textContent = c.name;
      start += c.points.length;
    });
    for (let v = 0; v <= ymax; v += 2){ el('line', {x1:m.l, x2:W-m.r, y1:y(v), y2:y(v), class:'grid'}, svg); const t = el('text', {x:m.l-6, y:y(v)+4, 'text-anchor':'end'}, svg); t.textContent = v; }
    for (const [v, lab] of [[0.5,'0.5 mm 목표'],[2,'2 mm']]){ el('line', {x1:m.l, x2:W-m.r, y1:y(v), y2:y(v), class:'ref'}, svg); const t = el('text', {x:W-m.r-4, y:y(v)-4, 'text-anchor':'end'}, svg); t.textContent = lab; }
    el('line', {x1:m.l, x2:W-m.r, y1:y(0), y2:y(0), class:'axis'}, svg);
    D.points.forEach((p, i) => {
      const cx = m.l + (i + 0.5) * gw;
      const t = el('text', {x:cx, y:H-m.b+16, 'text-anchor':'middle'}, svg); t.textContent = p;
      D.approaches.forEach((a, j) => {
        const mean = D.summary[a.key].mean[i], sd = D.summary[a.key].sd[i], mx = D.summary[a.key].max[i];
        const x = cx + (j - 1) * bw - bw/2 + 1;
        const r = el('rect', {x:x, y:y(mean), width:bw-2, height:y(0)-y(mean), rx:2, fill:'var(--'+a.key+')'}, svg);
        el('line', {x1:x+(bw-2)/2, x2:x+(bw-2)/2, y1:y(mean+sd), y2:y(Math.max(0,mean-sd)), stroke:'var(--ink-2)', 'stroke-width':1}, svg);
        const hit = el('rect', {x:x-1, y:m.t, width:bw, height:ph, fill:'transparent'}, svg);
        const html = `<b>${p}</b> · ${a.label}<br>평균 ${mean.toFixed(2)} mm ± ${sd.toFixed(2)}<br>최대 ${mx.toFixed(1)} mm`;
        hit.addEventListener('mousemove', ev => showTip(html, ev)); hit.addEventListener('mouseleave', hideTip);
      });
    });
    const yl = el('text', {x:12, y:m.t+ph/2, transform:`rotate(-90 12 ${m.t+ph/2})`, 'text-anchor':'middle'}, svg); yl.textContent = '3D 오차 (mm)';
  })();

  // 2. heatmaps
  const ramp = ['seq0','seq1','seq2','seq3','seq4','seq5','seq6'];
  (function(){ const R = document.getElementById('ramp'); ramp.forEach(k => { const i = document.createElement('i'); i.style.background = 'var(--'+k+')'; R.appendChild(i); }); })();
  const colorFor = v => 'var(--' + ramp[Math.min(ramp.length-1, Math.floor(Math.min(v, 7.999) / 8 * ramp.length))] + ')';
  const tabs = document.getElementById('heat-tabs');
  function drawHeat(key){
    const G = document.getElementById('heat'); G.innerHTML = '';
    G.style.gridTemplateColumns = `auto repeat(${D.points.length}, minmax(24px, 1fr))`;
    G.appendChild(document.createElement('div'));
    D.points.forEach(p => { const h = document.createElement('div'); h.className='h'; h.textContent = p.slice(1); G.appendChild(h); });
    D.hands.forEach((h, hi) => {
      const r = document.createElement('div'); r.className = 'r' + (D.flagged[hi] ? ' flag' : ''); r.textContent = D.short[hi] + (D.flagged[hi] ? ' ▲' : ''); G.appendChild(r);
      D.errors[key][h].forEach((v, pi) => {
        const c = document.createElement('div'); c.className = 'cell'; c.tabIndex = 0; c.style.background = colorFor(v);
        const html = `<b>${D.short[hi]} · ${D.points[pi]}</b><br>${v.toFixed(2)} mm`;
        c.setAttribute('aria-label', `${D.short[hi]} ${D.points[pi]} ${v.toFixed(2)} mm`);
        c.addEventListener('mousemove', ev => showTip(html, ev)); c.addEventListener('mouseleave', hideTip);
        c.addEventListener('focus', ev => { const b = c.getBoundingClientRect(); showTip(html, {clientX:b.left, clientY:b.top}); }); c.addEventListener('blur', hideTip);
        G.appendChild(c);
      });
    });
    [...tabs.children].forEach(b => b.setAttribute('aria-selected', b.dataset.key === key));
  }
  D.approaches.forEach((a, i) => { const b = document.createElement('button'); b.role = 'tab'; b.dataset.key = a.key; b.textContent = a.label.split(' (')[0]; b.addEventListener('click', () => drawHeat(a.key)); tabs.appendChild(b); });
  drawHeat('rule');

  // 3. thumb length dots
  (function(){
    const keys = ['rule','learned','best','baseline'];
    const W = 1080, H = 300, m = {l:44, r:12, t:16, b:34};
    const pw = W - m.l - m.r, ph = H - m.t - m.b, n = D.hands.length, gw = pw / n;
    const lim = 7;
    const y = v => m.t + ph/2 - Math.max(-lim, Math.min(lim, v)) / lim * ph/2;
    const svg = el('svg', {viewBox:`0 0 ${W} ${H}`, role:'img', 'aria-label':'사람별 엄지 길이 오차, 접근별 점'}, document.getElementById('thumb'));
    el('rect', {x:m.l, y:y(0.5), width:pw, height:y(-0.5)-y(0.5), class:'band'}, svg);
    for (let v = -6; v <= 6; v += 2){ el('line', {x1:m.l, x2:W-m.r, y1:y(v), y2:y(v), class:'grid'}, svg); const t = el('text', {x:m.l-6, y:y(v)+4, 'text-anchor':'end'}, svg); t.textContent = (v>0?'+':'')+v; }
    el('line', {x1:m.l, x2:W-m.r, y1:y(0), y2:y(0), class:'axis'}, svg);
    const bt = el('text', {x:W-m.r-4, y:y(0.5)-4, 'text-anchor':'end'}, svg); bt.textContent = '±0.5 mm 목표';
    D.hands.forEach((h, i) => {
      const cx = m.l + (i + 0.5) * gw;
      const t = el('text', {x:cx, y:H-m.b+16, 'text-anchor':'middle'}, svg); t.textContent = D.short[i];
      keys.forEach((k, j) => {
        const v = D.thumb[k][i];
        const x = cx + (j - 1.5) * 9;
        el('circle', {cx:x, cy:y(v), r:5, fill:'var(--'+k+')', stroke:'var(--surface)', 'stroke-width':2}, svg);
        const hit = el('rect', {x:x-6, y:m.t, width:12, height:ph, fill:'transparent'}, svg);
        const html = `<b>${D.short[i]}</b> · ${k==='rule'?'규칙':k==='learned'?'학습(19점)':k==='best'?'학습(apex/atlas)':'기준선'}<br>오차 ${v>0?'+':''}${v.toFixed(2)} mm · 기준값 ${D.thumb_ref[i].toFixed(1)}`;
        hit.addEventListener('mousemove', ev => showTip(html, ev)); hit.addEventListener('mouseleave', hideTip);
      });
    });
    const yl = el('text', {x:12, y:m.t+ph/2, transform:`rotate(-90 12 ${m.t+ph/2})`, 'text-anchor':'middle'}, svg); yl.textContent = '예측 − 기준값 (mm)';
  })();
})();
</script>
"""


def main():
    os.makedirs(REPORT, exist_ok=True)
    err, hands, rule = load_errors()
    summ = summary_table(err)
    summ.to_csv(os.path.join(REPORT, "landmark_errors_summary.csv"), index=False, float_format="%.3f")
    tl, ref = thumb_lengths(hands, rule)
    plt = setup_mpl()
    fig_bars(plt, summ)
    fig_heatmaps(plt, err, hands)
    fig_thumb(plt, tl, hands)
    fig_hands(plt, hands)
    build_html(err, hands, summ, tl, ref)
    print("overall mean 3D error (mm):", {k: round(float(err[k][ORDER].values.mean()), 2) for k, _ in APPROACHES})
    print("thumb length MAE (mm):", {k: round(float(np.abs(v).mean()), 2) for k, v in tl.items()})
    print("wrote", REPORT)


if __name__ == "__main__":
    main()
