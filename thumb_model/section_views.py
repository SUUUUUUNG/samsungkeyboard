"""Cross-section shape (w-z curve in the plane perpendicular to the thumb axis) as a function of t,
vs the SW's P15 (t-rule candidates; rejected). True axis, 15 hands.

    python thumb_model/section_views.py   -> thumb_model/report/t_rule_sections/sections_<name>.png
"""
import glob, os, sys, numpy as np
ROOT = r"C:\Users\heony\Documents\samsungkeyboard"; sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "thumb_model"))
from align_landmarks import read_obj, sample_surface
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "Malgun Gothic"; plt.rcParams["axes.unicode_minus"] = False
def unit(v): v = np.asarray(v, float); return v / np.linalg.norm(v)
def smooth(y, k=2):
    out = np.full_like(y, np.nan)
    for i in range(len(y)):
        s = y[max(0, i - k):i + k + 1]; s = s[~np.isnan(s)]; out[i] = s.mean() if len(s) else np.nan
    return out
W = 24.0   # lateral half-range of the section (mm)
rows = []; prof = {}; secs = {}
for f in sorted(glob.glob(os.path.join(ROOT, "aligned/*_aligned.lnd"))):
    n = os.path.basename(f)[:-12]; A = np.loadtxt(f)[:, 1:]; g1, g15, g20 = A[0], A[14], A[19]
    V, F, _ = read_obj(os.path.join(ROOT, "obj", n + ".obj")); S = np.vstack([V, sample_surface(V, F)])
    u = unit(g15[:2] - g1[:2]); lat = np.array([-u[1], u[0]])
    if (g20[:2] - g1[:2]) @ lat < 0: lat = -lat            # inner (index) side = +w
    rel = S[:, :2] - g1[:2]; t = rel @ u; w = rel @ lat; z = S[:, 2]
    t15 = (g15[:2] - g1[:2]) @ u; t20 = (g20[:2] - g1[:2]) @ u
    ts = np.arange(10, 70, 1.0); names_f = ["w_bottom", "w_ridge", "roll_deg", "asym_z6", "contact_w", "lift_w", "top_w_outer", "z_ridge", "bottom_flat_w", "inner_h", "outer_h"]
    P = {k: np.full(len(ts), np.nan) for k in names_f}; secs[n] = {}
    for i, t0 in enumerate(ts):
        m = (np.abs(t - t0) < 0.75) & (np.abs(w) < W)
        if m.sum() < 10: continue
        ww, zz = w[m], z[m]
        bins = np.arange(-W, W + 1, 1.0); ztop = np.full(len(bins) - 1, np.nan); zbot = ztop.copy()
        for b in range(len(bins) - 1):
            k = (ww >= bins[b]) & (ww < bins[b + 1])
            if k.sum(): ztop[b] = zz[k].max(); zbot[b] = zz[k].min()
        wc = bins[:-1] + 0.5; ok = ~np.isnan(ztop)
        if ok.sum() < 5: continue
        if int(t0) % 5 == 0: secs[n][int(t0)] = (wc[ok], ztop[ok], zbot[ok])
        P["w_bottom"][i] = wc[ok][np.argmin(zbot[ok])]                         # lateral position of the lowest point
        P["w_ridge"][i] = wc[ok][np.argmax(ztop[ok])]                          # lateral position of the ridge (highest point)
        P["z_ridge"][i] = ztop[ok].max()
        # roll: PCA of the top-surface points (w, z) within |w| < 10
        k10 = ok & (np.abs(wc) < 10)
        if k10.sum() >= 5:
            X = np.c_[wc[k10], ztop[k10]]; X = X - X.mean(0); _, _, vt = np.linalg.svd(X, full_matrices=False)
            P["roll_deg"][i] = np.degrees(np.arctan2(vt[0][1], vt[0][0]))
        def zat(wq, arr):
            j = np.argmin(np.abs(wc - wq)); return arr[j] if abs(wc[j] - wq) < 1 else np.nan
        P["asym_z6"][i] = zat(6, ztop) - zat(-6, ztop)                          # inner side higher than outer?
        P["inner_h"][i] = zat(8, ztop); P["outer_h"][i] = zat(-8, ztop)
        low = ok & (zbot < 1.0); P["contact_w"][i] = low.sum()                 # width (mm) touching the glass (z < 1)
        lifted = ok & (zbot > 1.0); P["lift_w"][i] = lifted.sum()               # width lifted off the glass
        P["bottom_flat_w"][i] = (ok & (zbot < zbot[ok].min() + 0.5)).sum()      # width of the flat bottom around the min
        # outer boundary of the section (most negative w with data)
        P["top_w_outer"][i] = wc[ok].min()
    for k in P: P[k] = smooth(P[k])
    prof[n] = (ts, P, t15, t20)
    lo, hi = t20 - 14, t20 + 14
    def arg(fn, y, lo, hi):
        k = np.where((ts >= lo) & (ts <= hi) & ~np.isnan(y))[0]; return ts[k[fn(y[k])]] if len(k) else np.nan
    feats = {}
    for k in names_f:
        y = P[k]; d = np.gradient(y, ts)
        feats[k + " max"] = arg(np.argmax, y, lo, hi); feats[k + " min"] = arg(np.argmin, y, lo, hi)
        feats[k + " d/dt max"] = arg(np.argmax, d, lo, hi); feats[k + " d/dt min"] = arg(np.argmin, d, lo, hi)
    rows.append((n, t15, t20, feats))
keys = list(rows[0][3].keys())
print("t15 - feature (window t20 +/- 14): mean +/- sd, sorted by sd (only sd < 3.0 shown, plus the count of window-edge hits)")
out = []
for k in keys:
    v = np.array([r[1] - r[3][k] for r in rows]); edge = sum(1 for r in rows if abs(r[3][k] - (r[2] - 14)) < 0.5 or abs(r[3][k] - (r[2] + 14)) < 0.5)
    out.append((np.nanstd(v), k, np.nanmean(v), edge))
for sd, k, mu, edge in sorted(out):
    if sd < 3.0: print(f"  {k:26s} {mu:+6.2f} +/- {sd:4.2f}   window-edge hits {edge}/15")
print(f"  (reference: P20 foot -0.42 +/- 2.37)")
# ---- figures: section stacks + feature curves for two hands
for n in ("20_F_0004G", "20_M_1113G"):
    ts, P, t15, t20 = prof[n]
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    ax = axes[0]
    for j, (t0, (wc, zt, zb)) in enumerate(sorted(secs[n].items())):
        if t0 < 25 or t0 > 65: continue
        col = plt.cm.viridis((t0 - 25) / 40)
        ax.plot(wc, zt + 0.0, color=col, lw=1.4, label=f"t={t0}" + ("  ← P15 근처" if abs(t0 - t15) <= 2.5 else ""))
        ax.plot(wc, zb, color=col, lw=0.8, ls="--")
    ax.axvline(0, color="#aaa", lw=0.8); ax.set_xlabel("w: 축에 수직 옆 방향 (mm, + = 검지 쪽)"); ax.set_ylabel("z (mm)")
    ax.set_title(f"{n}: t별 단면 (실선 = 위 표면, 파선 = 아래 표면)   t15={t15:.1f}, t20={t20:.1f}", loc="left", fontsize=10)
    ax.legend(fontsize=7.5, ncol=2); ax.set_aspect("equal")
    ax = axes[1]
    for k, col in (("w_bottom", "#2a78d6"), ("w_ridge", "#eb6834"), ("roll_deg", "#1a9850"), ("asym_z6", "#7a3fbf"), ("contact_w", "#0b0b0b"), ("lift_w", "#898781")):
        y = P[k]; y = (y - np.nanmean(y)) / (np.nanstd(y) + 1e-9)
        ax.plot(ts, y, color=col, lw=1.3, label=k)
    ax.axvline(t15, color="k", lw=1.4, label="정답 P15 t"); ax.axvline(t20, color="#00a0b0", ls=":", lw=1.2, label="P20 발 t")
    ax.set_xlabel("t (mm)"); ax.set_ylabel("표준화한 값"); ax.set_title("단면 지표의 t 프로파일 (표준화)", loc="left", fontsize=10); ax.legend(fontsize=8); ax.grid(color="#e1e0d9")
    fig.tight_layout(); fig.savefig(os.path.join(r"C:\Users\heony\AppData\Local\Temp\claude\c--Users-heony-Documents-samsungkeyboard\ff55d9d4-d37c-4ac9-9cb1-ed5b6c663370\scratchpad", f"sections_{n}.png"), dpi=110); plt.close(fig)
print("figures written")
