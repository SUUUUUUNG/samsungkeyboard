"""Align hand landmarks (lnd/*.lnd) onto matching hand meshes (obj/*.obj).

For each file pair with the same basename, a rigid transform (rotation +
translation) is estimated with trimmed ICP against the mesh surface, applied
to the landmarks, and the results are written to aligned/.
"""
import csv
import glob
import os

import numpy as np
from scipy.spatial import cKDTree

ROOT = os.path.dirname(os.path.abspath(__file__))
LND_DIR = os.path.join(ROOT, "lnd")
OBJ_DIR = os.path.join(ROOT, "obj")
OUT_DIR = os.path.join(ROOT, "aligned")

SAMPLES_PER_MM2 = 4.0
# use every fitted landmark each ICP step: trimming the worst few let the fit
# slide proximally along the fingers (tubes) and settle in a wrong minimum
TRIM_RATIO = 1.0
# extra ICP starts shifted along the mesh hand axis (mm), to escape that minimum
HAND_SHIFTS_MM = range(-6, 18, 2)
RMS_WARN_MM = 5.0
SPHERE_RADIUS = 1.5

# landmarks whose z is a filled-in value, not a 3D measurement (7-14: z=0,
# 28: z=1 at the xy midpoint of 26/27); excluded from fitting, but still
# transformed and written out
NON_3D_LANDMARKS = set(range(7, 15)) | {28}

THUMB_TIP, THUMB_BASE = 1, 15
THUMB_MATCH_TOL_MM = 0.1
# measured thumb straight length (mm), keyed by HUMAN_ID
REFERENCE_THUMB_MM = {
    "20_F_0004": 43.1, "20_F_0097": 53.8, "20_F_0179": 45.6,
    "20_F_2633": 46.7, "20_F_2634": 50.2, "20_F_2635": 44.6,
    "20_F_2636": 46.4, "20_F_2637": 50.0,
    "20_M_1113": 49.2, "20_M_1114": 53.5, "20_M_1115": 50.4,
    "20_M_1116": 55.1, "20_M_1117": 48.4, "20_M_1559": 56.7,
    "20_M_1573": 48.2,
}


def read_obj(path):
    verts, faces, lines = [], [], []
    with open(path) as fh:
        for line in fh:
            lines.append(line)
            if line.startswith("v "):
                verts.append(line.split()[1:4])
            elif line.startswith("f "):
                faces.append([int(tok.split("/")[0]) - 1 for tok in line.split()[1:4]])
    return np.array(verts, float), np.array(faces, int), lines


def read_lnd(path):
    data = np.loadtxt(path)
    return data[:, 0].astype(int), data[:, 1:4]


def sample_surface(V, F, seed=0):
    a, b, c = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]
    area = 0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1)
    n = int(area.sum() * SAMPLES_PER_MM2)
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(F), size=n, p=area / area.sum())
    u, v = rng.random(n), rng.random(n)
    flip = u + v > 1
    u[flip], v[flip] = 1 - u[flip], 1 - v[flip]
    pts = a[idx] + u[:, None] * (b[idx] - a[idx]) + v[:, None] * (c[idx] - a[idx])
    return np.vstack([V, pts])


def hand_axis(V):
    """Mesh-only hand frame: wrist-cut centre (xy) and unit axis towards the
    longest fingertip (max y), z component zero."""
    wrist = V[V[:, 1] < V[:, 1].min() + 3]
    origin = np.array([*wrist[:, :2].mean(0), 0.0])
    tip = V[np.argmax(V[:, 1])]
    axis = np.array([tip[0] - origin[0], tip[1] - origin[1], 0.0])
    return origin, axis / np.linalg.norm(axis)


def tip_apex_gap(X, V, tube_radius=12.0):
    """How far (mm) the thumb-tip landmark P1 sits short of the mesh's thumb
    apex, measured along the P15->P1 axis in xy. Near 0 when P1 is at the tip."""
    p1, p15 = X[0], X[14]
    d = p1 - p15
    d[2] = 0
    L = np.linalg.norm(d)
    u = d / L
    lat = np.array([-u[1], u[0], 0.0])
    rel = V - p15
    t = rel @ u
    perp = np.sqrt((rel @ lat) ** 2 + (rel[:, 2] - p1[2]) ** 2)
    tube = (perp < tube_radius) & (t > L - 15)
    if not tube.any():
        return np.nan
    return t[tube].max() - L


def kabsch(P, Q, w):
    # rotation restricted to the z axis: both hands rest palm-down on z=0,
    # and this keeps xy-plane measurements (e.g. thumb length) unchanged
    w = w / w.sum()
    pc, qc = w @ P, w @ Q
    p, q = P[:, :2] - pc[:2], Q[:, :2] - qc[:2]
    cross = w @ (p[:, 0] * q[:, 1] - p[:, 1] * q[:, 0])
    dot = w @ (p[:, 0] * q[:, 0] + p[:, 1] * q[:, 1])
    R = rot_z(np.degrees(np.arctan2(cross, dot)))
    return R, qc - R @ pc


def icp(P, tree, surf, T0, iters=100, tol=1e-6):
    T = T0.copy()
    prev = np.inf
    for _ in range(iters):
        X = P @ T[:3, :3].T + T[:3, 3]
        d, j = tree.query(X)
        keep = d <= np.quantile(d, TRIM_RATIO)
        w = keep.astype(float)
        R, t = kabsch(X, surf[j], w)
        step = np.eye(4)
        step[:3, :3], step[:3, 3] = R, t
        T = step @ T
        err = np.sqrt(np.mean(d[keep] ** 2))
        if abs(prev - err) < tol:
            break
        prev = err
    X = P @ T[:3, :3].T + T[:3, 3]
    d, _ = tree.query(X)
    trimmed = np.sort(d)[: max(1, int(len(d) * TRIM_RATIO))]
    return T, np.sqrt(np.mean(trimmed ** 2)), d


def rot_z(deg):
    r = np.radians(deg)
    c, s = np.cos(r), np.sin(r)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])


def align(P, V, F):
    surf = sample_surface(V, F)
    tree = cKDTree(surf)
    pc = P.mean(0)
    vc = V.mean(0)
    best = {}
    for mirror in (False, True):
        M = np.diag([-1.0, 1, 1]) if mirror else np.eye(3)
        cands = []
        for ang in range(0, 360, 10):
            T0 = np.eye(4)
            R0 = rot_z(ang) @ M
            T0[:3, :3] = R0
            # center xy on mesh centroid; keep z (both rest on z=0)
            T0[:3, 3] = np.array([vc[0], vc[1], 0]) - R0 @ np.array([pc[0], pc[1], 0])
            T, _, d = icp(P, tree, surf, T0, iters=30)
            T, _, d = icp(P, tree, surf, T)
            # score on all landmarks so wrong minima with a few far-off points lose
            cands.append((np.sqrt(np.mean(d ** 2)), ang, T))
        _, ang, T = min(cands, key=lambda c: c[0])
        # restart shifted along the hand axis: fingers are tubes, so a fit
        # that sits a few mm too proximal looks almost as good locally
        _, axis = hand_axis(V)
        shifted = []
        for s in HAND_SHIFTS_MM:
            T0 = T.copy()
            T0[:3, 3] += s * axis
            T1, rms1, d1 = icp(P, tree, surf, T0)
            shifted.append((d1.mean(), rms1, T1, d1))
        _, rms, T, d = min(shifted, key=lambda c: c[0])
        best[mirror] = (rms, T, d)
    return best[False], best[True][0], tree


def unit_sphere(n_lat=6, n_lon=10):
    verts = [(0, 0, 1)]
    for i in range(1, n_lat):
        th = np.pi * i / n_lat
        for j in range(n_lon):
            ph = 2 * np.pi * j / n_lon
            verts.append((np.sin(th) * np.cos(ph), np.sin(th) * np.sin(ph), np.cos(th)))
    verts.append((0, 0, -1))
    faces = []
    for j in range(n_lon):
        faces.append((0, 1 + j, 1 + (j + 1) % n_lon))
    for i in range(n_lat - 2):
        r0, r1 = 1 + i * n_lon, 1 + (i + 1) * n_lon
        for j in range(n_lon):
            j1 = (j + 1) % n_lon
            faces.append((r0 + j, r1 + j, r1 + j1))
            faces.append((r0 + j, r1 + j1, r0 + j1))
    last = len(verts) - 1
    base = 1 + (n_lat - 2) * n_lon
    for j in range(n_lon):
        faces.append((base + j, last, base + (j + 1) % n_lon))
    return np.array(verts), np.array(faces)


def landmark_color(i):
    if i <= 5:
        return (255, 40, 40)  # fingertips
    if i <= 25:
        return (40, 200, 40)  # joints / creases
    return (40, 80, 255)  # wrist


def write_outputs(name, ids, X, T, rms, fit_max, dist, obj_lines, n_verts):
    with open(os.path.join(OUT_DIR, f"{name}_aligned.lnd"), "w") as fh:
        for i, p in zip(ids, X):
            fh.write(f"{i} {p[0]:.6f} {p[1]:.6f} {p[2]:.6f}\n")

    with open(os.path.join(OUT_DIR, f"{name}_landmarks.ply"), "w") as fh:
        fh.write("ply\nformat ascii 1.0\n")
        fh.write(f"element vertex {len(X)}\n")
        fh.write("property float x\nproperty float y\nproperty float z\n")
        fh.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        fh.write("end_header\n")
        for i, p in zip(ids, X):
            r, g, b = landmark_color(i)
            fh.write(f"{p[0]:.6f} {p[1]:.6f} {p[2]:.6f} {r} {g} {b}\n")

    sv, sf = unit_sphere()
    with open(os.path.join(OUT_DIR, f"{name}_combined.obj"), "w") as fh:
        for line in obj_lines:
            if not line.startswith(("mtllib", "usemtl")):
                fh.write(line)
        offset = n_verts
        for i, p in zip(ids, X):
            fh.write(f"\ng landmark_{i:02d}\n")
            for q in sv * SPHERE_RADIUS + p:
                fh.write(f"v {q[0]:.6f} {q[1]:.6f} {q[2]:.6f}\n")
            for f in sf + offset + 1:
                fh.write(f"f {f[0]} {f[1]} {f[2]}\n")
            offset += len(sv)

    with open(os.path.join(OUT_DIR, f"{name}_transform.txt"), "w") as fh:
        fh.write("# 4x4 rigid transform: X_obj = T @ [X_lnd, 1]\n")
        np.savetxt(fh, T, fmt="%.9f")
        fh.write(f"# fitted on 3D-measured landmarks only; excluded: "
                 f"{', '.join(str(i) for i in sorted(NON_3D_LANDMARKS))}\n")
        fh.write(f"# RMS of fitted landmarks (mm): {rms:.4f}\n")
        fh.write(f"# max distance of fitted landmarks (mm): {fit_max:.4f}\n")
        for i, d in zip(ids, dist):
            note = " (not fitted, z filled-in)" if i in NON_3D_LANDMARKS else ""
            fh.write(f"# landmark {i}: {d:.4f}{note}\n")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    rows, thumb_rows = [], []
    for lnd_path in sorted(glob.glob(os.path.join(LND_DIR, "*.lnd"))):
        name = os.path.splitext(os.path.basename(lnd_path))[0]
        obj_path = os.path.join(OBJ_DIR, name + ".obj")
        if not os.path.exists(obj_path):
            print(f"[skip] {name}: no matching obj")
            continue
        ids, P = read_lnd(lnd_path)
        V, F, lines = read_obj(obj_path)
        fit = np.array([i not in NON_3D_LANDMARKS for i in ids])
        (rms, T, fit_dist), mirror_rms, tree = align(P[fit], V, F)
        X = P @ T[:3, :3].T + T[:3, 3]
        dist, _ = tree.query(X)
        fit_max = fit_dist.max()
        fit_mean = fit_dist.mean()
        apex_gap = tip_apex_gap(X, V)
        write_outputs(name, ids, X, T, rms, fit_max, dist, lines, len(V))

        angle = np.degrees(np.arctan2(T[1, 0], T[0, 0]))
        flags = []
        if rms > RMS_WARN_MM:
            flags.append("HIGH_RMS")
        if mirror_rms < 0.8 * rms:
            flags.append("MIRROR_FITS_BETTER")
        tip = X[list(ids).index(THUMB_TIP)]
        base = X[list(ids).index(THUMB_BASE)]
        thumb = np.linalg.norm(tip[:2] - base[:2])
        human_id = name[:-1] if name.endswith("G") else name
        ref = REFERENCE_THUMB_MM.get(human_id)
        diff = thumb - ref if ref is not None else None
        match = diff is not None and abs(diff) <= THUMB_MATCH_TOL_MM
        thumb_rows.append([name, human_id, f"{thumb:.2f}", "" if ref is None else f"{ref:.1f}",
                           "" if diff is None else f"{diff:+.2f}", match])

        print(f"{name}: rms={rms:.3f}mm mean={fit_mean:.3f}mm max={fit_max:.3f}mm "
              f"apex_gap={apex_gap:.2f}mm rotZ={angle:.1f}deg tz={T[2, 3]:.2f}mm "
              f"mirror_rms={mirror_rms:.3f} thumb={thumb:.2f}mm ref={ref} {' '.join(flags)}")
        rows.append([name, f"{rms:.4f}", f"{fit_mean:.4f}", f"{fit_max:.4f}", f"{apex_gap:.3f}",
                     f"{angle:.2f}", f"{mirror_rms:.4f}", f"{thumb:.2f}", " ".join(flags)])

    with open(os.path.join(OUT_DIR, "summary.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["name", "rms_fitted_mm", "mean_dist_fitted_mm", "max_dist_fitted_mm",
                    "tip_apex_gap_mm", "rot_z_deg", "mirror_rms_mm", "thumb_length_mm", "flags"])
        w.writerows(rows)

    with open(os.path.join(ROOT, "thumb_length.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["name", "human_id", "thumb_length_mm", "reference_mm", "diff_mm", "match"])
        w.writerows(thumb_rows)


if __name__ == "__main__":
    main()
