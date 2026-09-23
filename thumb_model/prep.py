"""Per-vertex geometric features and landmark labels for the thumb-length model.

Each obj mesh is moved into a mesh-only canonical frame (wrist-cut centre at
the origin, hand axis along +y, z unchanged), per-vertex features are computed
there, and the aligned P1/P15 landmarks are turned into heatmap + offset
labels. Everything is cached in thumb_model/cache/<name>.npz.
"""
import glob
import os
import sys
import time

import numpy as np
from scipy.spatial import cKDTree

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from align_landmarks import OBJ_DIR, OUT_DIR as ALIGNED_DIR, hand_axis, read_obj, rot_z  # noqa: E402

CACHE_DIR = os.path.join(HERE, "cache")
LANDMARKS = {"P1": 0, "P15": 14}  # row index in the .lnd file
HEAT_SIGMA = 5.0    # mm, coarse heatmap width
OFFSET_RADIUS = 20.0  # mm, vertices that get an offset-vector label
KNN = (12, 40, 120)   # neighbourhood sizes (~3, 6, 10 mm radius at this vertex spacing)


def canonical_frame(V):
    """Rotation (about z) and origin so that (V - origin) @ R.T has the wrist
    cut at the origin and the hand axis along +y."""
    origin, axis = hand_axis(V)
    R = rot_z(90.0 - np.degrees(np.arctan2(axis[1], axis[0])))
    return R, origin


def to_canonical(P, R, origin):
    return (P - origin) @ R.T


def from_canonical(P, R, origin):
    return P @ R + origin


def vertex_normals(V, F):
    a, b, c = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]
    fn = np.cross(b - a, c - a)  # area-weighted face normals
    N = np.zeros_like(V)
    for k in range(3):
        np.add.at(N, F[:, k], fn)
    N /= np.linalg.norm(N, axis=1, keepdims=True) + 1e-12
    # orient outward: the top of the hand should face up
    top = V[:, 2] > np.percentile(V[:, 2], 99)
    if N[top, 2].mean() < 0:
        N = -N
    return N


def vertex_features(Vc, N):
    """Per-vertex feature matrix (float32) in the canonical frame + names."""
    tree = cKDTree(Vc)
    hand_len = Vc[:, 1].max()
    feats = [Vc, Vc[:, :2] / hand_len, N]
    names = ["x", "y", "z", "x_rel", "y_rel", "nx", "ny", "nz"]

    for k in KNN:
        d, idx = tree.query(Vc, k=k + 1)
        nb_idx = idx[:, 1:]
        rel = Vc[nb_idx] - Vc[:, None, :]
        cen = rel.mean(1)
        X = rel - cen[:, None, :]
        cov = np.einsum("nki,nkj->nij", X, X) / k
        ev = np.maximum(np.linalg.eigvalsh(cov)[:, ::-1], 1e-12)  # descending
        ndot = np.einsum("nki,ni->nk", N[nb_idx], N).mean(1)
        height = np.einsum("ni,ni->n", cen, N)  # <0 where the surface is convex
        feats += [ev[:, 1:2] / ev[:, 0:1], ev[:, 2:3] / ev[:, 0:1],
                  ndot[:, None], height[:, None], d[:, -1:]]
        names += [f"k{k}_ev21", f"k{k}_ev31", f"k{k}_ndot", f"k{k}_height", f"k{k}_radius"]

    # thickness of the hand in the xy column through the vertex
    col = cKDTree(Vc[:, :2]).query_ball_point(Vc[:, :2], 2.0)
    zmin = np.array([Vc[p, 2].min() for p in col])
    zmax = np.array([Vc[p, 2].max() for p in col])
    feats += [(zmax - zmin)[:, None], (Vc[:, 2] - zmin)[:, None], (zmax - Vc[:, 2])[:, None]]
    names += ["col_thick", "z_above_col_min", "z_below_col_max"]

    # distances to mesh-only reference points
    refs = {"d_xmin": Vc[np.argmin(Vc[:, 0])], "d_xmax": Vc[np.argmax(Vc[:, 0])],
            "d_ymax": Vc[np.argmax(Vc[:, 1])], "d_centroid": Vc.mean(0)}
    for nm, p in refs.items():
        feats.append(np.linalg.norm(Vc - p, axis=1)[:, None])
        names.append(nm)
    feats.append(np.linalg.norm(Vc[:, :2], axis=1)[:, None])
    names.append("d_wrist")
    feats.append(np.arctan2(Vc[:, 0], Vc[:, 1])[:, None])
    names.append("polar_angle")
    return np.hstack(feats).astype(np.float32), names


def prepare_mesh(obj_path):
    """Everything the model needs from one obj file (no labels)."""
    V, F, _ = read_obj(obj_path)
    R, origin = canonical_frame(V)
    Vc = to_canonical(V, R, origin)
    N = vertex_normals(V, F) @ R.T
    feats, names = vertex_features(Vc, N)
    return {"V": V, "F": F, "Vc": Vc, "N": N, "feats": feats, "feat_names": names,
            "R": R, "origin": origin}


def landmark_labels(Vc, p, sigma=HEAT_SIGMA):
    d = np.linalg.norm(Vc - p, axis=1)
    heat = np.exp(-d ** 2 / (2 * sigma ** 2))
    return d.astype(np.float32), heat.astype(np.float32), (p - Vc).astype(np.float32)


def main():
    os.makedirs(CACHE_DIR, exist_ok=True)
    for obj_path in sorted(glob.glob(os.path.join(OBJ_DIR, "*.obj"))):
        name = os.path.splitext(os.path.basename(obj_path))[0]
        t0 = time.time()
        m = prepare_mesh(obj_path)
        out = {k: m[k] for k in ("V", "F", "Vc", "N", "feats", "R", "origin")}
        out["feat_names"] = np.array(m["feat_names"])
        lnd_path = os.path.join(ALIGNED_DIR, f"{name}_aligned.lnd")
        if os.path.exists(lnd_path):
            L = np.loadtxt(lnd_path)[:, 1:4]
            for lm, row in LANDMARKS.items():
                p = to_canonical(L[row], m["R"], m["origin"])
                out[f"{lm}_gt"] = p
                out[f"{lm}_dist"], out[f"{lm}_heat"], out[f"{lm}_off"] = landmark_labels(m["Vc"], p)
        np.savez_compressed(os.path.join(CACHE_DIR, f"{name}.npz"), **out)
        nan = int(np.isnan(m["feats"]).sum())
        print(f"{name}: {len(m['Vc'])} verts, {m['feats'].shape[1]} feats, nan={nan}, "
              f"{time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
