#!/usr/bin/env python3
"""Day 5: point-to-plane ICP with estimated normals and nearest neighbors.

The corridor is still synthetic, but the registration never sees the plane
equations. A map is a noisy point cloud. Normals come from a local PCA.
Correspondences are nearest neighbors. Gauss–Newton starts at the IMU guess.
The factor uses the Hessian at convergence.

Run:
    python3 run_estimated_normals.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import cKDTree

OUT = Path(__file__).resolve().parent / "out"

GAMMA_MID = 0.3
SIGMOID_TAU = 0.05
# Both the map and the scan are noisy, so a point-to-plane residual has
# variance 2 σ^2. One point with a perfect normal contributes 1 / (2 σ^2).
POINT_SIGMA = 0.02
RES_VAR = 2.0 * POINT_SIGMA ** 2
YAW_DEG = 35.0
DX = 1.0
N_POSES = 21
N_EDGES = N_POSES - 1
N_TRIALS = 40
N_DIAG = 80
SEG_LO = 8
SEG_HI = 15
PCA_K = 48
PCA_RADIUS = 0.45
ASSOC_GATE = 0.25
PLANAR_MAX = 0.02
MIN_NEIGHBORS = 12
# A direction with less than this fraction of the strongest information is
# treated as unobservable in the pose update. Its information is still written
# into the factor. This stops a near-null direction from exploding.
EIG_CUT_RATIO = 0.002
MAX_ITERS = 8
# IMU information is the analytic information of the partial end cap, so that
# patch sits near γ = 0.5 when its normals are accurate. Frozen before trials.
N_GRAY = 36
LAMBDA_IMU = N_GRAY / RES_VAR
LAMBDA_LOOP = 20.0 * LAMBDA_IMU


def sigmoid_weight(gamma: float) -> float:
    return float(1.0 / (1.0 + np.exp(-(gamma - GAMMA_MID) / SIGMOID_TAU)))


def world_from_lidar(yaw_deg: float) -> np.ndarray:
    yaw = np.deg2rad(yaw_deg)
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def _grid(a, b, step):
    n = int(np.floor((b - a) / step)) + 1
    return np.linspace(a, a + step * (n - 1), n)


def cap_points(n_cap: int) -> np.ndarray:
    """End-cap patch. n_cap == 0 is an open corridor. The partial cap is N_GRAY points."""
    if n_cap <= 0:
        return np.zeros((0, 3))
    if n_cap == N_GRAY:
        xs = np.linspace(-0.30, 0.30, 6)
        zs = np.linspace(-0.30, 0.30, 6)
    else:
        xs = _grid(-1.5, 1.5, 0.12)
        zs = _grid(-0.9, 0.9, 0.12)
    xx, zz = np.meshgrid(xs, zs, indexing="ij")
    return np.stack([xx.ravel(), np.full(xx.size, 5.0), zz.ravel()], axis=1)


def surface_points(n_cap: int, rng, noise: float) -> np.ndarray:
    """Corridor samples in the reference sensor frame. No plane coefficients leave this function."""
    chunks = []
    y = _grid(-4.0, 9.0, 0.125)
    z = _grid(-1.0, 1.0, 0.125)
    yy, zz = np.meshgrid(y, z, indexing="ij")
    wall = np.stack([np.full(yy.size, -2.5), yy.ravel(), zz.ravel()], axis=1)
    chunks.append(wall)
    chunks.append(wall * np.array([-1.0, 1.0, 1.0]))
    x = _grid(-1.6, 1.6, 0.125)
    y_g = _grid(-4.0, 9.0, 0.125)
    xx, yy = np.meshgrid(x, y_g, indexing="ij")
    chunks.append(np.stack([xx.ravel(), yy.ravel(), np.full(xx.size, -1.5)], axis=1))
    cap = cap_points(n_cap)
    if len(cap):
        chunks.append(cap)
    pts = np.vstack(chunks)
    if noise > 0:
        pts = pts + rng.normal(0.0, noise, size=pts.shape)
    return pts


def estimate_normals(pts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Local PCA inside a fixed radius, so a small end cap does not borrow wall points."""
    tree = cKDTree(pts)
    k = min(PCA_K, len(pts))
    dists, idx = tree.query(pts, k=k)
    neigh = pts[idx]
    keep = dists < PCA_RADIUS
    count = np.maximum(keep.sum(axis=1), 1)
    mean = (neigh * keep[..., None]).sum(axis=1) / count[:, None]
    centered = (neigh - mean[:, None, :]) * keep[..., None]
    cov = np.einsum("nki,nkj->nij", centered, centered) / count[:, None, None]
    evals, evecs = np.linalg.eigh(cov)
    normals = evecs[:, :, 0].copy()
    toward = -pts
    flip = np.einsum("ni,ni->n", normals, toward) < 0.0
    normals[flip] *= -1.0
    planar = (count >= MIN_NEIGHBORS) & (
        evals[:, 0] / np.maximum(evals[:, 2], 1e-12) < PLANAR_MAX
    )
    return normals, planar


def make_map(n_cap: int, rng):
    pts = surface_points(n_cap, rng, POINT_SIGMA)
    normals, planar = estimate_normals(pts)
    pts = pts[planar]
    normals = normals[planar]
    return pts, normals, cKDTree(pts)


def gn_icp(scan_l, map_pts, map_n, tree, t_init: np.ndarray):
    """World-frame translation. Rotation is the known sensor yaw."""
    R = world_from_lidar(YAW_DEG)
    t = np.asarray(t_init, dtype=float).copy()
    n = np.zeros((0, 3))
    n_used = 0
    for _ in range(MAX_ITERS):
        x = (R @ scan_l.T).T + t
        dist, idx = tree.query(x, k=1, workers=1)
        ok = dist < ASSOC_GATE
        if int(np.count_nonzero(ok)) < 10:
            break
        n = map_n[idx[ok]]
        q = map_pts[idx[ok]]
        r = np.einsum("ij,ij->i", n, x[ok] - q)
        # Information, not the raw Gram matrix.
        H_w = (n.T @ n) / RES_VAR
        g = (n.T @ r) / RES_VAR
        evals, evecs = np.linalg.eigh(H_w)
        delta = np.zeros(3)
        step = False
        cut = EIG_CUT_RATIO * float(evals[-1])
        for i, lam in enumerate(evals):
            if lam > cut:
                delta -= float(np.dot(evecs[:, i], g)) / lam * evecs[:, i]
                step = True
        if not step:
            break
        t = t + delta
        n_used = int(np.count_nonzero(ok))
        if np.linalg.norm(delta) < 1e-10:
            break
    n_l = (R.T @ n.T).T if n_used else np.zeros((0, 3))
    if n_used:
        H_l = (n_l.T @ n_l) / RES_VAR
        H_from_lidar = R @ H_l @ R.T
        H_xy = H_from_lidar[:2, :2]
    else:
        H_xy = np.zeros((2, 2))
    return t, H_xy, n_used


def register_step(n_cap: int, z_imu: np.ndarray, rng):
    map_pts, map_n, tree = make_map(n_cap, rng)
    # Independent sample of the same surfaces, seen from the true pose.
    scan_w = surface_points(n_cap, rng, POINT_SIGMA)
    t_true = np.array([0.0, DX, 0.0])
    R = world_from_lidar(YAW_DEG)
    scan_l = (R.T @ (scan_w - t_true).T).T
    t_init = np.array([z_imu[0], z_imu[1], 0.0])
    t_hat, H_xy, n_used = gn_icp(scan_l, map_pts, map_n, tree, t_init)
    return t_hat[:2], H_xy, n_used


def gamma_of(H: np.ndarray) -> tuple[float, float]:
    ev = np.linalg.eigvalsh(H)
    weak = float(ev[0])
    return weak / (weak + LAMBDA_IMU), weak


def factor_information(method: str, H_world: np.ndarray, lambda_nominal: float) -> np.ndarray:
    evals, evecs = np.linalg.eigh(0.5 * (H_world + H_world.T))
    evals = np.maximum(evals, 0.0)
    gammas = evals / (evals + LAMBDA_IMU)
    if method == "fixed":
        return np.eye(2) * lambda_nominal
    if method == "irr":
        return (evecs * evals) @ evecs.T
    if method == "hard":
        scale = np.array([lambda_nominal if g >= GAMMA_MID else 0.0 for g in gammas])
    elif method == "sigmoid":
        scale = np.array([sigmoid_weight(float(g)) * lambda_nominal for g in gammas])
    else:
        raise KeyError(method)
    return (evecs * scale) @ evecs.T


def fuse(z_a, La, z_b, Lb):
    L = La + Lb
    z = np.linalg.solve(L, La @ z_a + Lb @ z_b)
    return z, L


def add_between(H, rhs, i, j, z, L):
    L = np.asarray(L, dtype=float).reshape(2, 2)

    def sl(k):
        return slice(2 * (k - 1), 2 * k)

    if i == 0:
        H[sl(j), sl(j)] += L
        rhs[sl(j)] += L @ z
        return
    H[sl(i), sl(i)] += L
    H[sl(j), sl(j)] += L
    H[sl(i), sl(j)] -= L
    H[sl(j), sl(i)] -= L
    rhs[sl(i)] -= L @ z
    rhs[sl(j)] += L @ z


def solve_chain(meas, infos, loop_z):
    n_free = N_EDGES * 2
    H = np.zeros((n_free, n_free))
    rhs = np.zeros(n_free)
    for k in range(N_EDGES):
        add_between(H, rhs, k, k + 1, meas[k], infos[k])
    if loop_z is not None:
        add_between(H, rhs, 0, N_POSES - 1, loop_z, np.eye(2) * LAMBDA_LOOP)
    H.flat[:: n_free + 1] += 1e-8
    return np.linalg.solve(H, rhs), H


def truth_positions():
    idx = np.arange(N_POSES)
    return np.stack([np.zeros(N_POSES), idx * DX], axis=1)


def full_cap_count() -> int:
    return int(len(cap_points(N_GRAY + 1)))


def edge_caps(n_full: int) -> np.ndarray:
    caps = np.full(N_EDGES, n_full, dtype=int)
    caps[SEG_LO:SEG_HI] = N_GRAY
    return caps


def run_chain(rng, caps: np.ndarray, lambda_nominal: float):
    methods = ("fixed", "hard", "sigmoid", "irr", "front")
    true = np.array([0.0, DX])
    info_imu = np.eye(2) * LAMBDA_IMU
    z_imu = np.zeros((N_EDGES, 2))
    z_icp = np.zeros((N_EDGES, 2))
    hess = np.zeros((N_EDGES, 2, 2))
    for k, n_cap in enumerate(caps):
        z_imu[k] = true + rng.normal(0.0, 1.0 / np.sqrt(LAMBDA_IMU), size=2)
        z_icp[k], hess[k], _ = register_step(int(n_cap), z_imu[k], rng)
    truth = truth_positions()
    loop_z = truth[-1] + rng.normal(0.0, 1.0 / np.sqrt(LAMBDA_LOOP), size=2)
    out = {}
    for method in methods:
        meas = np.zeros((N_EDGES, 2))
        infos = np.zeros((N_EDGES, 2, 2))
        for k in range(N_EDGES):
            if method == "front":
                meas[k], _ = fuse(z_imu[k], info_imu, z_icp[k], hess[k])
                infos[k] = np.eye(2) * lambda_nominal
            else:
                lid = factor_information(method, hess[k], lambda_nominal)
                meas[k], infos[k] = fuse(z_imu[k], info_imu, z_icp[k], lid)
        x, Hmat = solve_chain(meas, infos, loop_z)
        est = x.reshape(-1, 2)
        err = (est - truth[1:]).ravel()
        ate_y = float(np.sqrt(np.mean((est[:, 1] - truth[1:, 1]) ** 2)))
        nees = float(err @ Hmat @ err / err.size)
        out[method] = (ate_y, nees, est[:, 1] - truth[1:, 1])
    return out


def monte_carlo(caps, lambda_nominal, seed: int):
    methods = ("fixed", "hard", "sigmoid", "irr", "front")
    ate = {m: np.zeros(N_TRIALS) for m in methods}
    nees = {m: np.zeros(N_TRIALS) for m in methods}
    path = {m: np.zeros((N_TRIALS, N_EDGES)) for m in methods}
    rng = np.random.default_rng(seed)
    for t in range(N_TRIALS):
        trial = run_chain(rng, caps, lambda_nominal)
        for m in methods:
            ate[m][t] = trial[m][0]
            nees[m][t] = trial[m][1]
            path[m][t] = trial[m][2]
    return ate, nees, path


def diagnose(rng, n_full: int):
    true = np.array([0.0, DX])
    kinds = (0, N_GRAY, n_full)
    err = {k: [] for k in kinds}
    gam = {k: [] for k in kinds}
    lam = {k: [] for k in kinds}
    imu = []
    align = []
    for n_cap in kinds:
        for _ in range(N_DIAG):
            z_imu = true + rng.normal(0.0, 1.0 / np.sqrt(LAMBDA_IMU), size=2)
            z_icp, H, _ = register_step(n_cap, z_imu, rng)
            g, weak = gamma_of(H)
            ev, evec = np.linalg.eigh(0.5 * (H + H.T))
            err[n_cap].append(z_icp[1] - DX)
            gam[n_cap].append(g)
            lam[n_cap].append(weak)
            if n_cap == 0:
                imu.append(z_imu[1] - DX)
                align.append(abs(float(np.dot(evec[:, 0], np.array([0.0, 1.0])))))
    return (
        {k: np.asarray(v) for k, v in err.items()},
        {k: np.asarray(v) for k, v in gam.items()},
        {k: np.asarray(v) for k, v in lam.items()},
        np.asarray(imu),
        np.asarray(align),
    )


def plot_axes(errors, imu_err, n_full: int) -> None:
    labels = ["IMU guess", "ICP, no cap", "ICP, partial cap", "ICP, full cap"]
    data = [imu_err, errors[0], errors[N_GRAY], errors[n_full]]
    means = [float(np.mean(np.abs(a))) for a in data]
    fig, ax = plt.subplots(figsize=(6.8, 3.6))
    ax.bar(labels, means, color=["0.45", "C3", "C1", "C2"])
    ax.set_ylabel(r"mean $|\mathrm{error}|$ along the tunnel (m)")
    ax.set_title("Estimated normals: tunnel error vs end cap")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "icp_axis_error.png", dpi=140)
    plt.close(fig)


def plot_segment(path) -> None:
    methods = [
        ("fixed", "fixed covariance", "C0"),
        ("hard", "hard switch", "C3"),
        ("sigmoid", r"sigmoid $\times$ nominal", "C1"),
        ("irr", "estimated Hessian in the factor", "C2"),
        ("front", "front-end soft, tight factor", "C4"),
    ]
    fig, ax = plt.subplots(figsize=(7.4, 3.7))
    steps = np.arange(1, N_POSES)
    ax.axvspan(SEG_LO + 0.5, SEG_HI + 0.5, color="0.85", zorder=0, label="partial end cap")
    for key, label, color in methods:
        ax.plot(steps, np.mean(np.abs(path[key]), axis=0), color=color, lw=1.7, label=label, zorder=2)
    ax.set_xlabel("pose index")
    ax.set_ylabel(r"mean $|\mathrm{error}|$ along the tunnel (m)")
    ax.set_title(rf"PCA normals, nearest neighbors, yaw {YAW_DEG:.0f}°")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(OUT / "segment_path.png", dpi=140)
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    failed = []

    def check(cond: bool, msg: str) -> None:
        print(("PASS  " if cond else "FAIL  ") + msg)
        if not cond:
            failed.append(msg)

    n_full = full_cap_count()
    rng = np.random.default_rng(4)
    # Nominal information: strong eigenvalue of one full-cap registration.
    z0 = np.array([0.0, DX])
    _, H_full, _ = register_step(n_full, z0, rng)
    lambda_nominal = float(np.linalg.eigvalsh(H_full)[-1])

    errors, gammas, lams, imu_err, align = diagnose(rng, n_full)
    plot_axes(errors, imu_err, n_full)

    icp_empty = errors[0]
    stuck = float(np.mean(np.abs(icp_empty - imu_err)))
    imu_mae = float(np.mean(np.abs(imu_err)))
    empty_mae = float(np.mean(np.abs(icp_empty)))
    gray_mae = float(np.mean(np.abs(errors[N_GRAY])))
    full_mae = float(np.mean(np.abs(errors[n_full])))
    g_empty = float(np.mean(gammas[0]))
    g_gray = float(np.mean(gammas[N_GRAY]))
    g_full = float(np.mean(gammas[n_full]))

    check(stuck < 0.35 * max(imu_mae, 1e-9),
          f"no-cap ICP stays near the IMU guess (|Δ|={stuck:.5f}, IMU MAE={imu_mae:.5f})")
    check(full_mae < 0.5 * empty_mae,
          f"full cap shrinks tunnel error ({full_mae:.5f} vs {empty_mae:.5f})")
    check(g_full > g_gray + 0.15 and g_gray > g_empty + 0.1,
          f"γ separates: full {g_full:.3f} > gray {g_gray:.3f} > empty {g_empty:.3f}")
    check(g_gray > GAMMA_MID, f"partial cap is above the hard cut (γ={g_gray:.3f})")
    check(g_empty < GAMMA_MID, f"no-cap γ stays below the hard cut (γ={g_empty:.3f})")
    check(float(np.mean(align)) > 0.9, f"no-cap weak axis aligns with the tunnel (|dot|={np.mean(align):.3f})")

    caps = edge_caps(n_full)
    ate, nees, path = monte_carlo(caps, lambda_nominal, seed=5)
    plot_segment(path)
    ate_h, _, _ = monte_carlo(np.full(N_EDGES, n_full, dtype=int), lambda_nominal, seed=9)

    methods = ("fixed", "hard", "sigmoid", "irr", "front")
    after = {m: float(np.mean(np.abs(path[m][:, SEG_HI - 1 :]))) for m in methods}
    lines = [
        f"N_TRIALS={N_TRIALS} yaw={YAW_DEG} σ={POINT_SIGMA} λ_imu={LAMBDA_IMU:.1f} λ_nominal={lambda_nominal:.1f}",
        f"N_GRAY={N_GRAY} N_FULL={n_full}",
        f"λ weak  empty={np.mean(lams[0]):.1f} gray={np.mean(lams[N_GRAY]):.1f} full={np.mean(lams[n_full]):.1f}",
        f"γ  empty={g_empty:.3f} gray={g_gray:.3f} full={g_full:.3f}",
        f"MAE tunnel IMU={imu_mae:.5f} empty={empty_mae:.5f} gray={gray_mae:.5f} full={full_mae:.5f}",
        f"signed bias empty ICP={np.mean(icp_empty):.5f} IMU={np.mean(imu_err):.5f}",
        "ATE_y   " + "  ".join(f"{ate[m].mean():10.5f}" for m in methods),
        "NEES    " + "  ".join(f"{nees[m].mean():10.2f}" for m in methods),
        "after   " + "  ".join(f"{after[m]:10.5f}" for m in methods),
        "healthy " + "  ".join(f"{ate_h[m].mean():10.5f}" for m in methods),
    ]
    text = "\n".join(lines) + "\n"
    (OUT / "summary.txt").write_text(text)
    print(text)

    check(0.4 <= nees["irr"].mean() <= 2.5, f"IRR NEES near 1 ({nees['irr'].mean():.2f})")
    check(after["irr"] < after["front"] < after["hard"],
          f"after window IRR {after['irr']:.5f} < front {after['front']:.5f} < hard {after['hard']:.5f}")
    check(after["irr"] < 0.75 * after["hard"],
          f"after window IRR {after['irr']:.5f} < 0.75 × hard {after['hard']:.5f}")
    healthy_vals = np.array([ate_h[m].mean() for m in ("hard", "front", "irr")])
    check(healthy_vals.max() < 2.0 * max(healthy_vals.min(), 1e-12),
          f"full-cap ATE stays comparable ({healthy_vals})")

    print(f"saved {OUT / 'icp_axis_error.png'}")
    print(f"saved {OUT / 'segment_path.png'}")
    if failed:
        print(f"{len(failed)} check(s) failed")
        return 1
    print("all checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
