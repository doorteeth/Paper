#!/usr/bin/env python3
"""Day 4: Gauss–Newton point-to-plane ICP, then the same factor graph.

Day 3 drew the relative pose from the cloud information matrix. Here each
step starts at the IMU guess, associates points to corridor planes, and
iterates the normal equations. Unobservable directions are not updated, so
they stay on the initial guess. The factor uses the Hessian at convergence,
rotated from the lidar frame into the world frame.

Run:
    python3 run_icp.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = Path(__file__).resolve().parent / "out"

GAMMA_MID = 0.3
SIGMOID_TAU = 0.05
LAMBDA_IMU = 2.0
LAMBDA_LOOP = 1.0e4
YAW_DEG = 35.0
DX = 1.0
POINT_SIGMA = 1.0  # point-to-plane residual scale; each point then contributes ~1
N_POSES = 21
N_EDGES = N_POSES - 1
N_TRIALS = 50
N_DIAG = 200
SEG_LO = 8
SEG_HI = 15
GRAY_CAP = 2
FULL_CAP = 240
N_WALL = 480  # per side; lateral eigenvalue ≈ 2 * N_WALL
N_GROUND = 160
EIG_CUT = 1e-6
MAX_ITERS = 8


def sigmoid_weight(gamma: float) -> float:
    return float(1.0 / (1.0 + np.exp(-(gamma - GAMMA_MID) / SIGMOID_TAU)))


def world_from_lidar(yaw_deg: float) -> np.ndarray:
    """3×3 world ← lidar, same yaw convention as Day 1."""
    yaw = np.deg2rad(yaw_deg)
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def corridor_planes(n_cap: int):
    """Axis-aligned planes n·x = d, plus how many scan points each contributes.

    Planes are tens of metres apart. Point noise has standard deviation 1, so
    nearest-plane association stays on the generating plane, while each point
    still contributes about 1 to the information matrix.
    """
    planes = [
        (np.array([1.0, 0.0, 0.0]), -30.0, N_WALL),   # x = -30
        (np.array([-1.0, 0.0, 0.0]), -30.0, N_WALL),  # x = +30
        (np.array([0.0, 0.0, 1.0]), 0.0, N_GROUND),   # z = 0
    ]
    if n_cap > 0:
        planes.append((np.array([0.0, -1.0, 0.0]), -80.0, n_cap))  # y = 80
    return planes


def sample_on_plane(normal, offset, count, rng) -> np.ndarray:
    """Patch on n·x = offset, kept far from every other plane."""
    if abs(normal[0]) > 0.5:  # side wall
        y = rng.uniform(0.0, 5.0, count)
        z = rng.uniform(12.0, 18.0, count)
        x = np.full(count, offset if normal[0] > 0 else -offset)
        # n=(1,0,0), offset=-30 → x=-30; n=(-1,0,0), offset=-30 → x=+30
        return np.stack([x, y, z], axis=1)
    if abs(normal[2]) > 0.5:  # ground
        x = rng.uniform(-8.0, 8.0, count)
        y = rng.uniform(0.0, 5.0, count)
        z = np.zeros(count)
        return np.stack([x, y, z], axis=1)
    # end cap, y = -offset
    x = rng.uniform(-4.0, 4.0, count)
    z = rng.uniform(12.0, 18.0, count)
    y = np.full(count, -offset)
    return np.stack([x, y, z], axis=1)


def make_scan(n_cap: int, t_true: np.ndarray, rng):
    """Lidar points at the true pose, with the plane that generated each point."""
    R = world_from_lidar(YAW_DEG)
    chunks_p = []
    chunks_n = []
    chunks_d = []
    labels = []
    for k, (normal, offset, count) in enumerate(corridor_planes(n_cap)):
        if count <= 0:
            continue
        pts = sample_on_plane(normal, offset, count, rng)
        pts = pts + rng.normal(0.0, POINT_SIGMA, size=pts.shape)
        chunks_p.append(pts)
        chunks_n.append(np.repeat(normal[None, :], count, axis=0))
        chunks_d.append(np.full(count, offset))
        labels.append(np.full(count, k, dtype=int))
    p_w = np.vstack(chunks_p)
    n_w = np.vstack(chunks_n)
    d = np.concatenate(chunks_d)
    label = np.concatenate(labels)
    p_l = (R.T @ (p_w - t_true).T).T
    return p_l, n_w, d, label


def gn_point_to_plane(p_l, plane_n, plane_d, t_init: np.ndarray):
    """Iterate nearest-plane association and a damped null-space-safe GN step.

    Translation is in the world frame. Rotation is held at the known yaw.
    Directions with eigenvalue <= EIG_CUT are left at the initial guess.
    Returns the world translation, the 2×2 horizontal information expressed
    in the world (lidar Hessian rotated by yaw), and the fraction of points
    whose nearest plane is the generating plane.
    """
    R = world_from_lidar(YAW_DEG)
    t = np.array(t_init, dtype=float).copy()
    assoc = np.zeros(len(p_l), dtype=int)
    H_w = np.zeros((3, 3))
    for _ in range(MAX_ITERS):
        x = (R @ p_l.T).T + t
        dist = x @ plane_n.T - plane_d
        assoc = np.argmin(np.abs(dist), axis=1)
        n = plane_n[assoc]
        r = np.einsum("ij,ij->i", n, x) - plane_d[assoc]
        H_w = n.T @ n
        g = n.T @ r
        evals, evecs = np.linalg.eigh(H_w)
        delta = np.zeros(3)
        for i, lam in enumerate(evals):
            if lam > EIG_CUT:
                delta -= float(np.dot(evecs[:, i], g)) / lam * evecs[:, i]
        t = t + delta
        if np.linalg.norm(delta) < 1e-10:
            break
    # Information used by the factor: same normals, expressed in the lidar
    # frame and rotated back, so the yaw is part of the pipeline.
    n_l = (R.T @ plane_n[assoc].T).T
    H_l = n_l.T @ n_l
    H_from_lidar = R @ H_l @ R.T
    return t, H_from_lidar[:2, :2], assoc


def plane_table(n_cap: int):
    planes = corridor_planes(n_cap)
    n = np.stack([p[0] for p in planes], axis=0)
    d = np.array([p[1] for p in planes])
    return n, d


def register_step(n_cap: int, z_imu: np.ndarray, rng):
    """One relative-pose ICP. Truth is a unit step along world +Y."""
    t_true = np.array([0.0, DX, 0.0])
    p_l, _, _, label = make_scan(n_cap, t_true, rng)
    plane_n, plane_d = plane_table(n_cap)
    t_init = np.array([z_imu[0], z_imu[1], 0.0])
    t_hat, H_xy, assoc = gn_point_to_plane(p_l, plane_n, plane_d, t_init)
    acc = float(np.mean(assoc == label))
    return t_hat[:2], H_xy, acc


def gamma_weak(H: np.ndarray) -> float:
    lam = float(np.linalg.eigvalsh(H)[0])
    return lam / (lam + LAMBDA_IMU)


def factor_information(method: str, H_world: np.ndarray, lambda_nominal: float) -> np.ndarray:
    evals, evecs = np.linalg.eigh(H_world)
    gammas = evals / (evals + LAMBDA_IMU)
    if method == "fixed":
        return np.eye(2) * lambda_nominal
    if method == "irr":
        return (evecs * np.maximum(evals, 0.0)) @ evecs.T
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
    H.flat[:: n_free + 1] += 1e-10
    return np.linalg.solve(H, rhs), H


def truth_positions():
    idx = np.arange(N_POSES)
    return np.stack([np.zeros(N_POSES), idx * DX], axis=1)


def edge_caps() -> np.ndarray:
    caps = np.full(N_EDGES, FULL_CAP, dtype=int)
    caps[SEG_LO:SEG_HI] = GRAY_CAP
    return caps


def run_chain(rng, caps: np.ndarray, lambda_nominal: float, with_loop: bool):
    methods = ("fixed", "hard", "sigmoid", "irr", "front")
    true = np.array([0.0, DX])
    info_imu = np.eye(2) * LAMBDA_IMU
    z_imu = np.zeros((N_EDGES, 2))
    z_icp = np.zeros((N_EDGES, 2))
    hess = np.zeros((N_EDGES, 2, 2))
    acc = np.zeros(N_EDGES)
    for k, n_cap in enumerate(caps):
        z_imu[k] = true + rng.normal(0.0, 1.0 / np.sqrt(LAMBDA_IMU), size=2)
        z_icp[k], hess[k], acc[k] = register_step(int(n_cap), z_imu[k], rng)

    truth = truth_positions()
    loop_z = None
    if with_loop:
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
        x, H = solve_chain(meas, infos, loop_z)
        est = x.reshape(-1, 2)
        err = (est - truth[1:]).ravel()
        ate_y = float(np.sqrt(np.mean((est[:, 1] - truth[1:, 1]) ** 2)))
        nees = float(err @ H @ err / err.size)
        out[method] = (ate_y, nees, est[:, 1] - truth[1:, 1])
    return out, float(np.mean(acc))


def monte_carlo(caps, lambda_nominal, seed: int):
    methods = ("fixed", "hard", "sigmoid", "irr", "front")
    ate = {m: np.zeros(N_TRIALS) for m in methods}
    nees = {m: np.zeros(N_TRIALS) for m in methods}
    path = {m: np.zeros((N_TRIALS, N_EDGES)) for m in methods}
    acc = np.zeros(N_TRIALS)
    rng = np.random.default_rng(seed)
    for t in range(N_TRIALS):
        trial, acc[t] = run_chain(rng, caps, lambda_nominal, with_loop=True)
        for m in methods:
            ate[m][t] = trial[m][0]
            nees[m][t] = trial[m][1]
            path[m][t] = trial[m][2]
    return ate, nees, path, acc


def diagnose_axes(rng):
    """Single-step ICP error on the tunnel axis, against the IMU guess."""
    true = np.array([0.0, DX])
    rows = {0: [], GRAY_CAP: [], FULL_CAP: []}
    gammas = {0: [], GRAY_CAP: [], FULL_CAP: []}
    imu_err = []
    align = []
    for n_cap in rows:
        for _ in range(N_DIAG):
            z_imu = true + rng.normal(0.0, 1.0 / np.sqrt(LAMBDA_IMU), size=2)
            z_icp, H, _ = register_step(n_cap, z_imu, rng)
            ev, evec = np.linalg.eigh(H)
            # Tunnel is world +Y. Error of the ICP output relative to truth.
            rows[n_cap].append(z_icp[1] - true[1])
            gammas[n_cap].append(ev[0] / (ev[0] + LAMBDA_IMU))
            if n_cap == 0:
                imu_err.append(z_imu[1] - true[1])
                align.append(abs(float(np.dot(evec[:, 0], np.array([0.0, 1.0])))))
    return (
        {k: np.asarray(v) for k, v in rows.items()},
        {k: np.asarray(v) for k, v in gammas.items()},
        np.asarray(imu_err),
        np.asarray(align),
    )


def nominal_from_full_cap(rng) -> float:
    z = np.array([0.0, DX])
    _, H, _ = register_step(FULL_CAP, z, rng)
    return float(np.linalg.eigvalsh(H)[-1])


def plot_axes(errors, imu_err) -> None:
    labels = ["IMU guess", "ICP, no cap", f"ICP, {GRAY_CAP} cap pts", "ICP, full cap"]
    data = [imu_err, errors[0], errors[GRAY_CAP], errors[FULL_CAP]]
    means = [float(np.mean(np.abs(a))) for a in data]
    fig, ax = plt.subplots(figsize=(6.6, 3.6))
    colors = ["0.45", "C3", "C1", "C2"]
    ax.bar(labels, means, color=colors)
    ax.set_ylabel(r"mean $|\mathrm{error}|$ along the tunnel")
    ax.set_title("Point-to-plane ICP: weak-axis error follows the end cap")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "icp_axis_error.png", dpi=140)
    plt.close(fig)


def plot_segment(path) -> None:
    methods = [
        ("fixed", "fixed covariance", "C0"),
        ("hard", "hard switch", "C3"),
        ("sigmoid", r"sigmoid $\times$ nominal", "C1"),
        ("irr", "ICP Hessian in the factor", "C2"),
        ("front", "front-end soft, tight factor", "C4"),
    ]
    fig, ax = plt.subplots(figsize=(7.4, 3.7))
    steps = np.arange(1, N_POSES)
    ax.axvspan(SEG_LO + 0.5, SEG_HI + 0.5, color="0.85", zorder=0, label="partial end cap")
    for key, label, color in methods:
        ax.plot(steps, np.mean(np.abs(path[key]), axis=0), color=color, lw=1.7, label=label, zorder=2)
    ax.set_xlabel("pose index")
    ax.set_ylabel(r"mean $|\mathrm{error}|$ along the tunnel")
    ax.set_title(rf"ICP measurements, yaw {YAW_DEG:.0f}°, loop pins the end pose")
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

    rng = np.random.default_rng(3)
    lambda_nominal = nominal_from_full_cap(rng)
    errors, gammas, imu_err, align = diagnose_axes(rng)
    plot_axes(errors, imu_err)

    # Pure corridor: ICP must stay on the IMU guess along the tunnel.
    icp_empty = errors[0]
    stuck = float(np.mean(np.abs(icp_empty - imu_err)))
    imu_mae = float(np.mean(np.abs(imu_err)))
    full_mae = float(np.mean(np.abs(errors[FULL_CAP])))
    gray_mae = float(np.mean(np.abs(errors[GRAY_CAP])))
    check(stuck < 0.05 * max(imu_mae, 1e-6), f"no-cap ICP matches IMU on the tunnel axis (mean |Δ|={stuck:.4f}, IMU MAE={imu_mae:.3f})")
    check(full_mae < 0.25 * float(np.mean(np.abs(icp_empty))), f"full cap shrinks tunnel error ({full_mae:.3f} vs no-cap {np.mean(np.abs(icp_empty)):.3f})")
    check(float(np.mean(align)) > 0.95, f"weak axis aligns with the tunnel (mean |dot|={np.mean(align):.3f})")
    check(float(np.mean(gammas[FULL_CAP])) > 0.9, f"full cap γ={np.mean(gammas[FULL_CAP]):.3f}")
    check(0.45 <= float(np.mean(gammas[GRAY_CAP])) <= 0.55, f"partial cap γ={np.mean(gammas[GRAY_CAP]):.3f}")
    check(float(np.mean(gammas[0])) < 0.05, f"no-cap γ={np.mean(gammas[0]):.3f}")
    # High γ has smaller absolute error than the unobservable axis.
    check(
        full_mae < gray_mae and full_mae < 0.25 * float(np.mean(np.abs(icp_empty))),
        f"tunnel |error| falls as the cap grows: none {np.mean(np.abs(icp_empty)):.3f}, gray {gray_mae:.3f}, full {full_mae:.3f}",
    )

    caps = edge_caps()
    ate, nees, path, acc = monte_carlo(caps, lambda_nominal, seed=5)
    plot_segment(path)
    healthy_caps = np.full(N_EDGES, FULL_CAP, dtype=int)
    ate_h, _, _, _ = monte_carlo(healthy_caps, lambda_nominal, seed=9)

    methods = ("fixed", "hard", "sigmoid", "irr", "front")
    after = {
        m: float(np.mean(np.abs(path[m][:, SEG_HI - 1 :])))
        for m in methods
    }
    lines = [
        f"N_TRIALS={N_TRIALS} yaw={YAW_DEG} λ_imu={LAMBDA_IMU} λ_nominal={lambda_nominal:.1f}",
        f"assoc mean={acc.mean():.3f}",
        f"γ full={np.mean(gammas[FULL_CAP]):.3f} gray={np.mean(gammas[GRAY_CAP]):.3f} empty={np.mean(gammas[0]):.3f}",
        f"MAE tunnel  IMU={imu_mae:.3f}  empty={np.mean(np.abs(icp_empty)):.3f}  gray={gray_mae:.3f}  full={full_mae:.3f}",
        f"signed bias empty ICP={np.mean(icp_empty):.3f} IMU={np.mean(imu_err):.3f}",
        "ATE_y   " + "  ".join(f"{ate[m].mean():10.3f}" for m in methods),
        "NEES    " + "  ".join(f"{nees[m].mean():10.1f}" for m in methods),
        "after   " + "  ".join(f"{after[m]:10.3f}" for m in methods),
        "healthy " + "  ".join(f"{ate_h[m].mean():10.3f}" for m in methods),
    ]
    text = "\n".join(lines) + "\n"
    (OUT / "summary.txt").write_text(text)
    print(text)

    check(acc.mean() > 0.98, f"nearest-plane association stays correct ({acc.mean():.3f})")
    check(0.7 <= nees["irr"].mean() <= 1.4, f"IRR NEES near 1 ({nees['irr'].mean():.3f})")
    check(after["irr"] < after["front"] < after["hard"],
          f"after window IRR {after['irr']:.3f} < front {after['front']:.3f} < hard {after['hard']:.3f}")
    check(after["irr"] < 0.6 * after["hard"],
          f"after window IRR {after['irr']:.3f} < 0.6 × hard {after['hard']:.3f}")
    healthy_vals = np.array([ate_h[m].mean() for m in ("hard", "front", "irr")])
    check(healthy_vals.max() < 2.0 * healthy_vals.min(),
          f"full-cap path ATE stays comparable (hard {ate_h['hard'].mean():.3f}, front {ate_h['front'].mean():.3f}, IRR {ate_h['irr'].mean():.3f})")

    print(f"saved {OUT / 'icp_axis_error.png'}")
    print(f"saved {OUT / 'segment_path.png'}")
    if failed:
        print(f"{len(failed)} check(s) failed")
        return 1
    print("all checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
