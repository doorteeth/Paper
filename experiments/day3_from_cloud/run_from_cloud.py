#!/usr/bin/env python3
"""Day 3: estimate γ from the point cloud, then use it in the factor graph.

Day 2 injected the true eigenvalue. Here the only geometric input is the
Day-1 corridor (walls, ground, a variable end cap) at a sensor yaw of 35°,
so the weak axis is not a world-coordinate axis.

    H = sum a a^T,  a = [p × n; n]
    γ_i = λ_i / (λ_i + λ_imu)

on the horizontal translation block, rotated into the world frame.
The lidar factor information along each eigenvector is:

    fixed      λ_nominal
    hard       λ_nominal if γ >= 0.3 else 0
    sigmoid    sigmoid((γ-0.3)/0.05) * λ_nominal
    irr        λ_i                         (the cloud eigenvalue)
    front      fuse with λ_i, publish λ_nominal * I

Run:
    python3 run_from_cloud.py
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = Path(__file__).resolve().parent / "out"
DAY1_PATH = (
    Path(__file__).resolve().parents[1] / "day1_hessian_spectrum" / "compute_spectrum.py"
)

GAMMA_MID = 0.3
SIGMOID_TAU = 0.05
# Matched to Day 2. A partial end cap of a few points then sits just above the
# hard threshold while its eigenvalue is far below the healthy-axis nominal.
LAMBDA_IMU = 2.0
LAMBDA_LOOP = 1.0e4
YAW_DEG = 35.0
DX = 1.0
N_POSES = 21
N_EDGES = N_POSES - 1
N_TRIALS = 50
# Edges [SEG_LO, SEG_HI) keep a partial end cap. The rest see the full cap.
SEG_LO = 8
SEG_HI = 15
# Two cap points: λ ≈ 2, so γ = λ/(λ+λ_imu) ≈ 0.5, and the hard switch accepts.
GRAY_CAP = 2
FULL_CAP = 240


def load_day1():
    spec = importlib.util.spec_from_file_location("day1_spectrum", DAY1_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def sigmoid_weight(gamma: float) -> float:
    return float(1.0 / (1.0 + np.exp(-(gamma - GAMMA_MID) / SIGMOID_TAU)))


def yaw_matrix(yaw_deg: float) -> np.ndarray:
    """World ← lidar, matching Day 1 scene_corridor."""
    yaw = np.deg2rad(yaw_deg)
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([[c, -s], [s, c]])


def cloud_xy_information(day1, n_cap: int, yaw_deg: float, seed: int) -> np.ndarray:
    """Horizontal translation information in the world frame, from the cloud.

    Translation information is sum n n^T. Point-position jitter does not
    enter it; the normals do. The end cap is built in the world frame and
    rotated into the lidar frame with the same convention as the corridor.
    """
    day1.RNG = np.random.default_rng(seed)
    P, N = day1.scene_corridor(yaw_deg=yaw_deg)
    if n_cap > 0:
        P_cap, N_cap = day1.sample_plane(
            np.array([0.0, 10.0, 1.5]),
            np.array([0.0, -1.0, 0.0]),
            np.array([4.0, 0.0, 0.0]),
            np.array([0.0, 0.0, 3.0]),
            20,
            12,
        )
        n_cap = min(n_cap, len(P_cap))
        R3 = np.eye(3)
        R3[:2, :2] = yaw_matrix(yaw_deg)
        # sample_plane is in the world frame. Express the cap in the lidar frame.
        P_cap = (R3.T @ P_cap[:n_cap].T).T
        N_cap = (R3.T @ N_cap[:n_cap].T).T
        P = np.vstack([P, P_cap])
        N = np.vstack([N, N_cap])
    H = day1.accumulate_H(P, N)
    H_lidar = H[3:5, 3:5]
    R = yaw_matrix(yaw_deg)
    return R @ H_lidar @ R.T


def factor_information(method: str, H_world: np.ndarray, lambda_nominal: float) -> np.ndarray:
    """2x2 lidar-factor information in the world frame."""
    evals, evecs = np.linalg.eigh(H_world)
    gammas = evals / (evals + LAMBDA_IMU)
    if method == "fixed":
        return np.eye(2) * lambda_nominal
    if method == "irr":
        return H_world.copy()
    if method == "hard":
        scale = np.array([lambda_nominal if g >= GAMMA_MID else 0.0 for g in gammas])
    elif method == "sigmoid":
        scale = np.array([sigmoid_weight(g) * lambda_nominal for g in gammas])
    else:
        raise KeyError(method)
    return evecs @ np.diag(scale) @ evecs.T


def sample_lidar(true_delta, z_imu, H_world, rng) -> np.ndarray:
    """Lidar measurement of the step.

    Observed axes: truth plus noise of variance 1/λ.
    Unobserved axes: the ICP initial guess, which is the IMU step.
    A method that paints nominal information on an unobserved axis is
    trusting the IMU guess at lidar strength.
    """
    evals, evecs = np.linalg.eigh(H_world)
    z = np.zeros(2)
    for i, lam in enumerate(evals):
        v = evecs[:, i]
        if lam < 1e-8:
            z += np.dot(z_imu, v) * v
        else:
            z += (np.dot(true_delta, v) + rng.normal(0.0, 1.0 / np.sqrt(lam))) * v
    return z


def fuse(z_a, La, z_b, Lb):
    L = La + Lb
    z = np.linalg.solve(L, La @ z_a + Lb @ z_b)
    return z, L


def add_between(H, rhs, i, j, z, L):
    """p_j - p_i = z. Pose 0 is fixed at the origin. L is 2x2."""
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
    x = np.linalg.solve(H, rhs)
    return x, H


def truth_positions():
    idx = np.arange(N_POSES)
    # Tunnel and motion are world +Y. Sensor yaw does not change the path.
    return np.stack([np.zeros(N_POSES), idx * DX], axis=1)


def edge_caps() -> np.ndarray:
    caps = np.full(N_EDGES, FULL_CAP, dtype=int)
    caps[SEG_LO:SEG_HI] = GRAY_CAP
    return caps


def precompute_geometry(day1):
    """One cloud per distinct cap count. Translation H does not use point jitter."""
    caps = edge_caps()
    library = {}
    for n_cap in np.unique(caps):
        library[int(n_cap)] = cloud_xy_information(day1, int(n_cap), YAW_DEG, seed=0)
    H_edges = [library[int(n)] for n in caps]
    # Nominal information: the strong eigenvalue when the end cap is present.
    evals = np.linalg.eigvalsh(library[FULL_CAP])
    lambda_nominal = float(evals[-1])
    return H_edges, lambda_nominal, library


def run_trial(rng, H_edges, lambda_nominal, with_loop: bool):
    methods = ("fixed", "hard", "sigmoid", "irr", "front")
    true = np.array([0.0, DX])
    z_imu = np.zeros((N_EDGES, 2))
    z_lid = np.zeros((N_EDGES, 2))
    info_imu = np.eye(2) * LAMBDA_IMU
    for k in range(N_EDGES):
        z_imu[k] = true + rng.normal(0.0, 1.0 / np.sqrt(LAMBDA_IMU), size=2)
        z_lid[k] = sample_lidar(true, z_imu[k], H_edges[k], rng)

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
                z_f, _ = fuse(z_imu[k], info_imu, z_lid[k], H_edges[k])
                meas[k] = z_f
                infos[k] = np.eye(2) * lambda_nominal
            else:
                lid = factor_information(method, H_edges[k], lambda_nominal)
                meas[k], infos[k] = fuse(z_imu[k], info_imu, z_lid[k], lid)
        x, H = solve_chain(meas, infos, loop_z)
        est = x.reshape(-1, 2)
        err = (est - truth[1:]).ravel()
        ate_y = float(np.sqrt(np.mean((est[:, 1] - truth[1:, 1]) ** 2)))
        nees = float(err @ H @ err / err.size)
        out[method] = (ate_y, nees, est[:, 1] - truth[1:, 1])
    return out


def monte_carlo(H_edges, lambda_nominal, with_loop: bool, seed: int):
    methods = ("fixed", "hard", "sigmoid", "irr", "front")
    ate = {m: np.zeros(N_TRIALS) for m in methods}
    nees = {m: np.zeros(N_TRIALS) for m in methods}
    path = {m: np.zeros((N_TRIALS, N_EDGES)) for m in methods}
    rng = np.random.default_rng(seed)
    for t in range(N_TRIALS):
        trial = run_trial(rng, H_edges, lambda_nominal, with_loop)
        for m in methods:
            ate[m][t] = trial[m][0]
            nees[m][t] = trial[m][1]
            path[m][t] = trial[m][2]
    return ate, nees, path


def plot_gamma_curve(day1, lambda_nominal: float) -> None:
    keeps = np.arange(0, FULL_CAP + 1, 8)
    gammas = []
    weaks = []
    for n in keeps:
        H = cloud_xy_information(day1, int(n), YAW_DEG, seed=1)
        ev = np.linalg.eigvalsh(H)
        weaks.append(ev[0])
        gammas.append(ev[0] / (ev[0] + LAMBDA_IMU))
    gammas = np.asarray(gammas)
    weaks = np.asarray(weaks)
    gone = 1.0 - keeps / FULL_CAP

    hard_info = np.where(gammas >= GAMMA_MID, lambda_nominal, 0.0)
    sig_info = np.array([sigmoid_weight(g) * lambda_nominal for g in gammas])

    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.6))
    ax = axes[0]
    ax.plot(gone, gammas, "C0-o", ms=3.5, label=r"weak axis $\gamma$ from cloud")
    ax.axhline(GAMMA_MID, color="0.4", ls="--", lw=1, label=rf"hard threshold {GAMMA_MID}")
    ax.set_xlabel("end wall removed (0 = full cap, 1 = pure corridor)")
    ax.set_ylabel(r"$\gamma$")
    ax.set_ylim(-0.02, 1.02)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    ax.set_title(rf"yaw {YAW_DEG:.0f}°, $\lambda_{{imu}}={LAMBDA_IMU:.0f}$")

    ax = axes[1]
    ax.plot(gone, weaks, "k-", lw=2, label=r"cloud $\lambda$")
    ax.plot(gone, weaks, "C2--", lw=1.4, label="IRR factor info")
    ax.plot(gone, hard_info, "C3-", lw=1.2, label="hard switch")
    ax.plot(gone, sig_info, "C1-", lw=1.2, label=r"sigmoid $\times$ nominal")
    ax.set_yscale("symlog", linthresh=1.0)
    ax.set_xlabel("end wall removed")
    ax.set_ylabel("information on the tunnel axis")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    ax.set_title(rf"nominal $\lambda={lambda_nominal:.0f}$")
    fig.tight_layout()
    fig.savefig(OUT / "cloud_gamma.png", dpi=140)
    plt.close(fig)


def plot_segment(path) -> None:
    methods = [
        ("fixed", "fixed covariance", "C0"),
        ("hard", "hard switch", "C3"),
        ("sigmoid", r"sigmoid $\times$ nominal", "C1"),
        ("irr", "IRR from cloud", "C2"),
        ("front", "front-end soft, tight factor", "C4"),
    ]
    fig, ax = plt.subplots(figsize=(7.4, 3.7))
    steps = np.arange(1, N_POSES)
    ax.axvspan(SEG_LO + 0.5, SEG_HI + 0.5, color="0.85", zorder=0, label="partial end cap")
    for key, label, color in methods:
        ax.plot(steps, np.mean(np.abs(path[key]), axis=0), color=color, lw=1.7, label=label, zorder=2)
    ax.set_xlabel("pose index")
    ax.set_ylabel(r"mean $|\mathrm{error}|$ along the tunnel")
    ax.set_title(rf"γ from the cloud, yaw {YAW_DEG:.0f}°, loop pins the end pose")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(OUT / "segment_path.png", dpi=140)
    plt.close(fig)


def gamma_of(H) -> float:
    lam = float(np.linalg.eigvalsh(H)[0])
    return lam / (lam + LAMBDA_IMU)


def weak_world_axis(H) -> np.ndarray:
    evals, evecs = np.linalg.eigh(H)
    return evecs[:, 0]


def report(ate, nees, path, library, lambda_nominal) -> str:
    methods = ("fixed", "hard", "sigmoid", "irr", "front")
    after = slice(SEG_HI - 1, None)
    lines = []
    lines.append(
        f"N_POSES={N_POSES} N_TRIALS={N_TRIALS} yaw={YAW_DEG} "
        f"λ_imu={LAMBDA_IMU} λ_nominal={lambda_nominal:.1f} λ_loop={LAMBDA_LOOP}"
    )
    lines.append(
        "γ full={:.3f}  gray={:.3f}  empty={:.3f}".format(
            gamma_of(library[FULL_CAP]),
            gamma_of(library[GRAY_CAP]),
            gamma_of(library[0]) if 0 in library else gamma_of(
                cloud_xy_information(load_day1(), 0, YAW_DEG, 0)
            ),
        )
    )
    # library may not contain 0; filled by caller.
    lines.append("ATE_y   " + "  ".join(f"{ate[m].mean():10.3f}" for m in methods))
    lines.append("NEES    " + "  ".join(f"{nees[m].mean():10.1f}" for m in methods))
    lines.append(
        "after   " + "  ".join(f"{np.mean(np.abs(path[m][:, after])):10.3f}" for m in methods)
    )
    text = "\n".join(lines) + "\n"
    (OUT / "summary.txt").write_text(text)
    return text


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    day1 = load_day1()
    failed = []

    def check(cond: bool, msg: str) -> None:
        print(("PASS  " if cond else "FAIL  ") + msg)
        if not cond:
            failed.append(msg)

    # Translation information ignores point jitter: two seeds, same normals.
    H_a = cloud_xy_information(day1, FULL_CAP, YAW_DEG, seed=1)
    H_b = cloud_xy_information(day1, FULL_CAP, YAW_DEG, seed=2)
    check(np.allclose(H_a, H_b, atol=1e-8), "translation H is invariant to point-position noise")

    H_edges, lambda_nominal, library = precompute_geometry(day1)
    library[0] = cloud_xy_information(day1, 0, YAW_DEG, seed=0)

    axis = weak_world_axis(library[0])
    align = abs(float(np.dot(axis, np.array([0.0, 1.0]))))
    check(align > 0.99, f"pure-corridor weak axis aligns with the tunnel (dot={align:.4f})")

    g_full = gamma_of(library[FULL_CAP])
    g_gray = gamma_of(library[GRAY_CAP])
    g_empty = gamma_of(library[0])
    check(g_full > 0.80, f"full end cap is localizable (γ={g_full:.3f})")
    check(0.45 <= g_gray <= 0.60, f"partial cap is above the hard cut (γ={g_gray:.3f})")
    check(g_empty < 0.02, f"pure corridor is degenerate (γ={g_empty:.3f})")

    # IRR factor on a healthy cloud is the cloud information, not a rescaled copy.
    irr = factor_information("irr", library[FULL_CAP], lambda_nominal)
    check(np.allclose(irr, library[FULL_CAP], atol=1e-8), "IRR factor information equals the cloud H")

    plot_gamma_curve(day1, lambda_nominal)

    ate, nees, path = monte_carlo(H_edges, lambda_nominal, with_loop=True, seed=5)
    plot_segment(path)
    text = report(ate, nees, path, library, lambda_nominal)
    print(text)

    methods_after = {
        m: float(np.mean(np.abs(path[m][:, SEG_HI - 1 :])))
        for m in ("fixed", "hard", "sigmoid", "irr", "front")
    }
    check(0.75 <= nees["irr"].mean() <= 1.35, f"IRR NEES near 1 ({nees['irr'].mean():.3f})")
    check(
        methods_after["irr"] < methods_after["front"],
        f"after window, IRR {methods_after['irr']:.3f} < front {methods_after['front']:.3f}",
    )
    check(
        methods_after["irr"] < 0.6 * methods_after["hard"],
        f"after window, IRR {methods_after['irr']:.3f} < 0.6 * hard {methods_after['hard']:.3f}",
    )
    check(
        ate["irr"].mean() < ate["hard"].mean(),
        f"tunnel ATE IRR {ate['irr'].mean():.3f} < hard {ate['hard'].mean():.3f}",
    )

    print(f"saved {OUT / 'cloud_gamma.png'}")
    print(f"saved {OUT / 'segment_path.png'}")
    if failed:
        print(f"{len(failed)} check(s) failed")
        return 1
    print("all checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
