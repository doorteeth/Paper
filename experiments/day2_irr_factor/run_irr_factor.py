#!/usr/bin/env python3
"""Day 2: what a continuous localizability score is allowed to do.

Two checks, both linear-Gaussian so the optimum is known.

1. Corridor geometry from Day 1. Prior information is isotropic.
   Along a Hessian axis with eigenvalue λ,

       γ = λ / (λ_imu + λ) = 1 - v^T P+ v / v^T P- v

   That is the information reduction ratio. A hard threshold slices it.

2. Pose-graph chain along a tunnel, plus an optional loop at the end.
   Same noisy lidar measurement, five ways of turning γ into a factor.
   A second scene keeps most edges healthy and degrades only a window,
   which is where a loop can repair the tunnel axis only if that window's
   factor is actually soft.

       fixed     information = λ_nominal on every axis
       hard      λ_nominal if γ >= 0.3 else 0          (SKF-style)
       sigmoid   w(γ) * λ_nominal, w = sigmoid((γ-0.3)/0.05)
       irr       λ_imu * γ / (1-γ)                     (equals the true λ)
       front     fuse with the irr weight, then publish a tight
                 isotropic between-factor (weight dies in the front end)

Run:
    python3 run_irr_factor.py
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

# Hard switch and sigmoid use the operating point from the research note.
GAMMA_MID = 0.3
SIGMOID_TAU = 0.05

# Pose-graph noise model. Units are arbitrary; only ratios matter.
LAMBDA_IMU = 2.0
LAMBDA_NOMINAL = 400.0  # "healthy scan" information used by fixed / hard / sigmoid
LAMBDA_LOOP = 1.0e4
N_POSES = 21  # pose 0 is pinned at the origin
N_TRIALS = 60
DX = 1.0


def load_day1():
    spec = importlib.util.spec_from_file_location("day1_spectrum", DAY1_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def sigmoid_weight(gamma: float) -> float:
    return float(1.0 / (1.0 + np.exp(-(gamma - GAMMA_MID) / SIGMOID_TAU)))


def irr_information(gamma: float, lambda_imu: float = LAMBDA_IMU) -> float:
    """Invert γ = λ / (λ_imu + λ). This is the lidar information, not a score."""
    g = min(max(float(gamma), 0.0), 1.0 - 1e-12)
    return lambda_imu * g / (1.0 - g)


def gamma_from_lambda(lam: float, lambda_imu: float) -> float:
    return float(lam / (lambda_imu + lam))


# ---------------------------------------------------------------------------
# Check 1. γ on the Day-1 corridor, as the end wall disappears.
# ---------------------------------------------------------------------------

def corridor_gamma_curve(day1, lambda_imu: float = 40.0, n_steps: int = 21):
    """Return end-wall fraction gone, eigenvalues (weak→strong), and γ."""
    day1.RNG = np.random.default_rng(0)
    gone = []
    eigs = []
    alphas = np.linspace(1.0, 0.0, n_steps)
    for alpha in alphas:
        P_side, N_side = day1.scene_corridor(yaw_deg=0.0)
        P_cap, N_cap = day1.sample_plane(
            np.array([0.0, 10.0, 1.5]),
            np.array([0.0, -1.0, 0.0]),
            np.array([4.0, 0.0, 0.0]),
            np.array([0.0, 0.0, 3.0]),
            20,
            12,
        )
        n_keep = int(round(alpha * len(P_cap)))
        if n_keep > 0:
            P = np.vstack([P_side, P_cap[:n_keep]])
            N = np.vstack([N_side, N_cap[:n_keep]])
        else:
            P, N = P_side, N_side
        H = day1.accumulate_H(P, N)
        # Residual variance is 1, so H_tt is already an information matrix.
        evals = np.linalg.eigvalsh(H[3:, 3:])
        eigs.append(evals)
        gone.append(1.0 - alpha)
    eigs = np.asarray(eigs)
    gammas = eigs / (lambda_imu + eigs)
    return np.asarray(gone), eigs, gammas, lambda_imu


def plot_corridor(gone, eigs, gammas, lambda_imu: float) -> None:
    weak = eigs[:, 0]
    gamma_weak = gammas[:, 0]
    # Healthy-axis scale: the strongest translation eigenvalue with the wall present.
    lambda_nominal = float(eigs[0, -1])

    hard_info = np.where(gamma_weak >= GAMMA_MID, lambda_nominal, 0.0)
    sig_info = np.array([sigmoid_weight(g) * lambda_nominal for g in gamma_weak])
    irr_info = weak.copy()  # irr inversion recovers λ exactly when the prior is isotropic

    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.6))

    ax = axes[0]
    labels = [r"weak (along tunnel)", r"mid", r"strong"]
    for i, lab in enumerate(labels):
        ax.plot(gone, gammas[:, i], "-o", ms=3.5, label=lab)
    ax.axhline(GAMMA_MID, color="0.4", ls="--", lw=1, label=rf"hard threshold {GAMMA_MID}")
    ax.set_xlabel("end wall disappeared (0 = cap present, 1 = pure corridor)")
    ax.set_ylabel(r"information reduction ratio $\gamma$")
    ax.set_ylim(-0.02, 1.02)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, loc="center left")
    ax.set_title(rf"Prior $\lambda_{{imu}}={lambda_imu:.0f}$")

    ax = axes[1]
    ax.plot(gone, weak, "k-", lw=2, label=r"true $\lambda$ on weak axis")
    ax.plot(gone, irr_info, "C2--", lw=1.5, label=r"IRR factor info")
    ax.plot(gone, hard_info, "C3-", lw=1.2, label="hard switch")
    ax.plot(gone, sig_info, "C1-", lw=1.2, label=r"sigmoid $\times$ nominal")
    ax.set_yscale("symlog", linthresh=1.0)
    ax.set_xlabel("end wall disappeared")
    ax.set_ylabel("information put on the tunnel axis")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    ax.set_title(rf"nominal $\lambda={lambda_nominal:.0f}$ (strong axis)")

    fig.tight_layout()
    fig.savefig(OUT / "corridor_gamma.png", dpi=140)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Check 2. Same measurement, five factor models, optional loop.
# ---------------------------------------------------------------------------

def add_between(H, rhs, i, j, z, info):
    """p_j - p_i = z. Pose 0 is fixed at the origin. info is length-2."""
    L = np.diag(np.asarray(info, dtype=float))

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


def solve_chain(meas, infos, loop_z=None, loop_info=None):
    """meas[k] is the between-measurement from pose k to k+1, k = 0..N-2.

    infos[k] is the length-2 information of that single between factor.
    A loop, if given, observes pose N-1 in the world frame (pose 0 at 0).
    """
    n_free = (N_POSES - 1) * 2
    H = np.zeros((n_free, n_free))
    rhs = np.zeros(n_free)
    for k in range(N_POSES - 1):
        add_between(H, rhs, k, k + 1, meas[k], infos[k])
    if loop_z is not None:
        add_between(H, rhs, 0, N_POSES - 1, loop_z, loop_info)
    # Relative factors always touch pose 0, so H is SPD as long as each
    # edge has positive information on both axes. Jitter covers exact zeros.
    H.flat[:: n_free + 1] += 1e-10
    x = np.linalg.solve(H, rhs)
    return x, H


def truth_positions():
    idx = np.arange(N_POSES)
    return np.stack([idx * DX, np.zeros(N_POSES)], axis=1)


def sample_edges(rng, lambda_x):
    """IMU and lidar measurements of each true step [DX, 0].

    lambda_x is a scalar, or one value per edge. Lateral lidar information
    stays at the healthy nominal on every edge.
    """
    n_edges = N_POSES - 1
    lambda_x = np.broadcast_to(np.asarray(lambda_x, dtype=float), (n_edges,)).copy()
    true = np.array([DX, 0.0])
    info_imu = np.array([LAMBDA_IMU, LAMBDA_IMU])
    z_imu = true + rng.normal(0.0, 1.0 / np.sqrt(info_imu), size=(n_edges, 2))
    z_lid = np.zeros((n_edges, 2))
    info_lid = np.zeros((n_edges, 2))
    for k in range(n_edges):
        info_lid[k] = (lambda_x[k], LAMBDA_NOMINAL)
        z_lid[k] = true + rng.normal(0.0, 1.0 / np.sqrt(info_lid[k]))
    return z_imu, z_lid, info_lid


def factor_info(method: str, info_lid: np.ndarray) -> np.ndarray:
    """Information of the lidar factor. Does not include the IMU factor."""
    gammas = info_lid / (LAMBDA_IMU + info_lid)
    if method == "fixed":
        return np.array([LAMBDA_NOMINAL, LAMBDA_NOMINAL])
    if method == "hard":
        return np.array(
            [
                LAMBDA_NOMINAL if gammas[0] >= GAMMA_MID else 0.0,
                LAMBDA_NOMINAL if gammas[1] >= GAMMA_MID else 0.0,
            ]
        )
    if method == "sigmoid":
        w = np.array([sigmoid_weight(g) for g in gammas])
        return w * LAMBDA_NOMINAL
    if method == "irr":
        return np.array([irr_information(g) for g in gammas])
    raise KeyError(method)


def fuse_edge(z_imu, info_imu, z_lid, info_lid):
    info = info_imu + info_lid
    z = (info_imu * z_imu + info_lid * z_lid) / info
    return z, info


def estimate(method: str, z_imu, z_lid, info_lid, loop_z):
    """Return free-pose vector (poses 1..N-1) and its information matrix.

    IMU and lidar are pre-fused into one between-factor. That is the same
    posterior as keeping them as two quadratic terms on the same edge.
    info_lid has shape (n_edges, 2) and is the true lidar information.
    """
    n_edges = N_POSES - 1
    info_imu = np.array([LAMBDA_IMU, LAMBDA_IMU])
    fused_z = np.zeros((n_edges, 2))
    fused_info = np.zeros((n_edges, 2))
    published = np.array([LAMBDA_NOMINAL, LAMBDA_NOMINAL])
    for k in range(n_edges):
        if method == "front":
            # Oracle front end blends with the true lidar information, then
            # throws that information away and publishes a tight isotropic factor.
            fused_z[k], _ = fuse_edge(z_imu[k], info_imu, z_lid[k], info_lid[k])
            fused_info[k] = published
        else:
            lid = factor_info(method, info_lid[k])
            fused_z[k], fused_info[k] = fuse_edge(z_imu[k], info_imu, z_lid[k], lid)
    loop_info = None if loop_z is None else np.array([LAMBDA_LOOP, LAMBDA_LOOP])
    return solve_chain(fused_z, fused_info, loop_z, loop_info)


def run_trial(rng, lambda_x, with_loop: bool):
    z_imu, z_lid, info_lid = sample_edges(rng, lambda_x)
    truth = truth_positions()
    loop_z = None
    if with_loop:
        loop_z = truth[-1] + rng.normal(0.0, 1.0 / np.sqrt(LAMBDA_LOOP), size=2)

    out = {}
    for method in ("fixed", "hard", "sigmoid", "irr", "front"):
        x, H = estimate(method, z_imu, z_lid, info_lid, loop_z)
        est = x.reshape(-1, 2)
        err = (est - truth[1:]).ravel()
        ate_x = float(np.sqrt(np.mean((est[:, 0] - truth[1:, 0]) ** 2)))
        nees = float(err @ H @ err / err.size)
        out[method] = (ate_x, nees, est[:, 0] - truth[1:, 0])
    return out


def sweep(with_loop: bool, lambdas: np.ndarray, seed: int):
    methods = ("fixed", "hard", "sigmoid", "irr", "front")
    ate = {m: np.zeros((len(lambdas), N_TRIALS)) for m in methods}
    nees = {m: np.zeros((len(lambdas), N_TRIALS)) for m in methods}
    rng = np.random.default_rng(seed)
    for i, lam in enumerate(lambdas):
        for t in range(N_TRIALS):
            trial = run_trial(rng, float(lam), with_loop)
            for m in methods:
                ate[m][i, t] = trial[m][0]
                nees[m][i, t] = trial[m][1]
    return ate, nees


# Edges whose tunnel-axis lidar information is partial. Healthy elsewhere.
# Pose k is after edge k-1, so this window moves poses 9..15.
SEG_LO = 8
SEG_HI = 15  # exclusive
SEG_GAMMA = 0.35


def segment_profile() -> np.ndarray:
    lam = np.full(N_POSES - 1, LAMBDA_NOMINAL)
    lam[SEG_LO:SEG_HI] = irr_information(SEG_GAMMA)
    return lam


def run_segment(seed: int = 11):
    """Healthy corridor, a partially degenerate window, then a loop."""
    methods = ("fixed", "hard", "sigmoid", "irr", "front")
    path = {m: np.zeros((N_TRIALS, N_POSES - 1)) for m in methods}
    ate = {m: np.zeros(N_TRIALS) for m in methods}
    nees = {m: np.zeros(N_TRIALS) for m in methods}
    rng = np.random.default_rng(seed)
    profile = segment_profile()
    for t in range(N_TRIALS):
        trial = run_trial(rng, profile, with_loop=True)
        for m in methods:
            ate[m][t] = trial[m][0]
            nees[m][t] = trial[m][1]
            path[m][t] = trial[m][2]
    return ate, nees, path


def plot_segment(path) -> None:
    methods = [
        ("fixed", "fixed covariance", "C0"),
        ("hard", "hard switch", "C3"),
        ("sigmoid", r"sigmoid $\times$ nominal", "C1"),
        ("irr", "IRR-consistent", "C2"),
        ("front", "front-end soft, tight factor", "C4"),
    ]
    fig, ax = plt.subplots(figsize=(7.4, 3.7))
    steps = np.arange(1, N_POSES)
    ax.axvspan(SEG_LO + 0.5, SEG_HI + 0.5, color="0.85", zorder=0, label="partial degeneracy")
    for key, label, color in methods:
        mu = np.mean(np.abs(path[key]), axis=0)
        ax.plot(steps, mu, color=color, lw=1.7, label=label, zorder=2)
    ax.set_xlabel("pose index")
    ax.set_ylabel(r"mean $|\mathrm{error}|$ on tunnel axis")
    ax.set_title(rf"local gray zone ($\gamma={SEG_GAMMA}$), loop pins the end pose")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(OUT / "segment_path.png", dpi=140)
    plt.close(fig)


def plot_sweep(lambdas, ate_loop, nees_loop, ate_open, nees_open) -> None:
    methods = [
        ("fixed", "fixed covariance", "C0"),
        ("hard", "hard switch", "C3"),
        ("sigmoid", r"sigmoid $\times$ nominal", "C1"),
        ("irr", "IRR-consistent", "C2"),
        ("front", "front-end soft, tight factor", "C4"),
    ]
    gamma_of = lambdas / (LAMBDA_IMU + lambdas)

    fig, axes = plt.subplots(2, 2, figsize=(9.4, 6.4), sharex=True)
    panels = (
        (axes[0, 0], ate_loop, r"ATE on tunnel axis, with loop", False),
        (axes[0, 1], nees_loop, r"NEES, with loop (1 = consistent)", True),
        (axes[1, 0], ate_open, r"ATE on tunnel axis, no loop", False),
        (axes[1, 1], nees_open, r"NEES, no loop", True),
    )
    for ax, series, title, is_nees in panels:
        for key, label, color in methods:
            mu = series[key].mean(axis=1)
            ax.plot(gamma_of, mu, color=color, lw=1.6, label=label)
        ax.axvline(GAMMA_MID, color="0.45", ls="--", lw=1)
        ax.set_title(title, fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_yscale("log")
        if is_nees:
            ax.axhline(1.0, color="k", lw=0.8, alpha=0.6)
    axes[1, 0].set_xlabel(r"true $\gamma$ on the tunnel axis")
    axes[1, 1].set_xlabel(r"true $\gamma$ on the tunnel axis")
    axes[0, 0].legend(fontsize=7, loc="upper right")
    fig.tight_layout()
    fig.savefig(OUT / "factor_sweep.png", dpi=140)
    plt.close(fig)


def self_check() -> None:
    """One edge, no noise: weights must move the mean as the formula says."""
    z_imu = np.array([0.0, 0.0])
    z_lid = np.array([1.0, 1.0])
    info_imu = np.array([LAMBDA_IMU, LAMBDA_IMU])
    # γ = 0.5 on x ⇒ λ = λ_imu. Above the hard threshold.
    info_true = np.array([LAMBDA_IMU, LAMBDA_NOMINAL])
    g = info_true / (info_imu + info_true)
    assert abs(g[0] - 0.5) < 1e-12

    z_irr, info_irr = fuse_edge(z_imu, info_imu, z_lid, np.array([irr_information(g[0]), irr_information(g[1])]))
    # x: equal IMU and lidar information, mean halfway.
    assert abs(z_irr[0] - 0.5) < 1e-9, z_irr
    # y: lidar dominates, mean near the lidar measurement.
    assert abs(z_irr[1] - 1.0) < 1e-2, z_irr

    z_hard, _ = fuse_edge(
        z_imu,
        info_imu,
        z_lid,
        np.array([LAMBDA_NOMINAL, LAMBDA_NOMINAL]),  # γ=0.5 passes the threshold
    )
    assert z_hard[0] > 0.98, z_hard

    z_drop, _ = fuse_edge(z_imu, info_imu, z_lid, np.array([0.0, LAMBDA_NOMINAL]))
    assert abs(z_drop[0] - 0.0) < 1e-12

    # Noise-free chain: every method must recover the truth.
    rng = np.random.default_rng(1)
    # Build zero-noise measurements by overriding sample scale via a direct call.
    true = np.array([DX, 0.0])
    z_imu_c = np.repeat(true[None, :], N_POSES - 1, axis=0)
    z_lid_c = z_imu_c.copy()
    info_lid = np.tile(
        np.array([irr_information(0.35), LAMBDA_NOMINAL]),
        (N_POSES - 1, 1),
    )
    truth = truth_positions()
    for method in ("fixed", "hard", "sigmoid", "irr", "front"):
        x, _ = estimate(method, z_imu_c, z_lid_c, info_lid, truth[-1])
        err = np.max(np.abs(x.reshape(-1, 2) - truth[1:]))
        assert err < 1e-6, (method, err)
    del rng


def summarize(lambdas, ate_loop, nees_loop, ate_open, seg_ate, seg_nees, seg_path) -> str:
    methods = ("fixed", "hard", "sigmoid", "irr", "front")
    gammas = lambdas / (LAMBDA_IMU + lambdas)
    targets = (0.05, 0.35, 0.995)
    lines = []
    lines.append(f"N_POSES={N_POSES}  N_TRIALS={N_TRIALS}  λ_imu={LAMBDA_IMU}  λ_nominal={LAMBDA_NOMINAL}  λ_loop={LAMBDA_LOOP}")
    lines.append("γ        " + "  ".join(f"{m:>10}" for m in methods))
    for g_star in targets:
        i = int(np.argmin(np.abs(gammas - g_star)))
        g = gammas[i]
        cells = [f"{ate_loop[m][i].mean():10.3f}" for m in methods]
        lines.append(f"{g:0.3f} ATE " + "  ".join(cells))
        cells = [f"{nees_loop[m][i].mean():10.1f}" for m in methods]
        lines.append(f"{g:0.3f} NEES" + "  ".join(cells))
        cells = [f"{ate_open[m][i].mean():10.3f}" for m in methods]
        lines.append(f"{g:0.3f} open" + "  ".join(cells))
    # IRR is the matched model: NEES stays near 1 on the whole sweep.
    nees_irr = nees_loop["irr"].mean(axis=1)
    lines.append(f"IRR NEES with loop: min={nees_irr.min():.3f} max={nees_irr.max():.3f}")
    lines.append(
        f"segment edges [{SEG_LO},{SEG_HI}) at γ={SEG_GAMMA}, loop on"
    )
    lines.append("segment ATE  " + "  ".join(f"{seg_ate[m].mean():10.3f}" for m in methods))
    lines.append("segment NEES " + "  ".join(f"{seg_nees[m].mean():10.1f}" for m in methods))
    # Poses after the degenerate window: a loop can repair these only if
    # the window's factor is soft enough to absorb the correction.
    after = slice(SEG_HI - 1, None)
    lines.append(
        "after-window |err| "
        + "  ".join(f"{np.mean(np.abs(seg_path[m][:, after])):10.3f}" for m in methods)
    )
    text = "\n".join(lines) + "\n"
    (OUT / "summary.txt").write_text(text)
    return text


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    self_check()
    print("self-check passed")

    day1 = load_day1()
    gone, eigs, gammas, lambda_imu = corridor_gamma_curve(day1)
    plot_corridor(gone, eigs, gammas, lambda_imu)
    print(f"corridor weak-axis γ from {gammas[0, 0]:.3f} (wall present) to {gammas[-1, 0]:.3f} (pure corridor)")
    print(f"saved {OUT / 'corridor_gamma.png'}")

    # Log sweep plus the hard threshold and the gray-zone point used in the path plot.
    lambdas = np.geomspace(1e-2, 1e4, 24)
    lambdas = np.unique(np.sort(np.concatenate([lambdas, [irr_information(0.35), LAMBDA_NOMINAL]])))

    ate_l, nees_l = sweep(True, lambdas, seed=7)
    ate_o, nees_o = sweep(False, lambdas, seed=7)
    plot_sweep(lambdas, ate_l, nees_l, ate_o, nees_o)
    seg_ate, seg_nees, seg_path = run_segment(seed=11)
    plot_segment(seg_path)
    text = summarize(lambdas, ate_l, nees_l, ate_o, seg_ate, seg_nees, seg_path)
    print(text)
    print(f"saved {OUT / 'factor_sweep.png'}")
    print(f"saved {OUT / 'segment_path.png'}")


if __name__ == "__main__":
    sys.exit(main())
