#!/usr/bin/env python3
"""Day 1: turn X-ICP / Zhang symbols into numbers.

Build synthetic point-to-plane correspondences, accumulate

    H = sum a_i a_i^T ,   a_i = [p × n ; n]

then print eigenvalues of the translation and rotation blocks
(same split as X-ICP, avoiding r/t scale mixing).

Run:
    python3 compute_spectrum.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

OUT = Path(__file__).resolve().parent / "out"
RNG = np.random.default_rng(0)


def hat(n: np.ndarray, p: np.ndarray) -> np.ndarray:
    """6-vector Jacobian column, x = [r, t], matching X-ICP eq. (2)."""
    return np.concatenate([np.cross(p, n), n])


def accumulate_H(points: np.ndarray, normals: np.ndarray) -> np.ndarray:
    H = np.zeros((6, 6))
    for p, n in zip(points, normals):
        a = hat(n, p)
        H += np.outer(a, a)
    return H


def split_eigs(H: np.ndarray):
    evals_r, evecs_r = np.linalg.eigh(H[:3, :3])
    evals_t, evecs_t = np.linalg.eigh(H[3:, 3:])
    # eigh returns ascending order: index 0 is the weakest axis
    return evals_r, evecs_r, evals_t, evecs_t


def fmt_vec(v: np.ndarray) -> str:
    return "[" + ", ".join(f"{x:+.3f}" for x in v) + "]"


def report(name: str, H: np.ndarray) -> None:
    evals_r, evecs_r, evals_t, evecs_t = split_eigs(H)
    print(f"\n=== {name} ===")
    print(f"  # correspondences implicit in H scale;  H_tt eigenvalues (weak→strong): {evals_t}")
    print(f"  weakest translation axis v_t_min = {fmt_vec(evecs_t[:, 0])}")
    print(f"  H_rr eigenvalues (weak→strong): {evals_r}")
    print(f"  weakest rotation axis    v_r_min = {fmt_vec(evecs_r[:, 0])}")
    cond_t = evals_t[-1] / max(evals_t[0], 1e-12)
    print(f"  cond(H_tt) = {cond_t:.1f}   (>>1 means translational degeneracy)")


def sample_plane(center, normal, u, v, nu, nv, noise=0.01):
    """Grid on a plane, n is unit normal."""
    n = normal / np.linalg.norm(normal)
    u = u / np.linalg.norm(u)
    v = v / np.linalg.norm(v)
    su = np.linspace(-0.5, 0.5, nu)
    sv = np.linspace(-0.5, 0.5, nv)
    pts, nrm = [], []
    for a in su:
        for b in sv:
            p = center + a * u + b * v + RNG.normal(0, noise, 3)
            pts.append(p)
            nrm.append(n)
    return np.asarray(pts), np.asarray(nrm)


def scene_corridor(length=20.0, width=4.0, yaw_deg=0.0):
    """Infinite-ish corridor along world +Y. LiDAR at origin of its own frame.

    yaw_deg rotates the robot in the world: degeneracy is along the corridor,
    so the weak axis in the LiDAR frame should follow that corridor.
    """
    yaw = np.deg2rad(yaw_deg)
    c, s = np.cos(yaw), np.sin(yaw)
    R_wl = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])  # world ← lidar
    # Build in world, then express p,n in lidar (paper Loc-Module frame).
    half_w, half_l, h = width / 2, length / 2, 3.0
    chunks = []
    # left / right walls (normals ±world X)
    chunks.append(sample_plane(np.array([-half_w, 0, 1.5]), np.array([1, 0, 0]),
                               np.array([0, length, 0]), np.array([0, 0, h]), 40, 12))
    chunks.append(sample_plane(np.array([half_w, 0, 1.5]), np.array([-1, 0, 0]),
                               np.array([0, length, 0]), np.array([0, 0, h]), 40, 12))
    # ground
    chunks.append(sample_plane(np.array([0, 0, 0]), np.array([0, 0, 1]),
                               np.array([width, 0, 0]), np.array([0, length, 0]), 16, 40))
    P_w = np.vstack([c[0] for c in chunks])
    N_w = np.vstack([c[1] for c in chunks])
    R_lw = R_wl.T
    P_l = (R_lw @ P_w.T).T
    N_l = (R_lw @ N_w.T).T
    N_l /= np.linalg.norm(N_l, axis=1, keepdims=True)
    return P_l, N_l


def scene_open_ground(size=30.0):
    """Only a ground plane: expect weak tx, ty, and yaw (rz)."""
    return sample_plane(np.array([0, 0, 0]), np.array([0, 0, 1]),
                        np.array([size, 0, 0]), np.array([0, size, 0]), 40, 40)


def scene_box_room(s=8.0):
    """Four walls + ground: should be well constrained translationally."""
    h = 3.0
    chunks = [
        sample_plane(np.array([0, 0, 0]), np.array([0, 0, 1]),
                     np.array([s, 0, 0]), np.array([0, s, 0]), 20, 20),
        sample_plane(np.array([-s / 2, 0, 1.5]), np.array([1, 0, 0]),
                     np.array([0, s, 0]), np.array([0, 0, h]), 20, 10),
        sample_plane(np.array([s / 2, 0, 1.5]), np.array([-1, 0, 0]),
                     np.array([0, s, 0]), np.array([0, 0, h]), 20, 10),
        sample_plane(np.array([0, -s / 2, 1.5]), np.array([0, 1, 0]),
                     np.array([s, 0, 0]), np.array([0, 0, h]), 20, 10),
        sample_plane(np.array([0, s / 2, 1.5]), np.array([0, -1, 0]),
                     np.array([s, 0, 0]), np.array([0, 0, h]), 20, 10),
    ]
    return np.vstack([c[0] for c in chunks]), np.vstack([c[1] for c in chunks])


def enter_corridor_series(n_steps=25):
    """Mix end-cap (constrains forward) with side walls. End-cap weight → 0."""
    lambdas = []
    axes = []
    for i, alpha in enumerate(np.linspace(1.0, 0.0, n_steps)):
        P_side, N_side = scene_corridor(yaw_deg=0.0)
        P_cap, N_cap = sample_plane(
            np.array([0, 10, 1.5]), np.array([0, -1, 0]),
            np.array([4, 0, 0]), np.array([0, 0, 3]), 20, 12)
        n_keep = int(alpha * len(P_cap))
        if n_keep > 0:
            P = np.vstack([P_side, P_cap[:n_keep]])
            N = np.vstack([N_side, N_cap[:n_keep]])
        else:
            P, N = P_side, N_side
        H = accumulate_H(P, N)
        evals_t, evecs_t = np.linalg.eigh(H[3:, 3:])
        lambdas.append(evals_t[0])
        axes.append(evecs_t[:, 0])
    return np.linspace(1.0, 0.0, n_steps), np.asarray(lambdas), np.asarray(axes)


def main():
    OUT.mkdir(parents=True, exist_ok=True)

    scenes = {
        "box_room (should be OK)": scene_box_room(),
        "open_ground (weak tx, ty, yaw)": scene_open_ground(),
        "corridor yaw=0 (weak along lidar +Y)": scene_corridor(yaw_deg=0.0),
        "corridor yaw=45 (weak axis should rotate in lidar frame)": scene_corridor(yaw_deg=45.0),
    }
    for name, (P, N) in scenes.items():
        report(name, accumulate_H(P, N))

    alphas, lmin, axes = enter_corridor_series()
    fig, ax = plt.subplots(figsize=(7, 3.6))
    ax.plot(1 - alphas, lmin, "k-o", ms=4)
    ax.set_xlabel("how much the end wall has disappeared (0=room-like, 1=pure corridor)")
    ax.set_ylabel(r"$\lambda_{\min}(H_{tt})$")
    ax.set_title("Entering a corridor: forward translation becomes unobservable")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "enter_corridor_lmin.png", dpi=140)
    print(f"\nSaved {OUT / 'enter_corridor_lmin.png'}")
    print("Weak translation axes while entering (lidar frame):")
    for a, v in zip(np.linspace(0, 1, len(axes))[::6], axes[::6]):
        print(f"  gone={a:.2f}  v_t_min={fmt_vec(v)}")

    print("\nWhat to look at:")
    print("  1. box_room: three translation eigs same order of magnitude.")
    print("  2. corridor yaw=0: v_t_min ≈ [0, 1, 0]  (along the tunnel).")
    print("  3. corridor yaw=45: v_t_min rotates, still along the tunnel, not lidar x/y axes.")
    print("  4. plot: lambda_min collapses as the end wall vanishes — that is degeneracy.")


if __name__ == "__main__":
    main()
