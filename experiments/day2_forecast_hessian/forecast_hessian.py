#!/usr/bin/env python3
"""Day 2: forecast Ĥ at T̂_{t+k} on a bounded local map.

Claim this script is allowed to test (and only this claim):

    Current-scan Hessian H_t uses a correspondence only if that world
    point is still in the local map. A 360° scan can still contain the
    entrance, while a FIFO radius map has already dropped it. Then H_t
    collapses even though the wall is in the current cloud. Keeping
    points that cover the *predicted* weak axis V_D of Ĥ_{t+k} stops
    that collapse. Historical frames of a wall the current scan does
    *not* see never enter H_t — this script does not claim they do.

Geometry (world frame, corridor along +Y):

    entrance / doorway  at y = 0,  normal +Y
    side walls          x = ±width/2
    ground              z = 0
    no far end-cap      (otherwise forward translation stays observable)

    robot starts just inside the corridor and drives +Y at constant speed.
    LiDAR is 360° range-limited. FIFO map radius < sensor range.

Run:
    python3 forecast_hessian.py
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = Path(__file__).resolve().parent / "out"

# Scene / sensor. Map radius is *shorter* than range so the entrance can
# still be in the scan after FIFO has dropped it.
Y0 = 2.5
VEL_Y = 2.0
DT = 0.2
T_END = 7.0
HORIZON = 1.0  # seconds ahead for Ĥ_{t+k}
R_SENSOR = 15.0
R_MAP = 8.0
WIDTH = 4.0
HEIGHT = 3.0
CORRIDOR_LEN = 30.0
LIDAR_Z = 1.2
ALIGN_COS = np.cos(np.deg2rad(60.0))  # |n · V_D| to protect a plane
DEGEN_ABS = 30.0  # λ_min below this ⇒ treat as degenerate


def sample_plane(center, normal, u, v, nu, nv) -> tuple[np.ndarray, np.ndarray]:
    """Axis-aligned patch. ``u`` and ``v`` set *extent*, not only direction."""
    n = np.asarray(normal, float)
    n = n / np.linalg.norm(n)
    u = np.asarray(u, float)
    v = np.asarray(v, float)
    hu, hv = 0.5 * np.linalg.norm(u), 0.5 * np.linalg.norm(v)
    u = u / np.linalg.norm(u)
    v = v / np.linalg.norm(v)
    su = np.linspace(-hu, hu, nu)
    sv = np.linspace(-hv, hv, nv)
    aa, bb = np.meshgrid(su, sv, indexing="xy")
    pts = np.asarray(center, float) + aa.reshape(-1, 1) * u + bb.reshape(-1, 1) * v
    nrm = np.repeat(n[None, :], len(pts), axis=0)
    return pts, nrm


def build_world() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    half_w = WIDTH / 2
    chunks: list[tuple[np.ndarray, np.ndarray, str]] = []
    chunks.append(
        (
            *sample_plane(
                np.array([0.0, 0.0, HEIGHT / 2]),
                np.array([0.0, 1.0, 0.0]),
                np.array([WIDTH, 0.0, 0.0]),
                np.array([0.0, 0.0, HEIGHT]),
                18,
                10,
            ),
            "entrance",
        )
    )
    chunks.append(
        (
            *sample_plane(
                np.array([-half_w, CORRIDOR_LEN / 2, HEIGHT / 2]),
                np.array([1.0, 0.0, 0.0]),
                np.array([0.0, CORRIDOR_LEN, 0.0]),
                np.array([0.0, 0.0, HEIGHT]),
                50,
                8,
            ),
            "left",
        )
    )
    chunks.append(
        (
            *sample_plane(
                np.array([half_w, CORRIDOR_LEN / 2, HEIGHT / 2]),
                np.array([-1.0, 0.0, 0.0]),
                np.array([0.0, CORRIDOR_LEN, 0.0]),
                np.array([0.0, 0.0, HEIGHT]),
                50,
                8,
            ),
            "right",
        )
    )
    chunks.append(
        (
            *sample_plane(
                np.array([0.0, CORRIDOR_LEN / 2, 0.0]),
                np.array([0.0, 0.0, 1.0]),
                np.array([WIDTH, 0.0, 0.0]),
                np.array([0.0, CORRIDOR_LEN, 0.0]),
                12,
                50,
            ),
            "ground",
        )
    )
    P = np.vstack([c[0] for c in chunks])
    N = np.vstack([c[1] for c in chunks])
    labels = np.concatenate([np.repeat(c[2], len(c[0])) for c in chunks])
    return P, N, labels


def accumulate_H(points: np.ndarray, normals: np.ndarray) -> np.ndarray:
    if len(points) == 0:
        return np.zeros((6, 6))
    cross = np.cross(points, normals)
    A = np.hstack([cross, normals])
    return A.T @ A


def robot_pose(t: float) -> np.ndarray:
    return np.array([0.0, Y0 + VEL_Y * t, LIDAR_Z])


def in_range(P_w: np.ndarray, origin: np.ndarray, radius: float) -> np.ndarray:
    return np.linalg.norm(P_w - origin, axis=1) < radius


def to_lidar(P_w: np.ndarray, N_w: np.ndarray, origin: np.ndarray):
    """Yaw = 0, so R = I. Translation only."""
    return P_w - origin, N_w


def hessian_from_masks(P_w, N_w, origin, scan_m, map_m):
    """Correspondence exists only if the same world point is in scan *and* map."""
    keep = scan_m & map_m
    P_l, N_l = to_lidar(P_w[keep], N_w[keep], origin)
    H = accumulate_H(P_l, N_l)
    evals_t, evecs_t = np.linalg.eigh(H[3:, 3:])
    return H, evals_t, evecs_t, int(keep.sum())


def predicted_vd(P_w, N_w, origin_hat):
    """V_D at the predicted pose under FIFO. None if Ĥ is still well posed."""
    scan = in_range(P_w, origin_hat, R_SENSOR)
    fifo = in_range(P_w, origin_hat, R_MAP)
    _, evals_t, evecs_t, _ = hessian_from_masks(P_w, N_w, origin_hat, scan, fifo)
    if evals_t[0] >= DEGEN_ABS:
        return None, evals_t
    v_world = evecs_t[:, 0]  # yaw = 0
    v_world = v_world / (np.linalg.norm(v_world) + 1e-12)
    return v_world, evals_t


@dataclass
class Row:
    t: float
    y: float
    scan_has_entrance: bool
    fifo_has_entrance: bool
    lmin_fifo: float
    lmin_hat: float
    lmin_vd: float
    lmin_oracle: float
    v_fifo: list
    v_hat: list
    v_vd: list
    n_fifo: int
    n_vd: int
    predicted_degen: bool


def fmt_vec(v: np.ndarray) -> str:
    return "[" + ", ".join(f"{x:+.3f}" for x in v) + "]"


def run() -> list[Row]:
    P_w, N_w, labels = build_world()
    is_ent = labels == "entrance"
    times = np.arange(0.0, T_END + 1e-9, DT)
    rows: list[Row] = []

    for t in times:
        origin = robot_pose(t)
        origin_hat = robot_pose(t + HORIZON)
        scan = in_range(P_w, origin, R_SENSOR)
        fifo = in_range(P_w, origin, R_MAP)
        oracle = scan.copy()

        vd, evals_hat = predicted_vd(P_w, N_w, origin_hat)
        if vd is None:
            protect = np.zeros(len(P_w), dtype=bool)
        else:
            protect = np.abs(N_w @ vd) >= ALIGN_COS
        vd_map = fifo | (scan & protect)

        _, ev_f, vc_f, n_f = hessian_from_masks(P_w, N_w, origin, scan, fifo)
        _, ev_v, vc_v, n_v = hessian_from_masks(P_w, N_w, origin, scan, vd_map)
        _, ev_o, _, _ = hessian_from_masks(P_w, N_w, origin, scan, oracle)

        v_hat = np.zeros(3) if vd is None else vd
        rows.append(
            Row(
                t=float(t),
                y=float(origin[1]),
                scan_has_entrance=bool((scan & is_ent).any()),
                fifo_has_entrance=bool((fifo & is_ent).any()),
                lmin_fifo=float(ev_f[0]),
                lmin_hat=float(evals_hat[0]),
                lmin_vd=float(ev_v[0]),
                lmin_oracle=float(ev_o[0]),
                v_fifo=[float(x) for x in vc_f[:, 0]],
                v_hat=[float(x) for x in v_hat],
                v_vd=[float(x) for x in vc_v[:, 0]],
                n_fifo=n_f,
                n_vd=n_v,
                predicted_degen=vd is not None,
            )
        )
    return rows


def verdict(rows: list[Row]) -> dict:
    """The three checks that decide whether Day 2 supports the claim."""
    gap_now_ok_hat_bad = [
        r
        for r in rows
        if r.lmin_fifo >= DEGEN_ABS and r.lmin_hat < DEGEN_ABS and r.scan_has_entrance
    ]
    visible_but_fifo_dropped = [
        r
        for r in rows
        if r.scan_has_entrance and not r.fifo_has_entrance and r.lmin_fifo < DEGEN_ABS
    ]
    vd_saves = [
        r
        for r in rows
        if r.scan_has_entrance and not r.fifo_has_entrance and r.lmin_vd >= DEGEN_ABS
    ]
    along_tunnel = [
        r
        for r in visible_but_fifo_dropped
        if abs(r.v_fifo[1]) > 0.85  # world +Y
    ]
    return {
        "forecast_ahead_of_current_fifo": len(gap_now_ok_hat_bad) > 0,
        "n_times_forecast_ahead": len(gap_now_ok_hat_bad),
        "visible_but_fifo_dropped_collapses_H": len(visible_but_fifo_dropped) > 0,
        "n_times_visible_fifo_drop": len(visible_but_fifo_dropped),
        "keeping_vd_saves_current_H": len(vd_saves) > 0,
        "n_times_vd_saves": len(vd_saves),
        "collapsed_axis_along_world_Y": len(along_tunnel) > 0,
        "first_forecast_lead_t": gap_now_ok_hat_bad[0].t if gap_now_ok_hat_bad else None,
        "first_visible_drop_t": visible_but_fifo_dropped[0].t
        if visible_but_fifo_dropped
        else None,
        "pass": bool(
            gap_now_ok_hat_bad
            and visible_but_fifo_dropped
            and vd_saves
            and along_tunnel
        ),
    }


def plot_rows(rows: list[Row]) -> Path:
    t = np.array([r.t for r in rows])
    fig, axes = plt.subplots(2, 1, figsize=(8.2, 6.2), sharex=True)

    ax = axes[0]
    ax.plot(t, [r.lmin_fifo for r in rows], "k-o", ms=3.5, label=r"$\lambda_{\min}(H_t)$ FIFO")
    ax.plot(
        t,
        [r.lmin_hat for r in rows],
        "C0--s",
        ms=3.5,
        label=rf"$\lambda_{{\min}}(\hat H_{{t+{HORIZON:.1f}s}})$ FIFO",
    )
    ax.plot(t, [r.lmin_vd for r in rows], "C3-^", ms=3.5, label=r"$\lambda_{\min}(H_t)$ keep $V_D$")
    ax.plot(t, [r.lmin_oracle for r in rows], "C2:", lw=2, label=r"$\lambda_{\min}(H_t)$ all visible")
    ax.axhline(DEGEN_ABS, color="0.5", ls=":", lw=1, label=f"degen thresh = {DEGEN_ABS}")
    shade = np.array([r.scan_has_entrance and not r.fifo_has_entrance for r in rows])
    if shade.any():
        ax.fill_between(t, 0, ax.get_ylim()[1] if ax.get_ylim()[1] > 1 else 1, where=shade, color="C3", alpha=0.12, label="scan has entrance, FIFO map does not")
    ax.set_ylabel(r"$\lambda_{\min}(H_{tt})$")
    ax.set_title("Day 2: forecast collapses before FIFO; keeping $V_D$ saves current $H$")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", fontsize=8)

    ax2 = axes[1]
    ax2.step(t, [int(r.scan_has_entrance) for r in rows], where="mid", label="entrance in current scan")
    ax2.step(t, [int(r.fifo_has_entrance) for r in rows], where="mid", label="entrance in FIFO map")
    ax2.step(t, [int(r.predicted_degen) for r in rows], where="mid", label=f"FIFO at t+{HORIZON:.1f}s degenerate")
    ax2.set_ylim(-0.1, 1.2)
    ax2.set_xlabel("time (s)   robot y = 2.5 + 2 t")
    ax2.set_ylabel("boolean")
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc="upper right", fontsize=8)

    fig.tight_layout()
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "forecast_lmin.png"
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = run()
    v = verdict(rows)
    plot_path = plot_rows(rows)

    print("Setup")
    print(f"  corridor +Y, entrance at y=0, start y={Y0}, v={VEL_Y} m/s")
    print(f"  R_sensor={R_SENSOR} m, R_map={R_MAP} m, horizon={HORIZON} s")
    print(f"  correspondence = current scan ∩ local map (same world points)")
    print()
    print(f"{'t':>5} {'y':>6} {'scanE':>5} {'mapE':>5} {'λFIFO':>8} {'λHAT':>8} {'λVD':>8} {'vFIFO':>22}")
    for r in rows[::2]:
        print(
            f"{r.t:5.1f} {r.y:6.2f} {str(r.scan_has_entrance):>5} {str(r.fifo_has_entrance):>5} "
            f"{r.lmin_fifo:8.1f} {r.lmin_hat:8.1f} {r.lmin_vd:8.1f} {fmt_vec(np.array(r.v_fifo)):>22}"
        )

    print("\nVerdict (must all be true for Day 2 to support the claim)")
    for k in (
        "forecast_ahead_of_current_fifo",
        "visible_but_fifo_dropped_collapses_H",
        "keeping_vd_saves_current_H",
        "collapsed_axis_along_world_Y",
        "pass",
    ):
        print(f"  {k}: {v[k]}")
    print(f"  first t where Ĥ is already bad but H_t FIFO is still OK: {v['first_forecast_lead_t']}")
    print(f"  first t where scan sees entrance but FIFO dropped it:     {v['first_visible_drop_t']}")
    print(f"\nSaved {plot_path}")

    with (OUT / "verdict.json").open("w") as f:
        json.dump({"verdict": v, "rows": [asdict(r) for r in rows]}, f, indent=2)

    if not v["pass"]:
        raise SystemExit("Day 2 criteria failed — do not treat this as support for the paper claim.")


if __name__ == "__main__":
    main()
