#!/usr/bin/env python3
import os
from pathlib import Path
import numpy as np
from numba import njit
import h5py
from eqmetad.potentials import constant
from eqmetad.utils import make_nested_grid, make_grid, import_custom_potential
from eqmetad.fast_utils import (
    cell_mass_from_bias,
    cell_mass_from_bias_interval,
    hill_height,
    gaussian,
    gaussian_periodic,
    sample_density,
    gaussian_integral_on_interval,
    interp_periodic,
    interp
)
from eqmetad.config_manager import load_config

kernel_modes = {"unit_peak": 0, "unit_integral": 1}
metad_modes = {"normal": 0, "wt": 1}


@njit
def read_chunk(
    method,
    m_mode,
    k_mode,
    bias_init,
    bias_level_init,
    time_init,
    x,
    edges,
    dx,
    F,
    alpha,
    beta,
    sigma,
    bias_factor,
    r_delta_T,
    height,
    height_step,
    start_step,
    chunk_size,
    stride,
    centers,
    left,
    right,
):
    bias_centered = bias_init.copy()
    bias_level = bias_level_init
    time = time_init


    max_saves = chunk_size // stride + 1
    history_bias = np.empty((max_saves, len(x)), dtype=np.float64)
    history_mass = np.empty((max_saves, len(x)), dtype=np.float64)
    history_center = np.empty(max_saves, dtype=np.float64)
    history_height = np.empty(max_saves, dtype=np.float64)
    history_steps = np.empty(max_saves, dtype=np.int64)
    history_time = np.empty(max_saves, dtype=np.float64)

    log_rho = np.empty_like(x)
    rho = np.empty_like(x)
    cell_mass = np.empty_like(x)
    gauss_val = np.empty_like(x)

    save_idx = 0
    r_length = 1.0 / (right - left)
    ab = alpha/beta
    
    hill_mean = 0.0
    s_clamped = x

    if method == 0:
        center_init = 0.5 * (left + right)
        hill_mean = r_length * gaussian_integral_on_interval(center_init, sigma, k_mode, left, right)
    elif method == 2:
        s_clamped = np.clip(x, left, right)

    #####################
    # Main loop
    #####################
    for i in range(0, chunk_size):
        current_step = start_step + i
        
        # 1. save to disk
        if current_step % stride == 0:
            if F is not None:
                if method == 2:
                    cell_mass_from_bias_interval(
                             bias_centered, F, s_clamped, x, log_rho, rho, cell_mass, beta, dx, alpha
                    )
                else:
                    cell_mass_from_bias(
                            bias_centered, F, log_rho, rho, cell_mass, beta, dx, alpha
                    )
                history_mass[save_idx] = cell_mass
            if method == 2:
                for i in range(len(s_clamped)):
                    bias_interval[i] = interp(s_clamped[i], x[0], dx, bias_centered)
                    history_bias[save_idx] = ab*bias_interval
            else:
                history_bias[save_idx] = ab*bias_centered
            if current_step == 0:
                history_center[save_idx] = 0.5*(left+right)
            else:
                history_center[save_idx] = centers[current_step-1]
            history_height[save_idx] = height_step
            history_steps[save_idx] = current_step
            history_time[save_idx] = time
            save_idx += 1
            
        #2. Sample distribution 
        center = centers[current_step]

        #3. Compute hill and its mean
        if method != 0:
            hill = gaussian(x, center, gauss_val, sigma, k_mode)
            hill_mean = r_length * gaussian_integral_on_interval(center, sigma, k_mode, left, right)
        else:
            hill = gaussian_periodic(x, center, gauss_val, sigma, k_mode, left, right)

        #4. Deposit hill and update V
        if m_mode == 1:
            center_clipped = max(left, min(center, right))
            if method == 0:
                bias_at_center = interp_periodic(center_clipped, x[0], dx, bias_centered)
            else:
                bias_at_center = interp(center_clipped, x[0], dx, bias_centered)
            bias_at_center += bias_level
            height_step = hill_height(height, r_delta_T, bias_at_center)
            time_factor = np.exp(-r_delta_T * bias_level)
            time += height * time_factor
            bias_level += height_step * hill_mean

        elif m_mode == 0:
            height_step = height
            time_factor = current_step
            time = height * time_factor

        bias_centered += height_step * (hill - hill_mean)
        removed_mean = r_length * dx * np.sum(bias_centered)
        bias_centered -= removed_mean

        if m_mode == 1:
            bias_level += removed_mean

    return (
        bias_centered,
        bias_level,
        time,
        height_step,
        history_steps[:save_idx],
        history_center[:save_idx],
        history_height[:save_idx],
        history_bias[:save_idx],
        history_mass[:save_idx],
        history_time[:save_idx],
    )


def save_data_to_disk(
    method, total_steps, chunk_size, stride, seed, n_grid,
    filename, m_mode, k_mode, x, edges, dx, F, alpha, beta,
    sigma, bias_factor, r_delta_T, height, centers, left, right
    ):

    @njit
    def set_seed(s):
        np.random.seed(s)
    set_seed(seed)

    bias_centered = np.zeros_like(x)
    bias_level = 0.0
    time = 0.0
    height_step = 0.0
        
    total_saves = total_steps // stride +1

    with h5py.File(filename, "w") as f:
        d_steps = f.create_dataset("steps", (total_saves,), dtype="i8")
        d_centers = f.create_dataset("centers", (total_saves,), dtype="f8")
        d_heights = f.create_dataset("heights", (total_saves,), dtype="f8")
        d_time = f.create_dataset("time", (total_saves,), dtype="f8")

        d_bias = f.create_dataset(
            "bias", (total_saves, n_grid), dtype="f8",
            compression="gzip", chunks=(min(100, total_saves), n_grid)
        )
        if F is not None:
            d_mass = f.create_dataset(
                "cell_mass", (total_saves, n_grid), dtype="f8",
                compression="gzip", chunks=(min(100, total_saves), n_grid)
            )

        f.create_dataset("grid_x", data=x)
        f.create_dataset("grid_edges", data=edges)

        print(f"Start {total_steps} steps...")

        write_idx = 0
        for start_step in range(0, total_steps, chunk_size):
            chunk_size_valid = min(chunk_size, total_steps-start_step)
            (
                bias_centered,
                bias_level,
                time,
                height_step,
                h_steps,
                h_centers,
                h_heights,
                h_bias,
                h_mass,
                h_time
            ) = read_chunk(
                method, m_mode, k_mode, bias_centered, bias_level, time,
                x, edges, dx, F, alpha, beta, sigma, bias_factor, r_delta_T,
                height, height_step, start_step, chunk_size_valid, total_steps, stride, centers,  left, right
            )

            n_new_records = len(h_steps)
            if n_new_records > 0:
                next_idx = write_idx + n_new_records

                d_steps[write_idx:next_idx] = h_steps
                d_time[write_idx:next_idx] = h_time
                d_centers[write_idx:next_idx] = h_centers
                d_heights[write_idx:next_idx] = h_heights
                d_bias[write_idx:next_idx, :] = h_bias
                if F is not None:
                    d_mass[write_idx:next_idx, :] = h_mass

                write_idx = next_idx

            print(f"Progress: {start_step + chunk_size_valid} / {total_steps} steps are saved.")


def main() -> None:
    cfg = load_config()
    beta = 1.0 / cfg.kBT
    r_delta_T = beta / (cfg.bias_factor - 1.0) if cfg.metad_mode == "wt" else -1
    alpha = beta + r_delta_T if cfg.metad_mode == "wt" else beta

    k_mode = kernel_modes[cfg.kernel_mode]
    if k_mode == 1:
        cfg.height *= np.sqrt(2.0 * np.pi) * cfg.sigma
    m_mode = metad_modes[cfg.metad_mode]

    method = None
    if cfg.method == "periodic":
        x, edges, dx = make_grid(cfg.min, cfg.max, cfg.n_grid)
        method = 0
    elif cfg.method == "bounds":
        x, edges, dx = make_grid(cfg.min, cfg.max, cfg.n_grid)
        method = 1
    elif cfg.method == "interval":
        x, edges, dx = make_nested_grid(cfg.out_min, cfg.out_max, cfg.min, cfg.max, cfg.n_grid)
        method = 2
    else:
        raise ValueError(f"Unknown method: {cfg.method}")

    if cfg.potential == "constant":
        F = constant(x,cfg.min, cfg.max)
    elif cfg.potential == "custom":
        custom_potential = import_custom_potential()
        F = custom_potential(x, cfg.min, cfg.max)
    elif cfg.potential=="none":
        F = None
    else:
        raise NotImplementedError("Please specify the implemented potential, or select 'None'.")

    current_dir = Path(os.getcwd())
    full_path = (current_dir / cfg.base_dir / cfg.filename).resolve()
    hill_path = (current_dir / cfg.base_dir / cfg.hill_file).resolve()

    centers = np.loadtxt(hill_path, usecols=(1,))
    cfg.total_steps = len(centers)

    save_data_to_disk(
        method=method,
        total_steps=cfg.total_steps,
        chunk_size=cfg.chunk_size,
        stride=cfg.stride,
        seed=cfg.seed,
        n_grid==len(x),
        filename=full_path,
        m_mode=m_mode,
        k_mode=k_mode,
        x=x,
        edges=edges,
        dx=dx,
        F=F,
        alpha=alpha,
        beta=beta,
        sigma=cfg.sigma,
        bias_factor=cfg.bias_factor,
        r_delta_T=r_delta_T,
        height=cfg.height,
        centers=centers,
        left=cfg.min,
        right=cfg.max
    )

if __name__ == "__main__":
    main()

