#!/usr/bin/env python3
import os
from pathlib import Path
import numpy as np
from numba import njit
import h5py
from eqmetad.potentials import constant
from eqmetad.utils import make_nested_grid, make_grid, import_custom_potential, periodic_kernel_parameters
from eqmetad.fast_utils import (
    cell_mass_from_bias,
    cell_mass_from_bias_interval,
    hill_height,
    gaussian,
    gaussian_periodic,
    sample_density,
    gaussian_integral_on_interval,
    interp_periodic,
    interp,
    calc_theta_step
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
    theta_init,
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
    total_steps,
    start_step,
    chunk_size,
    stride,
    centers,
    left,
    right,
    n_images, 
    peak_norm, 
    integral_norm
):
    bias_centered = bias_init.copy()
    bias_level = bias_level_init
    time = time_init
    theta = theta_init
    
    norm = 1/peak_norm if k_mode == 0 else 1/integral_norm
    if k_mode == 0:  # unit_peak
        time_coeff = 1.0
    elif method == 0:  # periodic, unit_integral
        time_coeff = peak_norm / integral_norm
    else:  # bounds / interval, unit_integral
        time_coeff = 1.0 / (np.sqrt(2.0 * np.pi) * sigma)
    

    max_saves = chunk_size // stride + 2
    history_bias_pb = np.empty((max_saves, len(x)), dtype=np.float64)
    history_mass_pb = np.empty((max_saves, len(x)), dtype=np.float64)
    history_mass_pw = np.empty((max_saves, len(x)), dtype=np.float64)
    history_center = np.empty(max_saves, dtype=np.float64)
    history_height = np.empty(max_saves, dtype=np.float64)
    history_steps = np.empty(max_saves, dtype=np.int64)
    history_time = np.empty(max_saves, dtype=np.float64)
    history_theta = np.empty(max_saves, dtype=np.float64)

    log_rho = np.empty_like(x)
    rho = np.empty_like(x)
    cell_mass = np.empty_like(x)
    gauss_val = np.zeros_like(x)
    bias_interval = np.empty_like(x)

    save_idx = 0
    r_length = 1.0 / (right - left)
    ab = alpha/beta
    
    hill_mean = 0.0
    s_clamped = x

    if method == 0 and k_mode == 0:
        hill_mean = r_length * integral_norm / peak_norm
    if method == 0 and k_mode == 1:
        hill_mean = r_length    
    elif method == 2:
        s_clamped = np.clip(x, left, right)

    # save zero step to disk
    if start_step == 0:
        if method == 2:
            if F is not None:
                logZw = cell_mass_from_bias_interval(
                     bias_centered, F, s_clamped, x, log_rho, rho, cell_mass, beta, dx, alpha
                )
                logZb =  history_mass_pw[save_idx] = cell_mass
                cell_mass_from_bias_interval(
                     bias_centered, F, s_clamped, x, log_rho, rho, cell_mass, beta, dx, beta
                )
                history_mass_pb[save_idx] = cell_mass 
            for i in range(len(s_clamped)):
                bias_interval[i] = interp(s_clamped[i], x[0], dx, bias_centered)
            history_bias_pb[save_idx] = bias_interval
        else:
            if F is not None:
                logZw = cell_mass_from_bias(
                    bias_centered, F, log_rho, rho, cell_mass, beta, dx, alpha
                )
                history_mass_pw[save_idx] = cell_mass
                logZb = cell_mass_from_bias(
                   bias_centered, F, log_rho, rho, cell_mass, beta, dx, beta
               )
                history_mass_pb[save_idx] = cell_mass
            history_bias_pb[save_idx] = bias_centered
            
        history_center[save_idx] = np.nan
        history_height[save_idx] = np.nan
        history_steps[save_idx] = 0
        history_time[save_idx] = time
        history_theta[save_idx] = theta
        save_idx += 1        

    #####################
    # Main loop
    #####################
    for i in range(1, chunk_size+1):
        current_step = start_step + i
        should_save = ( current_step % stride == 0
                        or current_step == total_steps
                       )       
            
        #1. Sample distribution 
        center = centers[current_step-1]

        #2. Compute hill and its mean
        if method != 0:
            hill = gaussian(x, center, gauss_val, sigma, k_mode)
            hill_mean = r_length * gaussian_integral_on_interval(center, sigma, k_mode, left, right)
        else:
            hill = gaussian_periodic(x, center, gauss_val, sigma, left, right, n_images, norm)
            
        #3. Deposit hill and update V
        if m_mode == 1:
            center_clipped = max(left, min(center, right))
            if method == 0:
                bias_at_center = interp_periodic(center_clipped, x[0], dx, bias_centered)
            else:
                bias_at_center = interp(center_clipped, x[0], dx, bias_centered)
            bias_at_center += bias_level
            height_step = hill_height(height, r_delta_T, bias_at_center)
            time_factor = np.exp(-r_delta_T * bias_level)
            delta_tau = height * time_coeff * time_factor
            time += delta_tau
            bias_level += height_step * hill_mean

        elif m_mode == 0:
            height_step = height
            time_factor = current_step
            time = height * time_coeff * time_factor
        
        bias_centered += height_step * (hill - hill_mean)

        if m_mode == 1 and F is not None:
            if method == 2 or method == 3:
                logZb = cell_mass_from_bias_interval(
                        bias_centered, F, s_clamped, x, log_rho, rho, cell_mass, beta, dx, beta
                )
            else:
                logZb = cell_mass_from_bias(
                        bias_centered, F, log_rho, rho, cell_mass, beta, dx, beta
                )
            delta_theta = calc_theta_step(bias_centered, cell_mass, r_delta_T, delta_tau)
            theta += delta_theta 
        elif m_mode == 1:
            theta += height_step
        else:
            theta = time      
        
        # 4. save to disk
        if should_save:
            if method == 2:
                if F is not None:
                    logZw = cell_mass_from_bias_interval(
                        bias_centered, F, s_clamped, x, log_rho, rho, cell_mass, beta, dx, alpha
                    )
                    history_mass_pw[save_idx] = cell_mass
                    logZb = cell_mass_from_bias_interval(
                         bias_centered, F, s_clamped, x, log_rho, rho, cell_mass, beta, dx, beta
                    )
                    history_mass_pb[save_idx] = cell_mass 
                for i in range(len(s_clamped)):
                    bias_interval[i] = interp(s_clamped[i], x[0], dx, bias_centered)
                history_bias_pb[save_idx] = bias_interval
            else:
                if F is not None:
                    logZw = cell_mass_from_bias(
                        bias_centered, F, log_rho, rho, cell_mass, beta, dx, alpha
                    )
                    history_mass_pw[save_idx] = cell_mass
                    logZb = cell_mass_from_bias(
                        bias_centered, F, log_rho, rho, cell_mass, beta, dx, beta
                    )
                    history_mass_pb[save_idx] = cell_mass
                history_bias_pb[save_idx] = bias_centered
            
            history_center[save_idx] = center
            history_height[save_idx] = height_step
            history_steps[save_idx] = current_step
            history_time[save_idx] = time
            history_theta[save_idx] = theta
            save_idx += 1 

    return (
        bias_centered,
        bias_level,
        time,
        theta,
        history_steps[:save_idx],
        history_center[:save_idx],
        history_height[:save_idx],
        history_bias_pb[:save_idx],
        history_mass_pb[:save_idx],
        history_mass_pw[:save_idx],
        history_time[:save_idx],
        history_theta[:save_idx]
    )


def save_data_to_disk(
    method, total_steps, chunk_size, stride, seed, n_grid,
    filename, m_mode, k_mode, x, edges, dx, F, alpha, beta,
    sigma, bias_factor, r_delta_T, height, centers, left, right, n_images, peak_norm, integral_norm
    ):

    @njit
    def set_seed(s):
        np.random.seed(s)
    set_seed(seed)

    bias_centered = np.zeros_like(x)
    bias_level = 0.0
    time, theta = 0.0, 0.0
        
    total_saves = ( 1 + total_steps // stride
    + (1 if total_steps % stride != 0 else 0))

    with h5py.File(filename, "w") as f:
        d_steps = f.create_dataset("steps", (total_saves,), dtype="i8")
        d_centers = f.create_dataset("centers", (total_saves,), dtype="f8")
        d_heights = f.create_dataset("heights", (total_saves,), dtype="f8")
        d_time = f.create_dataset("time", (total_saves,), dtype="f8")
        d_theta = f.create_dataset("theta", (total_saves,), dtype="f8")

        d_bias_pb = f.create_dataset(
            "bias_pb", (total_saves, n_grid), dtype="f8",
            compression="gzip", chunks=(min(100, total_saves), n_grid)
        )
        if F is not None:
            d_mass_pb = f.create_dataset(
            "cell_mass_pb", (total_saves, n_grid), dtype="f8",
            compression="gzip", chunks=(min(100, total_saves), n_grid)
            )
            d_mass_pw = f.create_dataset(
            "cell_mass_pw", (total_saves, n_grid), dtype="f8",
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
                theta,
                h_steps,
                h_centers,
                h_heights,
                h_bias_pb,
                h_mass_pb,
                h_mass_pw,
                h_time,
                h_theta
            ) = read_chunk(
                method, m_mode, k_mode, bias_centered, bias_level, time, theta, 
                x, edges, dx, F, alpha, beta, sigma, bias_factor, r_delta_T,
                height, total_steps, start_step, chunk_size_valid, stride, centers,  left, right,
                n_images, peak_norm, integral_norm
            )

            n_new_records = len(h_steps)
            if n_new_records > 0:
                next_idx = write_idx + n_new_records
                d_steps[write_idx:next_idx] = h_steps
                d_time[write_idx:next_idx] = h_time
                d_theta[write_idx:next_idx] = h_theta
                d_centers[write_idx:next_idx] = h_centers
                d_heights[write_idx:next_idx] = h_heights
                d_bias_pb[write_idx:next_idx, :] = h_bias_pb
                if F is not None:
                    d_mass_pb[write_idx:next_idx, :] = h_mass_pb
                    d_mass_pw[write_idx:next_idx, :] = h_mass_pw

                write_idx = next_idx

            print(f"Progress: {start_step + chunk_size_valid} / {total_steps} steps are saved.")


def main() -> None:
    cfg = load_config()
    beta = 1.0 / cfg.kBT
    r_delta_T = beta / (cfg.bias_factor - 1.0) if cfg.metad_mode == "wt" else 0
    alpha = beta + r_delta_T if cfg.metad_mode == "wt" else beta

    k_mode = kernel_modes[cfg.kernel_mode]
    m_mode = metad_modes[cfg.metad_mode]
    n_images, peak_norm, integral_norm = 0, 1.0, 1.0
    
    method = None
    if cfg.method == "periodic":
        x, edges, dx = make_grid(cfg.min, cfg.max, cfg.n_grid)
        method = 0
        n_images, peak_norm, integral_norm = (
            periodic_kernel_parameters(cfg.sigma, cfg.min, cfg.max, cfg.periodic_tol)
        )
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
        n_grid=len(x),
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
        right=cfg.max,
        n_images = n_images,
        peak_norm = peak_norm,
        integral_norm = integral_norm
    )

if __name__ == "__main__":
    main()

