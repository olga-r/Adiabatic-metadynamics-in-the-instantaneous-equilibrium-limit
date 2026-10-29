#!/usr/bin/env python3
import h5py
import numpy as np
import json
import math
import os
from pathlib import Path
from scipy.stats import entropy
from eqmetad.peaks import detect_peaks, analyze_peaks
from eqmetad.config_manager import load_config
from eqmetad.potentials import constant
from eqmetad.utils import make_nested_grid, make_grid, natural_sort,import_custom_potential
from matplotlib import pyplot as plt
from copy import deepcopy
from matplotlib.figure import Figure
import matplotlib as mpl


def plot_masses(mass_in_peaks, time):
    fig = Figure(linewidth=4, figsize = (7,6), dpi =300)
    gs = mpl.gridspec.GridSpec(nrows=1, ncols=1, left =0.1, right=0.95, bottom=0.17, top=0.92,  wspace=0.4, hspace=0.2)
    ax1 = fig.add_subplot(gs[0,0])
    ax1.set_xlabel("s time")
    ax1.set_ylabel("Mass in peaks")
    ax1.plot(time, mass_in_peaks, lw = 1, c="b")
    fig.savefig(os.path.normpath('./mass_in_peaks.png'))

def plot_p(p, step, grid_centers):
    fig = Figure(linewidth=4, figsize = (7,6), dpi =300)
    gs = mpl.gridspec.GridSpec(nrows=1, ncols=1, left =0.1, right=0.95, bottom=0.17, top=0.92,  wspace=0.4, hspace=0.2)
    ax1 = fig.add_subplot(gs[0,0])
    ax1.set_ylim(-0.2, 10)
    ax1.set_xlabel("x")
    ax1.set_ylabel("p")
    ax1.plot(grid_centers,p, lw = 1, c="b")
    fig.savefig(os.path.normpath('./p_step={}.png'.format(step)))

def plot_kl_distance(distance, time):
    fig = Figure(linewidth=4, figsize = (7,6), dpi =300)
    gs = mpl.gridspec.GridSpec(nrows=1, ncols=1, left =0.1, right=0.95, bottom=0.17, top=0.92,  wspace=0.4, hspace=0.2)
    ax1 = fig.add_subplot(gs[0,0])
    ax1.set_xlabel("s time")
    ax1.set_ylabel("KL distance")
    ax1.plot(time, distance, lw = 1, c="b")
    fig.savefig(os.path.normpath('./KL_distance.png'))


def main() -> None:
    cfg = load_config()
    beta = 1.0 / cfg.kBT
    r_delta_T = beta / (cfg.bias_factor - 1.0) if cfg.metad_mode == "wt" else 0
    alpha = beta + r_delta_T if cfg.metad_mode == "wt" else beta
    ab = alpha/beta
    md_time = np.arange(0, cfg.total_steps+1, cfg.stride)
    if md_time[-1] != cfg.total_steps:
        md_time = np.append(md_time, cfg.total_steps)
    md_time =md_time.reshape(-1, 1)
    
    if cfg.plot_stride <= 0:
        plt_stride = 0
        print(f"Plotting $p_w$ is disabled.")
    elif cfg.plot_stride >= cfg.stride:
        plt_stride = max(1, round(cfg.plot_stride / cfg.stride))
        print(f"Plotting $p_w$ every {plt_stride } saved frames.")
    else:
        plt_stride = cfg.plot_stride
        print(f"Plotting $p_w$ every {plt_stride } saved frames.")
    
    current_dir = Path(os.getcwd())
    full_path = (current_dir / cfg.base_dir / cfg.filename).resolve()

    f = h5py.File(full_path, 'r')
    ( bias_pb, hills_centers, tau, theta,
        heights, steps, grid_edges, grid_centers
    )  = (  f['bias_pb'][:],   f['centers'][:],
           f['tau'][:],  f['theta'][:], f['heights'][:], f['steps'][:],
           f['grid_edges'][:], f['grid_x'][:])
    dx = grid_edges[1]-grid_edges[0]
    
    if cfg.potential != "none":
        cell_mass_pb =f['cell_mass_pb'][:]
        cell_mass_pw =f['cell_mass_pw'][:]
        kl_distance = np.zeros_like(steps, dtype=float)
        mass_in_peaks = np.zeros_like(steps, dtype=float)
        all_data = {}
        periodic = False
        if cfg.method == "periodic":
            periodic=True
        for step_idx, step in enumerate(steps):
            masses = np.squeeze(cell_mass_pw[step_idx])
            p = masses / dx
            if plt_stride > 0 and step_idx%plt_stride == 0:
                plot_p(p, step,grid_centers)
            if cfg.method in ("interval", "mcgovern_interval"):
                p = p[(grid_centers > cfg.min) & (grid_centers < cfg.max)]
                masses = masses[(grid_centers > cfg.min) & (grid_centers < cfg.max)]
                grid_centers_interval = grid_centers[(grid_centers > cfg.min) & (grid_centers < cfg.max)]

            else:
                grid_centers_interval = grid_centers

            kl_distance[step_idx] = entropy(p, np.ones_like(p))

            peak_indices = detect_peaks(
                dx=dx, sigma=cfg.sigma,
                peak_threshold=cfg.peak_threshold,
                filter_width=cfg.filter_width, p = p, periodic=periodic
            )

            step_mass, step_data = analyze_peaks(
                peaks_indices=peak_indices, p=p, grid_centers=grid_centers_interval, 
                masses=masses, left = cfg.min, right = cfg.max,dx, periodic=periodic
            )
            mass_in_peaks[step_idx] = step_mass
            all_data[int(step)] = step_data

        with open("peaks.json", "w") as ff:
            json.dump(all_data, ff, indent=4)
        np.savetxt('mass_in_peaks.txt', mass_in_peaks)
        np.savetxt('KL_distance.txt', kl_distance)
        np.save("pw", f['cell_mass_pw'][:]/dx)
        np.save("pb", f['cell_mass_pb'][:]/dx)
        plot_masses(mass_in_peaks, theta*alpha)
        plot_kl_distance(kl_distance, theta*alpha)
        np.savetxt( 'time.txt', np.column_stack((md_time, tau, theta, theta*alpha)),
                   header="md_time    tau       theta         s"
                  )
    else:
        np.savetxt( 'time.txt', np.column_stack((md_time, tau, theta, theta*alpha)),
                   header="md_time      tau       theta_estimate        s_estimate"
                  )
    
    np.savetxt('heights.txt', np.column_stack((heights, heights*ab)),  
               header = 'heights (pb)    scaled_heights (pw)'
              )
    np.savetxt('centers.txt', hills_centers)   
    np.save("bias_V_pw", bias_pb*ab)
    np.save("bias_V_pb", bias_pb)
    np.save("bias_U", bias_pb*alpha)
    np.savetxt("grid_centers.txt", grid_centers)



if __name__ == "__main__":
    main()







