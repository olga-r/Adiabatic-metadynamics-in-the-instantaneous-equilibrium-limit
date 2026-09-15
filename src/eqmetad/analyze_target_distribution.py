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
    ax1.set_xlabel("Metad time")
    ax1.set_ylabel("Mass in peaks")
    ax1.plot(time, mass_in_peaks, lw = 1, c="b")
    fig.savefig(os.path.normpath('./mass_in_peaks.png'))

def plot_p(p, step, left, right):
    fig = Figure(linewidth=4, figsize = (7,6), dpi =300)
    gs = mpl.gridspec.GridSpec(nrows=1, ncols=1, left =0.1, right=0.95, bottom=0.17, top=0.92,  wspace=0.4, hspace=0.2)
    ax1 = fig.add_subplot(gs[0,0])
    ax1.set_ylim(-0.2, 10)
    ax1.set_xlabel("x")
    ax1.set_ylabel("p")
    ax1.plot(np.arange(left, right, (right-left)/len(p)),p, lw = 1, c="b")
    fig.savefig(os.path.normpath('./p_step={}.png'.format(step)))

def plot_kl_distance(distance, time):
    fig = Figure(linewidth=4, figsize = (7,6), dpi =300)
    gs = mpl.gridspec.GridSpec(nrows=1, ncols=1, left =0.1, right=0.95, bottom=0.17, top=0.92,  wspace=0.4, hspace=0.2)
    ax1 = fig.add_subplot(gs[0,0])
    ax1.set_xlabel("Metad time")
    ax1.set_ylabel("KL distance")
    ax1.plot(time, distance, lw = 1, c="b")
    fig.savefig(os.path.normpath('./KL_distance.png'))


def main() -> None:
    cfg = load_config()
    plt_stride = max(1, round(cfg.plot_stride / cfg.stride)) if cfg.plot_stride>=cfg.stride else cfg.plot_stride
    print(f"Plotting every {plt_stride} saved frames.")
    current_dir = Path(os.getcwd())
    full_path = (current_dir / cfg.base_dir / cfg.filename).resolve()
    if cfg.potential != "none":
        f = h5py.File(full_path, 'r')
        ( bias_pb, bias_pw, hills_centers, time,
        heights, steps, grid_edges, grid_centers
        )  = (  f['bias_pb'][:],  f['bias_pw'][:],  f['centers'][:],
           f['time'][:], f['heights'][:], f['steps'][:],
           f['grid_edges'][:], f['grid_x'][:])
        dx = grid_edges[1]-grid_edges[0]
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
            if step_idx%plt_stride == 0:
                plot_p(p, step,cfg.out_min, cfg.out_max)
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
                masses=masses, left = cfg.min, right = cfg.max, periodic=periodic
            )
            mass_in_peaks[step_idx] = step_mass
            all_data[int(step)] = step_data

        with open("peaks.json", "w") as ff:
            json.dump(all_data, ff, indent=4)
        np.savetxt('mass_in_peaks.txt', mass_in_peaks)
        np.savetxt('KL_distance.txt', kl_distance)
        np.savetxt('time.txt', time)
        np.save("pw", f['cell_mass_pw'][:]/dx)
        plot_masses(mass_in_peaks, time)
        plot_kl_distance(kl_distance, time)




if __name__ == "__main__":
    main()







