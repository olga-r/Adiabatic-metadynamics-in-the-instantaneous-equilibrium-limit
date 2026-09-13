import re
import numpy as np
from scipy.ndimage import gaussian_filter1d
import os
import importlib.util

def natural_sort(l):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    return sorted(l, key=alphanum_key)

def smoothen_log_density(p, filter_width, mode):
    minimum_level = max(np.max(p) * 1.0e-300, np.finfo(float).tiny)
    logp = np.log(np.maximum(p, minimum_level))
    smooth_logp = gaussian_filter1d(
        logp,
        sigma=filter_width,
        mode=mode,
    )
    return smooth_logp

def make_grid(left, right, n_grid):
    """Midpoint cells on [left, right]."""
    edges = np.linspace(left, right, n_grid + 1)
    dx = (right - left) / n_grid
    x = 0.5 * (edges[:-1] + edges[1:])
    return x, edges, dx


def make_nested_grid(out_left, out_right, left, right, n_grid):
    dx = (right - left) / n_grid

    n_left = max(0, int(np.ceil((left - out_left) / dx)))
    n_right = max(0, int(np.ceil((out_right - right) / dx)))

    adjusted_out_left = left - n_left * dx
    adjusted_out_right = right + n_right * dx

    total_cells = n_left + n_grid + n_right

    edges = np.linspace(adjusted_out_left, adjusted_out_right, total_cells + 1)
    x = 0.5 * (edges[:-1] + edges[1:])

    return x, edges, dx



def import_custom_potential(module_name="custom_potential", func_name="custom"):
    cwd = os.getcwd()
    file_path = os.path.join(cwd, f"{module_name}.py")

    spec = importlib.util.spec_from_file_location(module_name, file_path)
    user_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(user_module)

    return getattr(user_module, func_name)

