import json
from pathlib import Path
import os
from typing import Any, Dict
from types import SimpleNamespace
import textwrap

current_dir = Path(os.getcwd())
config_file = (current_dir / "job_config.json").resolve()
custom_potential_file =  (current_dir / "custom_potential.py").resolve()

DEFAULT_CONFIG: Dict[str, Any] = {
    "method": "bounds", #"interval", "periodic", "mcgovern_interval"
    "metad_mode": "wt",
    "n_grid": 2000,
    "sigma": 0.15,
    "kBT": 1.0,
    "height": 1,
    "bias_factor": 10,
    "potential": "constant",# "none"
    "min": 0,
    "max": 1,
    "out_min": 0,
    "out_max": 1,
    "kernel_mode": "unit_integral", #"unit_peak"
    "total_steps": 1_000_000_000,
    "chunk_size":  1_000_000,
    "stride":  10_000,
    "plot_stride": 100_000_000,
    "base_dir":  "./",
    "filename": "simulation_results.h5",
    "hill_file": "./HILLS",
    "filter_width": 1.5,
    "peak_threshold": 1.0,
    "seed": 8456835
}

def create_potential_file(custom_potential_file):
    template = textwrap.dedent('''\
        import numpy as np
        
        def custom(x, left, right):
            """left and right correspond to the min and max parameters.
               x is defined on the interval [out_min, out_max]
               min, max, out_min,and  out_max parameters should be set up in the job_config.json"""
            return np.zeros_like(x)
    ''')

def create_default_settings():
    with open(config_file, "w", encoding="utf-8") as f:
        json.dump(DEFAULT_CONFIG, f, indent=4)
    with open(custom_potential_file, "w") as f:
        f.write(template)

def load_config() -> SimpleNamespace:
    with open(config_file, "r", encoding="utf-8") as f:
        config_dict = json.load(f)
        return SimpleNamespace(**config_dict)


