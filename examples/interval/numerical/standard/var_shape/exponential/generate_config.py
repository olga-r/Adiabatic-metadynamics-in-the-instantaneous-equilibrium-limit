import json
from pathlib import Path
import os
from typing import Any, Dict
from types import SimpleNamespace
from random import randint
import shutil

DEFAULT_CONFIG: Dict[str, Any] = {
    "method": "interval", #"interval", "periodic", "mcgovern_interval"
    "metad_mode": "standard", #"wt"
    "n_grid": 500,
    "sigma": 0.05,
    "kBT": 1.0,
    "height": 0.001,
    "potential": "custom",# "constant","none"
    "func_name": "potential_exponential",
    "min": 0,
    "max": 1,
    "out_min": -0.5,
    "out_max": 1.5,
    "kernel_mode": "unit_peak", #"unit_peak"
    "total_steps": 200000,
    "chunk_size":  10000,
    "stride":  5,
    "plot_stride": 10000,
    "base_dir":  "./",
    "filename": "simulation_results.h5",
    "hill_file": "./HILLS",
    "filter_width": 1.5,
    "peak_threshold": 1.0,
}
randoms = [randint(0, 1000000000) for r in range(1, 16) ]
print(randoms)
for d in range(1, 11):
    os.mkdir("./{}".format(d))
    os.chdir("./{}".format(d))
    shutil.copyfile("../custom_potential.py", "./custom_potential.py")
    current_dir = Path(os.getcwd())
    config_file = (current_dir / "job_config.json").resolve()
    DEFAULT_CONFIG["seed"] =randoms[d-1]
    with open(config_file, "w", encoding="utf-8") as f:
        json.dump(DEFAULT_CONFIG, f, indent=4)
    os.chdir("../")


