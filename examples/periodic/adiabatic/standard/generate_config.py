import json
from pathlib import Path
import os
from typing import Any, Dict
from types import SimpleNamespace
from random import randint
import shutil

DEFAULT_CONFIG: Dict[str, Any] = {
    "method": "periodic", #"interval", "periodic", "mcgovern_interval"
    "metad_mode": "standard", #"wt"
    "periodic_tol": 1e-14,
    "n_grid": 250,
    "sigma": 0.06,
    "kBT": 1.0,
    "height": 0.001,
    "potential": "custom",# "none", "constant"
    "func_name": "periodic",
    "min": 0,
    "max": 1,
    "kernel_mode": "unit_peak", #"unit_peak"
    "total_steps": 2000000,
    "chunk_size":  1000000,
    "stride":  100,
    "plot_stride": 200000,
    "base_dir":  "./",
    "filename": "simulation_results.h5",
    "hill_file": "./HILLS",
    "filter_width": 1.5,
    "peak_threshold": 1.0,
}
randoms = [randint(0, 1000000000) for r in range(1, 16) ]
print(randoms)
for d in range(1, 16):
    os.mkdir("./{}".format(d))
    os.chdir("./{}".format(d))
    current_dir = Path(os.getcwd())
    config_file = (current_dir / "job_config.json").resolve()
    DEFAULT_CONFIG["seed"] =randoms[d-1]
    with open(config_file, "w", encoding="utf-8") as f:
        json.dump(DEFAULT_CONFIG, f, indent=4)
    shutil.copy("../custom_potential.py", "./custom_potential.py")
    os.chdir("../")


