from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


EXPERIMENTS = [
    (
        "FedAvg-IID",
        "config/ehealth/exp_heart_iid_fedavg.yaml",
        "config/ehealth/fedavg_heart.yaml",
    ),
    (
        "FedProx-IID",
        "config/ehealth/exp_heart_iid_fedprox.yaml",
        "config/ehealth/fedprox_heart.yaml",
    ),
    (
        "DPFedAVG-IID",
        "config/ehealth/exp_heart_iid_dpfedavg.yaml",
        "config/ehealth/dpfedavg_heart.yaml",
    ),
]


def run_one(name: str, exp_cfg: str, alg_cfg: str) -> None:
    cmd = ["fluke", "federation", exp_cfg, alg_cfg]
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    print(f"\n=== Running {name} ===")
    print(" ".join(cmd))
    proc = subprocess.run(cmd, cwd=ROOT, env=env)
    if proc.returncode != 0:
        raise RuntimeError(f"{name} failed with exit code {proc.returncode}")


def main() -> int:
    for name, exp_cfg, alg_cfg in EXPERIMENTS:
        run_one(name, exp_cfg, alg_cfg)
    print("\nAll Fluke HFL IID experiments completed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
