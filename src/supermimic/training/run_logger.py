"""TensorBoard + online Weights & Biases logging for production runs."""

from __future__ import annotations

import os
import platform
import socket
import subprocess
from pathlib import Path

import torch
from torch.utils.tensorboard import SummaryWriter


def _cmd(args: list[str], cwd: Path | None = None) -> str | None:
    try:
        return subprocess.run(args, cwd=cwd, capture_output=True, text=True, timeout=10, check=True).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None


def hardware_info(device: str) -> dict:
    idx = torch.device(device).index or 0
    return {
        "host": socket.gethostname(),
        "gpu": torch.cuda.get_device_name(idx),
        "gpu_count_visible": torch.cuda.device_count(),
        "gpu_memory_gb": round(torch.cuda.get_device_properties(idx).total_memory / 1e9, 1),
        "driver": _cmd(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader", f"--id={idx}"]),
        "cuda_runtime": torch.version.cuda,
        "torch": torch.__version__,
        "cpu_count_affinity": len(os.sched_getaffinity(0)),
        "python": platform.python_version(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_node": os.environ.get("SLURMD_NODENAME"),
    }


def git_info() -> dict:
    root = Path(__file__).resolve().parents[3]
    return {
        "commit": _cmd(["git", "rev-parse", "HEAD"], root),
        "branch": _cmd(["git", "rev-parse", "--abbrev-ref", "HEAD"], root),
        "dirty": bool(_cmd(["git", "status", "--porcelain", "--untracked-files=no"], root)),
    }


def init_wandb_online(**kwargs):
    import wandb

    # Production runs must log online; never fall back to offline or disabled.
    if os.environ.get("WANDB_MODE", "online") != "online":
        raise RuntimeError(f"WANDB_MODE={os.environ['WANDB_MODE']}: production runs need online W&B")
    run = wandb.init(mode="online", **kwargs)
    if run is None or getattr(run, "disabled", False) or getattr(run, "offline", False):
        raise RuntimeError("W&B did not start in online mode")
    print(f"[INFO] W&B run {run.url}", flush=True)
    return run


class RunLogger:
    """Same interface as SummaryWriter.add_scalar. Scalars of one step go to W&B in one call at commit()."""

    def __init__(self, log_dir: Path, run_name: str, config: dict, use_wandb: bool, project: str, entity: str | None):
        self.tb = SummaryWriter(str(log_dir / "tb"))
        self.run = None
        self._row: dict[str, float] = {}
        self._step: int | None = None
        if not use_wandb:
            print("[WARN] W&B is off (--no-wandb): smoke test only, not a production run", flush=True)
            return
        self.run = init_wandb_online(project=project, entity=entity, name=run_name, config=config, dir=str(log_dir))

    def add_scalar(self, key: str, value: float, step: int) -> None:
        self.tb.add_scalar(key, value, step)
        if self.run is None:
            return
        if self._step is not None and step != self._step:
            self.commit()
        self._step = step
        self._row[key] = float(value)

    def commit(self) -> None:
        if self.run is not None and self._row:
            self.run.log(self._row, step=self._step)
        self._row, self._step = {}, None

    def summary(self, values: dict) -> None:
        if self.run is not None:
            self.run.summary.update(values)

    def flush(self) -> None:
        self.commit()
        self.tb.flush()

    def finish(self) -> None:
        self.flush()
        if self.run is not None:
            self.run.finish()
