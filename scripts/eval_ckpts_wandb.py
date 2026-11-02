"""Score the saved checkpoints of a training run on a fixed benchmark and log the curve to W&B (paper Fig. 3a, 4a, A.1).

  uv run python scripts/eval_ckpts_wandb.py --run-dir logs/lafan1_paper_sky --npz-dir data/npz/lafan1
  uv run python scripts/eval_ckpts_wandb.py --run-dir logs/e11c_repro_sky --library data/library/bones_seed_compact_paper.pt --num-envs 8192 --watch

Benchmark protocol as in scripts/eval_benchmark.py (every clip once from its first frame, deterministic policy, no noise
or DR). One env is built and reused for all checkpoints. Per-checkpoint results go to <run-dir>/eval_<set>/ckpt_*.json
and are reused on restart. The W&B run (<train run>_eval) resumes from <run-dir>/eval_<set>/wandb_run.json.

x axes: train/iteration, and train/hours = the run's TensorBoard time/hours at that iteration (training-loop wall-clock
since the loop started, setup excluded, logging and earlier checkpoint saves included).
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import asdict
from pathlib import Path

import torch
import tyro

from supermimic.deploy.waist_coupling import resolve_coupling
from supermimic.deploy.waist_roll import resolve_range as resolve_waist_roll_range
from supermimic.envs.mdp.motion_library import discover_npz
from supermimic.eval.benchmark import (
    load_policy_from_checkpoint,
    make_benchmark_env,
    run_benchmark_on_env,
    summarize,
)
from supermimic.training.run_logger import git_info, hardware_info, init_wandb_online

CKPT_RE = re.compile(r"ckpt_(\d{7})\.pt$")
ENV_KEYS = ("future_frames", "anchor_obs", "lock_waist", "lock_waist_refs")


def list_ckpts(run_dir: Path, every: int) -> list[tuple[int, Path]]:
    out = [(int(m.group(1)), p) for p in run_dir.glob("ckpt_*.pt") if (m := CKPT_RE.search(p.name))]
    return sorted((it, p) for it, p in out if it % every == 0)


def train_hours(run_dir: Path) -> dict[int, float]:
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    acc = EventAccumulator(str(run_dir / "tb"), size_guidance={"scalars": 0})
    acc.Reload()
    if "time/hours" not in acc.Tags()["scalars"]:
        return {}
    return {e.step: e.value for e in acc.Scalars("time/hours")}


def train_wandb_id(run_dir: Path) -> str | None:
    latest = run_dir / "wandb" / "latest-run"
    return latest.resolve().name.split("-")[-1] if latest.exists() else None


def wandb_row(it: int, hours: float | None, s: dict, eval_s: float) -> dict:
    row = {"train/iteration": it, "time/eval_s": eval_s}
    if hours is not None:
        row["train/hours"] = hours
    for k, v in s.items():
        if k == "terms":
            row.update({f"eval/terms/{t}": x for t, x in v.items()})
        else:
            row[f"eval/{k}"] = v
    row["eval/failed_frac"] = s["failed"] / max(s["clips"], 1)
    return row


def main(run_dir: Path, npz_dir: Path = Path("data/npz/lafan1"), library: Path | None = None, clips: list[str] | None = None,
         set_name: str | None = None, num_envs: int = 1024, device: str = "cuda:0", every: int = 4000,
         watch: bool = False, poll_s: float = 300.0, idle_hours: float = 3.0, min_ckpt_age_s: float = 60.0,
         wandb_project: str = "supermimic", wandb_entity: str | None = None) -> None:
    """every: score checkpoints whose iteration is a multiple of this. watch: wait for new checkpoints until the
    run's last iteration is scored or no new checkpoint arrives for idle_hours."""
    files = [str(library)] if library is not None else [str(f) for f in discover_npz(npz_dir, names=clips)]
    set_name = set_name or (library.stem if library is not None else npz_dir.name) + ("_subset" if clips else "")
    out_dir = run_dir / f"eval_{set_name}"
    out_dir.mkdir(exist_ok=True)
    state_path = out_dir / "wandb_run.json"
    state = json.loads(state_path.read_text()) if state_path.exists() else {"id": None, "logged": []}
    logged = set(state["logged"])

    t_wait = time.time()
    while not (ck := list_ckpts(run_dir, every)):
        if not watch or time.time() - t_wait > idle_hours * 3600:
            raise SystemExit(f"no ckpt_*.pt in {run_dir} with iteration % {every} == 0")
        time.sleep(poll_s)
    first_cfg = torch.load(ck[0][1], map_location="cpu", weights_only=False)["config"]
    env_cfg = first_cfg["env"]
    total_iterations = int(first_cfg["schedule"]["total_iterations"])
    lock = env_cfg.get("lock_waist", False)
    refs = env_cfg.get("lock_waist_refs", "feasible")
    wc = resolve_coupling(env_cfg)
    wr = resolve_waist_roll_range(env_cfg)

    t0 = time.perf_counter()
    env = make_benchmark_env(files, device, num_envs, env_cfg["future_frames"], env_cfg.get("anchor_obs", False), lock, refs,
                             waist_coupling=wc, waist_roll_range=wr)
    env_build_s = time.perf_counter() - t0
    lib = env.command_manager.get_term("motion").library
    print(f"[INFO] benchmark {set_name}: {lib.num_clips} clips, {lib.num_frames} frames, {env.num_envs} envs, built in {env_build_s:.1f} s", flush=True)

    config = {
        "train_run_dir": str(run_dir), "train_run_name": first_cfg.get("run_name"), "train_wandb_id": train_wandb_id(run_dir),
        "train_total_iterations": total_iterations, "benchmark_set": set_name, "benchmark_files": files if library is not None else len(files),
        "benchmark_clips": lib.num_clips, "benchmark_frames": lib.num_frames, "num_envs": env.num_envs, "every": every,
        "lock_waist": lock, "lock_waist_refs": refs, "waist_coupling": wc, "waist_roll_range": wr,
        "protocol": "every clip once from frame 0, deterministic policy, no obs/action/reset noise, no DR, seed 0",
        "hours_definition": "TensorBoard time/hours of the training run at the checkpoint iteration",
        "hardware": hardware_info(device), "git": git_info(),
    }
    train_name = first_cfg.get("run_name") or run_dir.name
    run = init_wandb_online(project=wandb_project, entity=wandb_entity, name=f"{train_name}_eval", group=train_name, job_type="eval",
                            id=state["id"], resume="allow", config=config, dir=str(out_dir))
    state["id"] = run.id
    state_path.write_text(json.dumps(state))
    run.define_metric("train/iteration")
    run.define_metric("train/hours")
    run.define_metric("eval/*", step_metric="train/hours")
    run.define_metric("time/eval_s", step_metric="train/iteration")
    run.summary["time/env_build_s"] = env_build_s

    hours: dict[int, float] = {}
    last_new = time.time()
    try:
        while True:
            for it, path in list_ckpts(run_dir, every):
                if it in logged:
                    continue
                if time.time() - path.stat().st_mtime < min_ckpt_age_s:
                    break
                if it not in hours:
                    hours = train_hours(run_dir)
                if it not in hours and time.time() - path.stat().st_mtime < 900:
                    break  # TensorBoard flushes every 2 min, wait for this iteration's time/hours
                res_path = out_dir / f"{path.stem}.json"
                if res_path.exists():
                    rec = json.loads(res_path.read_text())
                else:
                    t0 = time.perf_counter()
                    policy, cfg = load_policy_from_checkpoint(path, device)
                    if any(cfg["env"].get(k) != env_cfg.get(k) for k in ENV_KEYS):
                        raise RuntimeError(f"{path.name}: env config differs from the first checkpoint")
                    results = run_benchmark_on_env(env, policy, verbose=False)
                    torch.cuda.synchronize(device)
                    eval_s = time.perf_counter() - t0
                    rec = {"iteration": it, "eval_s": eval_s, "summary": summarize(results), "clips": [asdict(r) for r in results]}
                    res_path.write_text(json.dumps(rec, indent=1))
                    del policy
                s = rec["summary"]
                h = hours.get(it)
                run.log(wandb_row(it, h, s, rec["eval_s"]))
                logged.add(it)
                state["logged"] = sorted(logged)
                state_path.write_text(json.dumps(state))
                last_new = time.time()
                hs = f"{h:.2f} h" if h is not None else "? h"
                print(f"it {it} | {hs} | mean norm return {s['mean_norm_return']:.4f} | step-weighted {s['step_weighted_norm_return']:.4f} | "
                      f"failed {s['failed']}/{s['clips']} | eval {rec['eval_s']:.0f} s", flush=True)
            done = bool(logged) and max(logged) >= total_iterations
            if not watch or done or time.time() - last_new > idle_hours * 3600:
                break
            time.sleep(poll_s)
    finally:
        env.close()
        run.finish()
    if logged:
        print(f"[INFO] scored {len(logged)} checkpoints, last iteration {max(logged)}")


if __name__ == "__main__":
    tyro.cli(main)
