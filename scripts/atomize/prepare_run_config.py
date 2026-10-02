"""Write a per-job training config that resumes from the newest checkpoint.

Used by train_h100.sbatch so that a preempted / requeued / chained job picks up
where it left off instead of restarting from the cc89 init.

Checkpoint discovery is done here rather than with a shell `ls | sed | sort`
pipeline: `ls` may be aliased to long format, and `epoch10` must sort after
`epoch9` numerically rather than lexically.

Usage:
    python prepare_run_config.py <base_cfg> <out_cfg> <checkpoint_dir>
"""

from __future__ import annotations

import glob
import os
import re
import sys

import yaml


def newest_checkpoint(ckdir: str) -> tuple[str | None, int]:
    best, best_ep = None, -1
    for path in glob.glob(os.path.join(ckdir, "epoch*_training_state.pth")):
        m = re.search(r"epoch(\d+)_training_state\.pth$", os.path.basename(path))
        if m and int(m.group(1)) > best_ep:
            best, best_ep = path, int(m.group(1))
    return best, best_ep


def main() -> None:
    base_cfg, out_cfg, ckdir = sys.argv[1], sys.argv[2], sys.argv[3]
    with open(base_cfg, encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)

    latest, epoch = newest_checkpoint(ckdir)
    if latest:
        cfg["train"]["ckpt_path"] = latest
        print(f"RESUMING from epoch {epoch}: {latest}", flush=True)
    else:
        print(f"STARTING from {cfg['train']['ckpt_path']}", flush=True)

    os.makedirs(os.path.dirname(out_cfg), exist_ok=True)
    with open(out_cfg, "w", encoding="utf-8") as fh:
        yaml.safe_dump(cfg, fh)


if __name__ == "__main__":
    main()
