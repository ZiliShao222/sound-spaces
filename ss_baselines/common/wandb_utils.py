#!/usr/bin/env python3

import os
from typing import Any, Dict, Optional

from habitat import logger


class WandbRun:
    def __init__(self, config, run_type: str = "train"):
        self.enabled = False
        self.run = None

        wandb_cfg = getattr(config, "WANDB", None)
        if wandb_cfg is None or not getattr(wandb_cfg, "ENABLED", False):
            return

        import wandb

        run_dir = str(getattr(wandb_cfg, "DIR", "") or os.path.dirname(config.TENSORBOARD_DIR))
        os.makedirs(run_dir, exist_ok=True)

        init_kwargs: Dict[str, Any] = {
            "project": str(getattr(wandb_cfg, "PROJECT", "sound-spaces")),
            "name": str(getattr(wandb_cfg, "NAME", "") or "") or None,
            "group": str(getattr(wandb_cfg, "GROUP", "") or "") or None,
            "job_type": str(getattr(wandb_cfg, "JOB_TYPE", run_type)),
            "notes": str(getattr(wandb_cfg, "NOTES", "") or "") or None,
            "tags": list(getattr(wandb_cfg, "TAGS", [])),
            "mode": str(getattr(wandb_cfg, "MODE", "online")),
            "dir": run_dir,
            "sync_tensorboard": bool(getattr(wandb_cfg, "SYNC_TENSORBOARD", True)),
            "resume": str(getattr(wandb_cfg, "RESUME", "allow")),
            "reinit": True,
        }

        entity = str(getattr(wandb_cfg, "ENTITY", "") or "")
        if entity:
            init_kwargs["entity"] = entity

        self.run = wandb.init(**init_kwargs)
        if self.run is None:
            logger.warning("Failed to initialize wandb; disabling Weights & Biases logging.")
            return

        config_snapshot: Dict[str, Any] = {
            "trainer_name": str(getattr(config, "TRAINER_NAME", "")),
            "env_name": str(getattr(config, "ENV_NAME", "")),
            "base_task_config_path": str(getattr(config, "BASE_TASK_CONFIG_PATH", "")),
            "num_processes": int(getattr(config, "NUM_PROCESSES", 0)),
            "num_updates": int(getattr(config, "NUM_UPDATES", 0)),
            "seed": int(getattr(config, "SEED", 0)),
            "tensorboard_dir": str(getattr(config, "TENSORBOARD_DIR", "")),
            "checkpoint_folder": str(getattr(config, "CHECKPOINT_FOLDER", "")),
            "run_type": run_type,
            "cmd_trailing_opts": list(getattr(config, "CMD_TRAILING_OPTS", [])),
            "config_yaml": config.dump(),
        }
        self.run.config.update(config_snapshot, allow_val_change=True)
        self.enabled = True
        logger.info(
            "Initialized wandb run `%s` in mode `%s`.",
            self.run.name,
            init_kwargs["mode"],
        )

    def log(self, payload: Dict[str, Any], step: Optional[int] = None) -> None:
        if not self.enabled or self.run is None or not payload:
            return
        self.run.log(payload, step=step)

    def finish(self) -> None:
        if not self.enabled or self.run is None:
            return
        self.run.finish()
        self.enabled = False
