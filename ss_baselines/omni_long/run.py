#!/usr/bin/env python3

import argparse
import logging

import warnings

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

import torch

import soundspaces
from ss_baselines.common.baseline_registry import baseline_registry
from ss_baselines.omni_long.config.default import get_config
from ss_baselines.omni_long.ppo import OmniLongDDPPOTrainer, OmniLongPPOTrainer  # noqa: F401


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run-type",
        choices=["train", "eval"],
        default="train",
        help="run type of the experiment (train or eval)",
    )
    parser.add_argument(
        "--exp-config",
        type=str,
        default="ss_baselines/omni_long/config/omni_long/mp3d/ppo_spectrogram_pointgoal_rgb.yaml",
        help="path to config yaml containing info about experiment",
    )
    parser.add_argument(
        "opts",
        default=None,
        nargs=argparse.REMAINDER,
        help="Modify config options from command line",
    )
    parser.add_argument("--model-dir", default=None)
    parser.add_argument("--eval-interval", type=int, default=1)
    parser.add_argument("--overwrite", default=False, action="store_true")
    parser.add_argument("--prev-ckpt-ind", type=int, default=-1)
    args = parser.parse_args()

    config = get_config(
        args.exp_config,
        args.opts,
        args.model_dir,
        args.run_type,
        args.overwrite,
    )
    trainer_init = baseline_registry.get_trainer(config.TRAINER_NAME)
    assert trainer_init is not None, f"{config.TRAINER_NAME} is not supported"
    trainer = trainer_init(config)
    torch.set_num_threads(1)

    level = logging.DEBUG if config.DEBUG else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s, %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    if args.run_type == "train":
        trainer.train()
    else:
        trainer.eval(args.eval_interval, args.prev_ckpt_ind, config.USE_LAST_CKPT)


if __name__ == "__main__":
    main()
