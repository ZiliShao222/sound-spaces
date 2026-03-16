#!/usr/bin/env python3

from ss_baselines.common.base_trainer import BaseRLTrainer
from ss_baselines.common.baseline_registry import baseline_registry


_REMOVAL_MESSAGE = (
    "The previous omni-long PPO implementation has been removed. "
    "Rewrite `ss_baselines/omni_long/ppo/ppo_trainer.py` before using OmniLongPPOTrainer."
)


@baseline_registry.register_trainer(name="OmniLongPPOTrainer")
class OmniLongPPOTrainer(BaseRLTrainer):
    supported_tasks = ["OmniLongSemanticAudioNav"]

    def __init__(self, config=None):
        super().__init__(config)
        raise NotImplementedError(_REMOVAL_MESSAGE)

    def train(self) -> None:
        raise NotImplementedError(_REMOVAL_MESSAGE)

    def eval(self, eval_interval=1, prev_ckpt_ind=-1, use_last_ckpt=False) -> None:
        raise NotImplementedError(_REMOVAL_MESSAGE)

    def save_checkpoint(self, file_name) -> None:
        raise NotImplementedError(_REMOVAL_MESSAGE)

    def load_checkpoint(self, checkpoint_path, *args, **kwargs):
        raise NotImplementedError(_REMOVAL_MESSAGE)

    def _eval_checkpoint(self, checkpoint_path, writer, checkpoint_index: int = 0) -> None:
        raise NotImplementedError(_REMOVAL_MESSAGE)
