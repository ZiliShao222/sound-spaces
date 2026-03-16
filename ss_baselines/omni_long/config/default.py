#!/usr/bin/env python3

from typing import List, Optional, Union
import logging
import os
import shutil

import habitat
from habitat.config import Config as CN


CONFIG_FILE_SEPARATOR = ","


_C = CN()
_C.SEED = 0
_C.BASE_TASK_CONFIG_PATH = "configs/omni-long/mp3d/omni-long_semantic_audio_ppo.yaml"
_C.TASK_CONFIG = CN()
_C.CMD_TRAILING_OPTS = []
_C.TRAINER_NAME = "OmniLongPPOTrainer"
_C.ENV_NAME = "OmniLongRLEnv"
_C.SIMULATOR_GPU_ID = 0
_C.TORCH_GPU_ID = 0
_C.MODEL_DIR = "data/models/omni_long"
_C.VIDEO_OPTION = []
_C.VISUALIZATION_OPTION = []
_C.TENSORBOARD_DIR = "tb"
_C.VIDEO_DIR = "video_dir"
_C.TEST_EPISODE_COUNT = 2
_C.EVAL_CKPT_PATH_DIR = "data/checkpoints"
_C.NUM_PROCESSES = 8
_C.SENSORS = ["RGB_SENSOR", "DEPTH_SENSOR"]
_C.CHECKPOINT_FOLDER = "data/checkpoints"
_C.NUM_UPDATES = 20000
_C.LOG_INTERVAL = 10
_C.LOG_FILE = "train.log"
_C.CHECKPOINT_INTERVAL = 50
_C.USE_VECENV = True
_C.USE_SYNC_VECENV = False
_C.DEBUG = False
_C.USE_LAST_CKPT = False
_C.DISPLAY_RESOLUTION = 128
_C.CONTINUOUS = True
_C.FOLLOW_SHORTEST_PATH = False

_C.WANDB = CN()
_C.WANDB.ENABLED = False
_C.WANDB.PROJECT = "omni-long"
_C.WANDB.ENTITY = ""
_C.WANDB.NAME = ""
_C.WANDB.GROUP = ""
_C.WANDB.JOB_TYPE = "train"
_C.WANDB.NOTES = ""
_C.WANDB.TAGS = []
_C.WANDB.MODE = "online"
_C.WANDB.DIR = ""
_C.WANDB.RESUME = "allow"
_C.WANDB.SYNC_TENSORBOARD = True

_C.EVAL = CN()
_C.EVAL.SPLIT = "val"
_C.EVAL.USE_CKPT_CONFIG = True

_C.RL = CN()
_C.RL.SUCCESS_REWARD = 10.0
_C.RL.SLACK_REWARD = -0.01
_C.RL.ORDERED_SUCCESS_REWARD = 10.0
_C.RL.UNORDERED_SUCCESS_REWARD = 10.0
_C.RL.ORDERED_SLACK_REWARD = -0.01
_C.RL.UNORDERED_SLACK_REWARD = -0.01
_C.RL.DYNAMIC_TIME_PENALTY = True
_C.RL.TIME_PENALTY_DECAY = 0.75
_C.RL.TIME_PENALTY_MIN_SCALE = 0.25
_C.RL.ORDERED_TIME_PENALTY_DECAY = 0.75
_C.RL.UNORDERED_TIME_PENALTY_DECAY = 0.75
_C.RL.ORDERED_TIME_PENALTY_MIN_SCALE = 0.25
_C.RL.UNORDERED_TIME_PENALTY_MIN_SCALE = 0.25
_C.RL.WITH_TIME_PENALTY = True
_C.RL.WITH_DISTANCE_REWARD = True
_C.RL.DISTANCE_REWARD_SCALE = 1.0
_C.RL.REWARD_MEASURE = "omni_long_distance_to_goal_reward"
_C.RL.SUCCESS_MEASURE = "lifelong_task_success"
_C.RL.END_ON_SUCCESS = True

_C.RL.PPO = CN()
_C.RL.PPO.clip_param = 0.2
_C.RL.PPO.ppo_epoch = 4
_C.RL.PPO.num_mini_batch = 1
_C.RL.PPO.value_loss_coef = 0.5
_C.RL.PPO.entropy_coef = 0.01
_C.RL.PPO.lr = 2.5e-4
_C.RL.PPO.eps = 1e-5
_C.RL.PPO.max_grad_norm = 0.5
_C.RL.PPO.num_steps = 64
_C.RL.PPO.hidden_size = 512
_C.RL.PPO.transformer_memory_size = 32
_C.RL.PPO.transformer_nhead = 8
_C.RL.PPO.transformer_num_layers = 2
_C.RL.PPO.transformer_dropout = 0.1
_C.RL.PPO.transformer_dim_feedforward = 1024
_C.RL.PPO.use_gae = True
_C.RL.PPO.use_linear_lr_decay = True
_C.RL.PPO.use_linear_clip_decay = False
_C.RL.PPO.gamma = 0.99
_C.RL.PPO.tau = 0.95
_C.RL.PPO.reward_window_size = 50
_C.RL.PPO.clip_model = "RN50"
_C.RL.PPO.use_clip_rgb = True
_C.RL.PPO.audio_sensor_uuid = "spectrogram"
_C.RL.PPO.goal_embedding_size = 512
_C.RL.PPO.depth_embedding_size = 128
_C.RL.PPO.audio_embedding_size = 128
_C.RL.PPO.pointgoal_embedding_size = 32
_C.RL.PPO.gps_embedding_size = 32
_C.RL.PPO.compass_embedding_size = 32
_C.RL.PPO.prev_action_embedding_size = 32
_C.RL.PPO.use_depth_encoder = True
_C.RL.PPO.use_audio_encoder = True
_C.RL.PPO.goal_attention_debug = False
_C.RL.PPO.goal_attention_debug_interval = 200
_C.RL.PPO.goal_attention_debug_max_logs = 20


_TC = habitat.get_config()
_TC.defrost()

_TC.TASK.AUDIOGOAL_SENSOR = CN()
_TC.TASK.AUDIOGOAL_SENSOR.TYPE = "AudioGoalSensor"

_TC.TASK.SPECTROGRAM_SENSOR = CN()
_TC.TASK.SPECTROGRAM_SENSOR.TYPE = "SpectrogramSensor"

_TC.TASK.OMNI_LONG_GOAL_SENSOR = CN()
_TC.TASK.OMNI_LONG_GOAL_SENSOR.TYPE = "OmniLongGoalSensor"
_TC.TASK.OMNI_LONG_GOAL_SENSOR.CLIP_MODEL = "RN50"
_TC.TASK.OMNI_LONG_GOAL_SENSOR.IMAGE_ROOT = "output"
_TC.TASK.OMNI_LONG_GOAL_SENSOR.MAX_GOALS = 5

_TC.TASK.GOAL_ORDER_MODE = "ordered"
_TC.TASK.UNORDERED_TARGET_LOCK_ENABLED = True
_TC.TASK.UNORDERED_TARGET_LOCK_ACQUIRE_PROGRESS = 1.0
_TC.TASK.UNORDERED_TARGET_LOCK_ALLOW_RELEASE = True
_TC.TASK.UNORDERED_TARGET_LOCK_RELEASE_DISTANCE_DELTA = 2.0

_TC.TASK.ACTIONS.LIFELONG_SUBMIT = CN()
_TC.TASK.ACTIONS.LIFELONG_SUBMIT.TYPE = "LifelongSubmitAction"

_TC.TASK.SUCCESS.TYPE = "OmniLongSuccess"
_TC.TASK.SPL.TYPE = "OmniLongSPL"
_TC.TASK.SOFT_SPL.TYPE = "OmniLongSoftSPL"

_TC.TASK.LIFELONG_GOALS_FOUND = CN()
_TC.TASK.LIFELONG_GOALS_FOUND.TYPE = "LifelongGoalsFound"

_TC.TASK.LIFELONG_GOAL_COMPLETION = CN()
_TC.TASK.LIFELONG_GOAL_COMPLETION.TYPE = "LifelongGoalCompletion"

_TC.TASK.LIFELONG_TASK_SUCCESS = CN()
_TC.TASK.LIFELONG_TASK_SUCCESS.TYPE = "LifelongTaskSuccess"

_TC.TASK.OMNI_LONG_DISTANCE_TO_GOAL_REWARD = CN()
_TC.TASK.OMNI_LONG_DISTANCE_TO_GOAL_REWARD.TYPE = "OmniLongDistanceToGoalReward"
_TC.TASK.OMNI_LONG_DISTANCE_TO_GOAL_REWARD.DISTANCE_REWARD_SCALE = 1.0
_TC.TASK.OMNI_LONG_DISTANCE_TO_GOAL_REWARD.SUBMIT_SUCCESS_REWARD = 10.0
_TC.TASK.OMNI_LONG_DISTANCE_TO_GOAL_REWARD.SUBMIT_SUCCESS_REWARD_INCREMENT = 0.0
_TC.TASK.OMNI_LONG_DISTANCE_TO_GOAL_REWARD.ORDERED_DISTANCE_REWARD_SCALE = 1.0
_TC.TASK.OMNI_LONG_DISTANCE_TO_GOAL_REWARD.UNORDERED_DISTANCE_REWARD_SCALE = 1.0
_TC.TASK.OMNI_LONG_DISTANCE_TO_GOAL_REWARD.ORDERED_SUBMIT_SUCCESS_REWARD = 10.0
_TC.TASK.OMNI_LONG_DISTANCE_TO_GOAL_REWARD.UNORDERED_SUBMIT_SUCCESS_REWARD = 10.0
_TC.TASK.OMNI_LONG_DISTANCE_TO_GOAL_REWARD.ORDERED_SUBMIT_SUCCESS_REWARD_INCREMENT = 5.0
_TC.TASK.OMNI_LONG_DISTANCE_TO_GOAL_REWARD.UNORDERED_SUBMIT_SUCCESS_REWARD_INCREMENT = 0.0
_TC.TASK.OMNI_LONG_DISTANCE_TO_GOAL_REWARD.ORDERED_INACTIVE_GOAL_PENALTY = 0.0
_TC.TASK.OMNI_LONG_DISTANCE_TO_GOAL_REWARD.UNORDERED_DENSE_REWARD_MODE = "global_path_reduction"

_TC.TASK.NUM_ACTION = CN()
_TC.TASK.NUM_ACTION.TYPE = "NA"

_TC.TASK.DISTANCE_TO_GOAL = CN()
_TC.TASK.DISTANCE_TO_GOAL.TYPE = "DistanceToGoal"
_TC.TASK.DISTANCE_TO_GOAL.DISTANCE_TO = "VIEW_POINTS"

_TC.SIMULATOR.GRID_SIZE = 1.0
_TC.SIMULATOR.CONTINUOUS_VIEW_CHANGE = False
_TC.SIMULATOR.VIEW_CHANGE_FPS = 10
_TC.SIMULATOR.SCENE_DATASET = "data/scene_datasets/mp3d/mp3d.scene_dataset_config.json"
_TC.SIMULATOR.USE_RENDERED_OBSERVATIONS = False
_TC.SIMULATOR.SCENE_OBSERVATION_DIR = "data/scene_observations"
_TC.SIMULATOR.STEP_TIME = 0.25
_TC.SIMULATOR.AUDIO = CN()
_TC.SIMULATOR.AUDIO.SCENE = ""
_TC.SIMULATOR.AUDIO.BINAURAL_RIR_DIR = "data/binaural_rirs"
_TC.SIMULATOR.AUDIO.RIR_SAMPLING_RATE = 16000
_TC.SIMULATOR.AUDIO.SOURCE_SOUND_DIR = "data/sounds/semantic_splits"
_TC.SIMULATOR.AUDIO.METADATA_DIR = "data/metadata"
_TC.SIMULATOR.AUDIO.POINTS_FILE = "points.txt"
_TC.SIMULATOR.AUDIO.GRAPH_FILE = "graph.pkl"
_TC.SIMULATOR.AUDIO.HAS_DISTRACTOR_SOUND = False
_TC.SIMULATOR.AUDIO.EVERLASTING = True
_TC.SIMULATOR.AUDIO.CROSSFADE = True

_TC.DATASET.VERSION = "v1"
_TC.DATASET.CONTINUOUS = True


def merge_from_path(config, config_paths):
    if config_paths:
        if isinstance(config_paths, str):
            if CONFIG_FILE_SEPARATOR in config_paths:
                config_paths = config_paths.split(CONFIG_FILE_SEPARATOR)
            else:
                config_paths = [config_paths]

        for config_path in config_paths:
            config.merge_from_file(config_path)
    return config


def get_config(
    config_paths: Optional[Union[List[str], str]] = None,
    opts: Optional[list] = None,
    model_dir: Optional[str] = None,
    run_type: Optional[str] = None,
    overwrite: bool = False,
) -> CN:
    config = merge_from_path(_C.clone(), config_paths)
    config.TASK_CONFIG = get_task_config(config_paths=config.BASE_TASK_CONFIG_PATH)

    if model_dir is not None:
        config.MODEL_DIR = model_dir

    config.TENSORBOARD_DIR = os.path.join(config.MODEL_DIR, "tb")
    config.CHECKPOINT_FOLDER = os.path.join(config.MODEL_DIR, "data")
    config.VIDEO_DIR = os.path.join(config.MODEL_DIR, "video_dir")
    config.LOG_FILE = os.path.join(config.MODEL_DIR, "train.log")
    config.EVAL_CKPT_PATH_DIR = os.path.join(config.MODEL_DIR, "data")

    if opts:
        config.CMD_TRAILING_OPTS = opts
        config.merge_from_list(opts)

    dirs = [config.VIDEO_DIR, config.TENSORBOARD_DIR, config.CHECKPOINT_FOLDER]
    if run_type == "train" and overwrite:
        for directory in dirs:
            if os.path.exists(directory):
                logging.warning("Removing existing directory: %s", directory)
                shutil.rmtree(directory)

    config.TASK_CONFIG.defrost()
    config.TASK_CONFIG.SIMULATOR.USE_SYNC_VECENV = config.USE_SYNC_VECENV
    if config.CONTINUOUS:
        config.TASK_CONFIG.SIMULATOR.TYPE = "ContinuousSoundSpacesSim"
        config.TASK_CONFIG.SIMULATOR.USE_RENDERED_OBSERVATIONS = False
        config.TASK_CONFIG.SIMULATOR.STEP_TIME = 0.25
        config.TASK_CONFIG.SIMULATOR.AUDIO.CROSSFADE = True
        config.TASK_CONFIG.DATASET.CONTINUOUS = True
    else:
        config.TASK_CONFIG.SIMULATOR.FORWARD_STEP_SIZE = config.TASK_CONFIG.SIMULATOR.GRID_SIZE
        config.TASK_CONFIG.DATASET.CONTINUOUS = False
    config.TASK_CONFIG.freeze()

    config.freeze()
    return config


def get_task_config(
    config_paths: Optional[Union[List[str], str]] = None,
    opts: Optional[list] = None,
) -> habitat.Config:
    config = _TC.clone()
    if config_paths:
        if isinstance(config_paths, str):
            if CONFIG_FILE_SEPARATOR in config_paths:
                config_paths = config_paths.split(CONFIG_FILE_SEPARATOR)
            else:
                config_paths = [config_paths]

        for config_path in config_paths:
            config.merge_from_file(config_path)

    if opts:
        config.merge_from_list(opts)

    config.freeze()
    return config
