# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, Dict, List, Optional, Tuple
from abc import ABC
from collections import defaultdict
import logging
import os

import librosa
import math
import psutil
from scipy.signal import fftconvolve
import numpy as np
from PIL import Image
from gym import spaces

from habitat.core.registry import registry
import habitat_sim
from habitat_sim.utils.common import quat_from_angle_axis, quat_from_coeffs, quat_to_angle_axis
from habitat.sims.habitat_simulator.habitat_simulator import HabitatSimSensor, overwrite_config
from habitat.core.simulator import (
    AgentState,
    Config,
    Observations,
    SensorSuite,
    ShortestPathPoint,
    Simulator,
)
from pathlib import Path


def calculate_mem_usage():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024.0 / 1024 / 1024  # in GB


def _as_int_or_none(value: Any) -> Optional[int]:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, str):
        token = value.strip()
        if token.startswith(("+", "-")):
            sign = token[0]
            digits = token[1:]
            if digits.isdigit():
                return int(sign + digits)
        if token.isdigit():
            return int(token)
    return None


def crossfade(x1, x2, sr):
    crossfade_samples = int(0.05 * sr)  # 30 ms
    x2_weight = np.arange(crossfade_samples + 1) / crossfade_samples
    x1_weight = np.flip(x2_weight)
    x3 = [x1[:, :crossfade_samples+1] * x1_weight + x2[:, :crossfade_samples+1] * x2_weight, x2[:, crossfade_samples+1:]]

    return np.concatenate(x3, axis=1)


def _resolve_scene_dataset_config(scene_dataset: str) -> Optional[str]:
    if not scene_dataset:
        return None
    if os.path.isfile(scene_dataset):
        return scene_dataset
    # Common mp3d layout: data/scene_datasets/mp3d/mp3d.scene_dataset_config.json
    candidate = os.path.join(
        "data", "scene_datasets", scene_dataset, f"{scene_dataset}.scene_dataset_config.json"
    )
    if os.path.isfile(candidate):
        return candidate
    return None


def _quat_to_list(quat) -> List[float]:
    if all(hasattr(quat, attr) for attr in ("x", "y", "z", "w")):
        return [float(quat.x), float(quat.y), float(quat.z), float(quat.w)]
    if hasattr(quat, "vector") and hasattr(quat, "scalar"):
        v = quat.vector
        return [float(v[0]), float(v[1]), float(v[2]), float(quat.scalar)]
    if isinstance(quat, (list, tuple, np.ndarray)) and len(quat) == 4:
        return [float(x) for x in quat]
    return [0.0, 0.0, 0.0, 1.0]


@registry.register_simulator()
class ContinuousSoundSpacesSim(Simulator, ABC):
    r"""Continuous Habitat-Sim wrapper with optional online audio rendering.

    This simulator uses continuous agent poses and Habitat-Sim pathfinder queries.
    Legacy discrete graph metadata and precomputed observation modes are not supported.
    """

    def action_space_shortest_path(self, source: AgentState, targets: List[AgentState], agent_id: int = 0) -> List[
            ShortestPathPoint]:
        pass

    def __init__(self, config: Config) -> None:
        self.config = self.habitat_config = config
        self._validate_legacy_config()
        agent_config = self._get_agent_config()
        sim_sensors = []
        for sensor_name in agent_config.SENSORS:
            sensor_cfg = getattr(self.config, sensor_name)
            sensor_type = registry.get_sensor(sensor_cfg.TYPE)

            assert sensor_type is not None, "invalid sensor type {}".format(
                sensor_cfg.TYPE
            )
            sim_sensors.append(sensor_type(sensor_cfg))

        self._sensor_suite = SensorSuite(sim_sensors)
        self.sim_config = self.create_sim_config(self._sensor_suite)
        self._current_scene = self.sim_config.sim_cfg.scene_id
        self._action_space = spaces.Discrete(
            len(self.sim_config.agents[0].action_space)
        )
        self._prev_sim_obs = None

        self._current_sound = None
        self._sound_ids = None
        self._sound_positions = None
        self._active_sound_idx = None
        self._sound_schedule = None
        self._offset = None
        self._duration = None
        self._audio_index = None
        self._audio_length = None
        self._source_sound_dict = dict()
        self._sampling_rate = None
        self._episode_step_count = None
        self._is_episode_active = None
        self._previous_step_collided = False

        self._sim = habitat_sim.Simulator(config=self.sim_config)
        self.add_acoustic_config()
        self._last_rir = None
        self._current_sample_index = 0

    def _validate_legacy_config(self) -> None:
        if bool(getattr(self.config, "USE_RENDERED_OBSERVATIONS", False)):
            raise RuntimeError(
                "ContinuousSoundSpacesSim no longer supports USE_RENDERED_OBSERVATIONS. "
                "Use live Habitat-Sim rendering instead."
            )
        if bool(getattr(self.config.AUDIO, "HAS_DISTRACTOR_SOUND", False)):
            raise RuntimeError(
                "ContinuousSoundSpacesSim no longer supports legacy distractor-sound graph mode. "
                "Use SOUND_SOURCES with continuous positions instead."
            )

    def add_acoustic_config(self):
        if not self._audio_enabled():
            return
        audio_sensor_spec = habitat_sim.AudioSensorSpec()
        audio_sensor_spec.uuid = "audio_sensor"
        audio_sensor_spec.enableMaterials = False
        audio_sensor_spec.channelLayout.type = habitat_sim.sensor.RLRAudioPropagationChannelLayoutType.Binaural
        audio_sensor_spec.channelLayout.channelCount = 2
        audio_sensor_spec.acousticsConfig.sampleRate = self.config.AUDIO.RIR_SAMPLING_RATE
        audio_sensor_spec.acousticsConfig.threadCount = 1
        audio_sensor_spec.acousticsConfig.indirectRayCount = 500
        audio_sensor_spec.acousticsConfig.temporalCoherence = True
        audio_sensor_spec.acousticsConfig.transmission = True
        if bool(getattr(self.config.AUDIO, "DISABLE_REVERB", False)):
            audio_sensor_spec.acousticsConfig.indirectRayCount = 0
            audio_sensor_spec.acousticsConfig.temporalCoherence = False
            audio_sensor_spec.acousticsConfig.transmission = False
        self._sim.add_sensor(audio_sensor_spec)

    def _audio_enabled(self) -> bool:
        return bool(getattr(self.config.AUDIO, "ENABLED", True))

    def _audio_materials_path(self, scene_path: str) -> str:
        scene_lower = (scene_path or "").lower()
        if "hm3d" in scene_lower:
            hm3d_cfg = os.path.join("data", "hm3d_material_config.json")
            if os.path.isfile(hm3d_cfg):
                return hm3d_cfg
        return os.path.join("data", "mp3d_material_config.json")

    def create_sim_config(
        self, _sensor_suite: SensorSuite
    ) -> habitat_sim.Configuration:
        sim_config = habitat_sim.SimulatorConfiguration()
        # Check if Habitat-Sim is post Scene Config Update
        if not hasattr(sim_config, "scene_id"):
            raise RuntimeError(
                "Incompatible version of Habitat-Sim detected, please upgrade habitat_sim"
            )
        overwrite_config(
            config_from=self.config.HABITAT_SIM_V0,
            config_to=sim_config,
            # Ignore key as it gets propogated to sensor below
            ignore_keys={"gpu_gpu"},
        )
        scene_dataset_cfg = _resolve_scene_dataset_config(self.config.SCENE_DATASET)
        if scene_dataset_cfg is not None:
            sim_config.scene_dataset_config_file = scene_dataset_cfg
        sim_config.scene_id = self.config.SCENE
        sim_config.enable_physics = False
        # sim_config.scene_dataset_config_file = 'data/scene_datasets/mp3d/mp3d.scene_dataset_config.json'
        # sim_config.scene_dataset_config_file = 'data/scene_datasets/replica/replica.scene_dataset_config.json'
        agent_config = habitat_sim.AgentConfiguration()
        overwrite_config(
            config_from=self._get_agent_config(),
            config_to=agent_config,
            # These keys are only used by Hab-Lab
            ignore_keys={
                "is_set_start_state",
                # This is the Sensor Config. Unpacked below
                "sensors",
                "start_position",
                "start_rotation",
                "goal_position",
                "goal_positions",
                "offset",
                "duration",
                "sound_id",
                "sound_sources",
                "sound_source_schedule",
                "mass",
                "linear_acceleration",
                "angular_acceleration",
                "linear_friction",
                "angular_friction",
                "coefficient_of_restitution",
                "distractor_sound_id",
                "distractor_position_index"
            },
        )

        sensor_specifications = []
        for sensor in _sensor_suite.sensors.values():
            assert isinstance(sensor, HabitatSimSensor)
            sim_sensor_cfg = sensor._get_default_spec()  # type: ignore[misc]
            overwrite_config(
                config_from=sensor.config,
                config_to=sim_sensor_cfg,
                # These keys are only used by Hab-Lab
                # or translated into the sensor config manually
                ignore_keys=sensor._config_ignore_keys,
                # TODO consider making trans_dict a sensor class var too.
                trans_dict={
                    "sensor_model_type": lambda v: getattr(
                        habitat_sim.FisheyeSensorModelType, v
                    ),
                    "sensor_subtype": lambda v: getattr(
                        habitat_sim.SensorSubType, v
                    ),
                },
            )
            sim_sensor_cfg.uuid = sensor.uuid
            sim_sensor_cfg.resolution = list(
                sensor.observation_space.shape[:2]
            )

            # TODO(maksymets): Add configure method to Sensor API to avoid
            # accessing child attributes through parent interface
            # We know that the Sensor has to be one of these Sensors
            sim_sensor_cfg.sensor_type = sensor.sim_sensor_type
            sim_sensor_cfg.gpu2gpu_transfer = (
                self.config.HABITAT_SIM_V0.GPU_GPU
            )
            sensor_specifications.append(sim_sensor_cfg)

        agent_config.sensor_specifications = sensor_specifications
        agent_config.action_space = registry.get_action_space_configuration(
            self.config.ACTION_SPACE_CONFIG
        )(self.config).get()

        return habitat_sim.Configuration(sim_config, [agent_config])

    @property
    def sensor_suite(self) -> SensorSuite:
        return self._sensor_suite

    def _update_agents_state(self) -> bool:
        is_updated = False
        for agent_id, _ in enumerate(self.config.AGENTS):
            agent_cfg = self._get_agent_config(agent_id)
            if agent_cfg.IS_SET_START_STATE:
                self.set_agent_state(
                    agent_cfg.START_POSITION,
                    agent_cfg.START_ROTATION,
                    agent_id,
                )
                is_updated = True

        return is_updated

    def _get_agent_config(self, agent_id: Optional[int] = None) -> Any:
        if agent_id is None:
            agent_id = self.config.DEFAULT_AGENT_ID
        agent_name = self.config.AGENTS[agent_id]
        agent_config = getattr(self.config, agent_name)
        return agent_config

    def get_agent_state(self, agent_id: int = 0) -> habitat_sim.AgentState:
        return self._sim.get_agent(agent_id).get_state()

    def set_agent_state(
        self,
        position: List[float],
        rotation: List[float],
        agent_id: int = 0,
        reset_sensors: bool = True,
    ) -> bool:
        agent = self._sim.get_agent(agent_id)
        new_state = self.get_agent_state(agent_id)
        new_state.position = position
        new_state.rotation = rotation

        # NB: The agent state also contains the sensor states in _absolute_
        # coordinates. In order to set the agent's body to a specific
        # location and have the sensors follow, we must not provide any
        # state for the sensors. This will cause them to follow the agent's
        # body
        new_state.sensor_states = {}
        agent.set_state(new_state, reset_sensors)
        return True

    @property
    def source_sound_dir(self):
        return self.config.AUDIO.SOURCE_SOUND_DIR

    @property
    def current_scene_name(self):
        scene_path = Path(self._current_scene)
        parts = scene_path.parts
        if "hm3d" in parts:
            idx = parts.index("hm3d")
            if len(parts) > idx + 2:
                return parts[idx + 2]
        if "mp3d" in parts:
            idx = parts.index("mp3d")
            if len(parts) > idx + 1:
                candidate = parts[idx + 1]
                if candidate in ("train", "val", "test") and len(parts) > idx + 2:
                    return parts[idx + 2]
                return candidate
        if "replica" in parts:
            idx = parts.index("replica")
            if len(parts) > idx + 1:
                return parts[idx + 1]
        return scene_path.stem

    def _scene_dataset_name(self) -> str:
        scene_parts = Path(self._current_scene).parts
        for name in ("hm3d", "mp3d", "replica"):
            if name in scene_parts:
                return name
        ds = self.config.SCENE_DATASET
        if ds and os.path.isfile(ds):
            ds_lower = ds.lower()
            for name in ("hm3d", "mp3d", "replica"):
                if name in ds_lower:
                    return name
            return Path(ds).stem
        return str(ds)

    @property
    def current_scene_observation_file(self):
        return os.path.join(self.config.SCENE_OBSERVATION_DIR, self.config.SCENE_DATASET,
                            self.current_scene_name + '.pkl')

    @property
    def current_source_sound(self):
        return self._source_sound_dict[self._current_sound]

    @property
    def is_silent(self):
        return self._episode_step_count > self._duration

    @property
    def pathfinder(self):
        return self._sim.pathfinder

    def get_agent(self, agent_id):
        return self._sim.get_agent(agent_id)

    def reconfigure(self, config: Config) -> None:
        self.config = config
        self._validate_legacy_config()
        if hasattr(self.config.AGENT_0, 'OFFSET'):
            self._offset = int(self.config.AGENT_0.OFFSET)
        else:
            self._offset = 0
        if self.config.AUDIO.EVERLASTING:
            self._duration = 500
        else:
            assert hasattr(self.config.AGENT_0, 'DURATION')
            self._duration = int(self.config.AGENT_0.DURATION)
        self._audio_index = 0
        sound_sources = getattr(self.config.AGENT_0, "SOUND_SOURCES", None)
        if sound_sources:
            self._sound_ids = []
            self._sound_positions = []
            for src in sound_sources:
                self._sound_ids.append(src["sound_id"])
                self._sound_positions.append(src["position"])
            self._sound_schedule = getattr(
                self.config.AGENT_0, "SOUND_SOURCE_SCHEDULE", ["round_robin", 25]
            )
        else:
            self._sound_ids = [self.config.AGENT_0.SOUND_ID]
            self._sound_positions = [self.config.AGENT_0.GOAL_POSITION]
            self._sound_schedule = None

        self._active_sound_idx = 0
        self._current_sound = self._sound_ids[0]
        for sid in self._sound_ids:
            self._load_single_source_sound(sid)
        logging.debug(
            "Switch to sound(s) {} with duration {} seconds".format(
                self._sound_ids, self._duration
            )
        )

        is_same_scene = config.SCENE == self._current_scene
        if not is_same_scene:
            self._current_scene = config.SCENE
            logging.debug('Current scene: {} and sound: {}'.format(self.current_scene_name, self._current_sound))

            self._sim.close()
            del self._sim
            self.sim_config = self.create_sim_config(self._sensor_suite)
            self._sim = habitat_sim.Simulator(self.sim_config)
            self.add_acoustic_config()
            if self._audio_enabled():
                audio_sensor = self._sim.get_agent(0)._sensors["audio_sensor"]
                audio_sensor.setAudioMaterialsJSON(
                    self._audio_materials_path(self._current_scene)
                )
            logging.debug('Loaded scene {}'.format(self.current_scene_name))

        self._update_agents_state()
        if self._audio_enabled():
            audio_sensor = self._sim.get_agent(0)._sensors["audio_sensor"]
            # 1.5 is the offset for the height
            audio_sensor.setAudioSourceTransform(
                np.array(self._sound_positions[self._active_sound_idx]) + np.array([0, 1.5, 0])
            )
        self._episode_step_count = 0
        self._last_rir = None
        self._current_sample_index = np.random.randint(self.config.AUDIO.RIR_SAMPLING_RATE * self.config.STEP_TIME)

    def _snap_to_navmesh(self, position):
        if hasattr(self.pathfinder, "snap_point"):
            return np.array(self.pathfinder.snap_point(position), dtype=np.float32)
        return np.array(position, dtype=np.float32)

    def reset(self):
        logging.debug('Reset simulation')
        sim_obs = self._sim.reset()
        if self._update_agents_state():
            sim_obs = self._sim.get_sensor_observations()

        self._is_episode_active = True
        self._prev_sim_obs = sim_obs
        self._previous_step_collided = False
        # Encapsule data under Observations class
        observations = self._sensor_suite.get_observations(sim_obs)

        return observations

    def step(self, action):
        """
        All angle calculations in this function is w.r.t habitat coordinate frame, on X-Z plane
        where +Y is upward, -Z is forward and +X is rightward.
        Angle 0 corresponds to +X, angle 90 corresponds to +y and 290 corresponds to 270.

        :param action: action to be taken
        :return:
        Dict of observations
        """
        assert self._is_episode_active, (
            "episode is not active, environment not RESET or "
            "STOP action called previously"
        )
        self._maybe_switch_active_sound()
        if self._audio_enabled():
            self._last_rir = np.transpose(np.array(self._prev_sim_obs["audio_sensor"]))
        sim_obs = self._sim.step(action)
        self._prev_sim_obs = sim_obs
        observations = self._sensor_suite.get_observations(sim_obs)
        self._episode_step_count += 1
        self._current_sample_index = int(
            self._current_sample_index
            + self.config.AUDIO.RIR_SAMPLING_RATE * self.config.STEP_TIME
        ) % self.current_source_sound.shape[0]

        return observations

    def _load_source_sounds(self):
        # load all mono files at once
        sound_files = os.listdir(self.source_sound_dir)
        for sound_file in sound_files:
            sound = sound_file.split('.')[0]
            audio_data, sr = librosa.load(os.path.join(self.source_sound_dir, sound),
                                          sr=self.config.AUDIO.RIR_SAMPLING_RATE)
            self._source_sound_dict[sound] = audio_data
            self._audio_length = audio_data.shape[0] // self.config.AUDIO.RIR_SAMPLING_RATE

    def _load_single_source_sound(self, sound_id: str):
        if sound_id not in self._source_sound_dict:
            audio_data, sr = librosa.load(os.path.join(self.source_sound_dir, sound_id),
                                          sr=self.config.AUDIO.RIR_SAMPLING_RATE)
            if audio_data.shape[0]//self.config.AUDIO.RIR_SAMPLING_RATE == 1:
                audio_data = np.concatenate([audio_data] * 3, axis=0)  # duplicate to be longer than longest RIR
            self._source_sound_dict[sound_id] = audio_data
        self._audio_length = self._source_sound_dict[sound_id].shape[0]//self.config.AUDIO.RIR_SAMPLING_RATE

    def _compute_audiogoal(self):
        sampling_rate = self.config.AUDIO.RIR_SAMPLING_RATE
        if not self._audio_enabled():
            return np.zeros((2, sampling_rate))
        if self._episode_step_count > self._duration:
            logging.debug('Step count is greater than duration. Empty spectrogram.')
            audiogoal = np.zeros((2, sampling_rate))
        else:
            binaural_rir = np.transpose(np.array(self._prev_sim_obs["audio_sensor"]))
            audiogoal = self._convolve_with_rir(binaural_rir)

            if self.config.AUDIO.CROSSFADE and self._last_rir is not None:
                audiogoal_from_last_rir = self._convolve_with_rir(self._last_rir)
                audiogoal = crossfade(audiogoal_from_last_rir, audiogoal, sampling_rate)

        return audiogoal

    def _maybe_switch_active_sound(self):
        if not self._audio_enabled():
            return
        if not self._sound_ids or len(self._sound_ids) == 1:
            return
        interval = 25
        if self._sound_schedule and isinstance(self._sound_schedule, list):
            if len(self._sound_schedule) > 1:
                interval = int(self._sound_schedule[1])
        if interval <= 0:
            interval = 1
        idx = (self._episode_step_count // interval) % len(self._sound_ids)
        if idx == self._active_sound_idx:
            return
        self._active_sound_idx = idx
        self._current_sound = self._sound_ids[idx]
        self._current_sample_index = 0
        # print("interval in use:", interval)

        audio_sensor = self._sim.get_agent(0)._sensors["audio_sensor"]
        audio_sensor.setAudioSourceTransform(
            np.array(self._sound_positions[idx]) + np.array([0, 1.5, 0])
        )

    def _convolve_with_rir(self, rir):
        sampling_rate = self.config.AUDIO.RIR_SAMPLING_RATE
        num_sample = int(sampling_rate * self.config.STEP_TIME)

        index = self._current_sample_index
        if index - rir.shape[0] < 0:
            sound_segment = self.current_source_sound[: index + num_sample]
            binaural_convolved = np.array([fftconvolve(sound_segment, rir[:, channel]
                                                       ) for channel in range(rir.shape[-1])])
            audiogoal = binaural_convolved[:, index: index + num_sample]
        else:
            # include reverb from previous time step
            if index + num_sample < self.current_source_sound.shape[0]:
                sound_segment = self.current_source_sound[index - rir.shape[0] + 1: index + num_sample]
            else:
                wraparound_sample = index + num_sample - self.current_source_sound.shape[0]
                sound_segment = np.concatenate([self.current_source_sound[index - rir.shape[0] + 1:],
                                                self.current_source_sound[: wraparound_sample]])
            # sound_segment = self.current_source_sound[index - rir.shape[0] + 1: index + num_sample]
            binaural_convolved = np.array([fftconvolve(sound_segment, rir[:, channel], mode='valid',
                                                       ) for channel in range(rir.shape[-1])])
            audiogoal = binaural_convolved

        # audiogoal = np.array([fftconvolve(self.current_source_sound, rir[:, channel], mode='full',
        #                                   ) for channel in range(rir.shape[-1])])
        # audiogoal = audiogoal[:, self._episode_step_count * num_sample: (self._episode_step_count + 1) * num_sample]
        audiogoal = np.pad(audiogoal, [(0, 0), (0, sampling_rate - audiogoal.shape[1])])

        return audiogoal

    def get_current_audiogoal_observation(self):
        return self._compute_audiogoal()

    def get_current_spectrogram_observation(self, audiogoal2spectrogram):
        return audiogoal2spectrogram(self.get_current_audiogoal_observation())

    def _get_semantic_uuid(self) -> Optional[str]:
        for uuid, sensor in self._sensor_suite.sensors.items():
            if getattr(sensor, "sim_sensor_type", None) == habitat_sim.SensorType.SEMANTIC:
                return uuid
        if "semantic" in self._sensor_suite.sensors:
            return "semantic"
        return None

    def _get_rgb_uuid(self) -> Optional[str]:
        for uuid, sensor in self._sensor_suite.sensors.items():
            if getattr(sensor, "sim_sensor_type", None) == habitat_sim.SensorType.COLOR:
                return uuid
        if "rgb" in self._sensor_suite.sensors:
            return "rgb"
        return None

    def _semantic_objects(self) -> List[Any]:
        scene = getattr(self._sim, "semantic_scene", None)
        if scene is None:
            return []
        objects = getattr(scene, "objects", None)
        if objects is None:
            return []
        if isinstance(objects, list):
            return objects
        if isinstance(objects, tuple):
            return list(objects)
        if hasattr(objects, "__iter__"):
            return list(objects)
        return []

    def resolve_semantic_target(
        self,
        object_id: int,
        goal_position: Optional[List[float]] = None,
        goal_category: Optional[str] = None,
    ) -> Tuple[int, str]:
        objects = self._semantic_objects()
        if not objects:
            return int(object_id), ""

        def _get_semantic_id(obj: Any) -> Optional[int]:
            for attr in ("semantic_id", "semanticID"):
                if hasattr(obj, attr):
                    return _as_int_or_none(getattr(obj, attr))
            return None

        # direct match on semantic id if possible (but respect goal_category when provided)
        for obj in objects:
            sem_id = _get_semantic_id(obj)
            if sem_id is not None and sem_id == int(object_id):
                name = getattr(obj, "category", None)
                cat_name = name.name() if name is not None else ""
                if goal_category and cat_name and cat_name != goal_category:
                    continue
                return sem_id, cat_name
            object_instance_id = _as_int_or_none(getattr(obj, "id", -1))
            if object_instance_id == int(object_id):
                name = getattr(obj, "category", None)
                cat_name = name.name() if name is not None else ""
                if goal_category and cat_name and cat_name != goal_category:
                    continue
                return sem_id if sem_id is not None else int(object_id), cat_name

        # fallback: nearest object with matching category name
        if goal_position is not None and goal_category:
            goal_pos = np.array(goal_position, dtype=np.float32)
            best = (float("inf"), None)
            for obj in objects:
                cat = getattr(obj, "category", None)
                if cat is None:
                    continue
                name_fn = getattr(cat, "name", None)
                if not callable(name_fn):
                    continue
                name = name_fn()
                if name != goal_category:
                    continue
                aabb = getattr(obj, "aabb", None)
                if aabb is None:
                    continue
                center = np.array(aabb.center, dtype=np.float32)
                dist = float(np.linalg.norm(center - goal_pos))
                if dist < best[0]:
                    best = (dist, obj)
            if best[1] is not None:
                obj = best[1]
                sem_id = _get_semantic_id(obj)
                name = getattr(obj, "category", None)
                return sem_id if sem_id is not None else int(getattr(obj, "id", object_id)), (
                    name.name() if name is not None else ""
                )

        return int(object_id), ""

    def find_best_viewpoint_for_object(
        self,
        object_id: int,
        goal_position: Optional[List[float]] = None,
        goal_category: Optional[str] = None,
        num_samples: int = 200,
        radius_min: float = 1.0,
        radius_max: float = 4.0,
        yaw_jitter: int = 4,
        id_offset: int = 0,
        local_sampling: bool = True,
        top_k: int = 5,
        exhaustive: bool = False,
        max_geodesic_dist: float = 1.0,
        obstacle_clearance: float = 0.3,
    ) -> Dict[str, Any]:
        semantic_uuid = self._get_semantic_uuid()
        if semantic_uuid is None:
            raise RuntimeError("Semantic sensor not configured. Add SEMANTIC_SENSOR to agent sensors.")

        if self.pathfinder is None:
            raise RuntimeError("Pathfinder not available.")

        if goal_position is None:
            if not hasattr(self.config.AGENT_0, "GOAL_POSITION"):
                raise AttributeError("GOAL_POSITION")
            goal_position = self.config.AGENT_0.GOAL_POSITION
        goal_pos = np.array(goal_position, dtype=np.float32)
        if hasattr(self.pathfinder, "snap_point"):
            snapped_goal = self.pathfinder.snap_point(goal_pos)
            goal_pos = np.array(snapped_goal, dtype=np.float32)
        target_id, target_name = self.resolve_semantic_target(
            object_id, goal_position, goal_category
        )
        if target_name == "" and target_id == int(object_id):
            raise RuntimeError(
                f"Semantic object id {object_id} not found in semantic scene."
            )
        best: Dict[str, Any] = {"coverage": -1.0, "position": None, "rotation": None}
        candidates: List[Dict[str, Any]] = []

        if exhaustive:
            verts = self.pathfinder.build_navmesh_vertices()
            if len(verts) == 0:
                raise RuntimeError("No navmesh vertices available.")
            positions = np.stack(verts, axis=0).astype(np.float32)
            # filter by radius bounds in XZ plane
            dx = positions[:, 0] - goal_pos[0]
            dz = positions[:, 2] - goal_pos[2]
            dist = np.sqrt(dx * dx + dz * dz)
            mask = (dist >= radius_min) & (dist <= radius_max)
            positions = positions[mask]
            if positions.shape[0] == 0:
                # fallback: take nearest vertices to goal_position
                order = np.argsort(dist)
                k = min(500, positions.shape[0] if positions is not None else 0)
                if k == 0:
                    positions = np.stack(verts, axis=0).astype(np.float32)
                    order = np.argsort(dist)
                    k = min(500, positions.shape[0])
                positions = positions[order[:k]]
        else:
            positions = None

        if positions is not None:
            pos_iter = positions
        else:
            pos_iter = range(num_samples)

        for pos_item in pos_iter:
            if positions is not None:
                pos = np.array(pos_item, dtype=np.float32)
            elif local_sampling:
                angle = np.random.uniform(0.0, 2 * math.pi)
                radius = np.random.uniform(radius_min, radius_max)
                candidate = np.array(
                    [
                        goal_pos[0] + radius * math.cos(angle),
                        goal_pos[1],
                        goal_pos[2] + radius * math.sin(angle),
                    ],
                    dtype=np.float32,
                )
                if hasattr(self.pathfinder, "snap_point"):
                    pos = self.pathfinder.snap_point(candidate)
                else:
                    pos = candidate
                if not self.pathfinder.is_navigable(pos):
                    continue
            else:
                pos = self.pathfinder.get_random_navigable_point()
                if pos is None:
                    continue
                pos = np.array(pos, dtype=np.float32)
                dist = np.linalg.norm(pos[[0, 2]] - goal_pos[[0, 2]])
                if dist < radius_min or dist > radius_max:
                    continue

            # enforce obstacle clearance if supported
            if hasattr(self.pathfinder, "distance_to_closest_obstacle"):
                dist_to_obs = self.pathfinder.distance_to_closest_obstacle(pos)
                if dist_to_obs < obstacle_clearance:
                    continue

            # enforce geodesic distance constraint
            geo = self.geodesic_distance(pos, [goal_pos])
            if geo is None or (not np.isfinite(geo)) or geo > max_geodesic_dist:
                continue

            # normalize position height to snapped goal height (agent stays on navmesh plane)
            pos[1] = goal_pos[1]
            base_yaw = math.atan2(goal_pos[0] - pos[0], goal_pos[2] - pos[2])
            if exhaustive:
                yaw_list = [math.radians(60 * k) for k in range(6)]
            else:
                yaw_list = []
                for jitter_idx in range(max(1, yaw_jitter)):
                    jitter = (
                        (jitter_idx - (yaw_jitter - 1) / 2.0)
                        * (math.pi / 18.0)
                    )
                    yaw_list.append(base_yaw + jitter)

            for yaw in yaw_list:
                rot = quat_from_angle_axis(yaw, np.array([0.0, 1.0, 0.0]))
                self.set_agent_state(pos, rot, reset_sensors=False)
                sim_obs = self._sim.get_sensor_observations()
                sem = sim_obs.get(semantic_uuid)
                if sem is None:
                    continue
                if sem.ndim == 3:
                    sem = sem[:, :, 0]
                mask = (sem == (target_id + id_offset))
                obj_pixels = float(np.sum(mask))
                total_pixels = float(mask.size)
                frame_coverage = obj_pixels / total_pixels if total_pixels > 0 else 0.0
                if obj_pixels > 0:
                    ys, xs = np.where(mask)
                    bbox_area = float((ys.max() - ys.min() + 1) * (xs.max() - xs.min() + 1))
                else:
                    bbox_area = 0.0
                object_coverage = obj_pixels / bbox_area if bbox_area > 0 else 0.0
                cov = object_coverage
                rot_list = _quat_to_list(rot)
                candidate = {
                    "coverage": cov,
                    "frame_coverage": frame_coverage,
                    "object_coverage": object_coverage,
                    "position": pos.tolist(),
                    "rotation": rot_list,
                }
                candidates.append(candidate)
                if cov > best["coverage"]:
                    best.update(
                        {"coverage": cov, "position": pos.tolist(), "rotation": rot_list}
                    )

        if best["position"] is None:
            raise RuntimeError("No valid viewpoint found. Increase num_samples or adjust radius.")
        candidates.sort(key=lambda x: x["coverage"], reverse=True)
        best["top_viewpoints"] = candidates[: max(1, top_k)]
        best["target_id"] = target_id
        best["target_name"] = target_name
        return best

    def render_goal_image(
        self,
        position: List[float],
        rotation,
        out_path: str,
    ) -> None:
        rgb_uuid = self._get_rgb_uuid()
        if rgb_uuid is None:
            raise RuntimeError("RGB sensor not configured. Add RGB_SENSOR to agent sensors.")
        self.set_agent_state(position, rotation, reset_sensors=False)
        sim_obs = self._sim.get_sensor_observations()
        rgb = sim_obs.get(rgb_uuid)
        if rgb is None:
            raise RuntimeError("RGB observation missing.")
        Image.fromarray(rgb).save(out_path)

    def set_active_sound_index(self, idx: int) -> bool:
        if not self._sound_ids or not self._sound_positions:
            return False
        if idx < 0 or idx >= len(self._sound_ids):
            return False
        if idx == self._active_sound_idx:
            return True
        self._active_sound_idx = idx
        self._current_sound = self._sound_ids[idx]
        self._current_sample_index = 0
        audio_sensor = self._sim.get_agent(0)._sensors["audio_sensor"]
        audio_sensor.setAudioSourceTransform(
            np.array(self._sound_positions[idx]) + np.array([0, 1.5, 0])
        )
        return True

    def geodesic_distance(self, position_a, position_bs, episode=None):
        requested_ends = np.array(
            [self._snap_to_navmesh(p) for p in position_bs]
        )
        use_cache = (
            episode is not None
            and episode._shortest_path_cache is not None
            and len(position_bs) == 1
        )
        if use_cache:
            cached_path = episode._shortest_path_cache
            cached_ends = getattr(cached_path, "requested_ends", None)
            use_cache = (
                cached_ends is not None
                and np.shape(cached_ends) == np.shape(requested_ends)
                and np.allclose(cached_ends, requested_ends)
            )

        if use_cache:
            path = episode._shortest_path_cache
        else:
            path = habitat_sim.MultiGoalShortestPath()
            path.requested_ends = requested_ends

        path.requested_start = self._snap_to_navmesh(position_a)

        found = self.pathfinder.find_path(path)
        if episode is not None:
            episode._shortest_path_cache = path
        if (not found) or (not np.isfinite(path.geodesic_distance)):
            return float("inf")
        return float(path.geodesic_distance)

    @property
    def previous_step_collided(self):
        return self._previous_step_collided

    def seed(self, seed):
        self._sim.seed(seed)

    def get_observations_at(
            self,
            position: Optional[List[float]] = None,
            rotation: Optional[List[float]] = None,
            keep_agent_at_new_pose: bool = False,
    ) -> Optional[Observations]:
        current_state = self.get_agent_state()
        if position is None or rotation is None:
            success = True
        else:
            success = self.set_agent_state(
                position, rotation, reset_sensors=False
            )

        if success:
            sim_obs = self._sim.get_sensor_observations()

            self._prev_sim_obs = sim_obs

            observations = self._sensor_suite.get_observations(sim_obs)
            if not keep_agent_at_new_pose:
                self.set_agent_state(
                    current_state.position,
                    current_state.rotation,
                    reset_sensors=False,
                )
            return observations
        else:
            return None

    def get_straight_shortest_path_points(self, position_a, position_b):
        path = habitat_sim.ShortestPath()
        path.requested_start = self._snap_to_navmesh(position_a)
        path.requested_end = self._snap_to_navmesh(position_b)
        found = self.pathfinder.find_path(path)
        if not found:
            return []
        return path.points

    def make_greedy_follower(self, *args, **kwargs):
        return self._sim.make_greedy_follower(*args, **kwargs)
