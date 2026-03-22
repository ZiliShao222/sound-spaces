from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch


from ss_baselines.omega_nav.audio_runtime import (
    encode_clap_audio,
    load_audio_file,
    load_clap_backend,
    load_librosa_module,
    resample_audio,
)
from ss_baselines.omega_nav.utils import cosine_similarity, normalize_text


DEFAULT_CATEGORY_ALIASES: Dict[str, Tuple[str, ...]] = {
    "chair": ("chair",),
    "table": ("table", "dining table", "desk", "coffee table", "end table"),
    "picture": ("picture", "painting", "poster", "wall art", "photo"),
    "cabinet": ("cabinet", "cupboard"),
    "cushion": ("cushion", "pillow"),
    "sofa": ("sofa", "couch"),
    "bed": ("bed",),
    "chest_of_drawers": (
        "chest_of_drawers",
        "chest of drawers",
        "dresser",
        "drawer cabinet",
    ),
    "plant": ("plant", "potted plant"),
    "sink": ("sink", "basin"),
    "toilet": ("toilet",),
    "stool": ("stool",),
    "towel": ("towel",),
    "tv_monitor": ("tv_monitor", "tv monitor", "tv", "television", "monitor", "screen"),
    "shower": ("shower", "shower stall"),
    "bathtub": ("bathtub", "bath tub", "tub"),
    "counter": ("counter", "countertop"),
    "fireplace": ("fireplace",),
    "gym_equipment": (
        "gym_equipment",
        "gym equipment",
        "exercise equipment",
        "treadmill",
        "exercise bike",
        "elliptical",
    ),
    "seating": ("seating", "seat", "armchair", "bench"),
    "clothes": ("clothes", "clothing", "garment", "set of clothing"),
}


def canonicalize_audio_category(value: Any) -> str:
    text = normalize_text(value).strip().lower()
    text = text.replace("/", " ")
    text = text.replace("-", " ")
    text = re.sub(r"[^a-z0-9_ ]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text.replace(" ", "_")


class OfflineAcousticPrototypeLibrary:
    def __init__(
        self,
        *,
        clean_sound_dir: str,
        observation_sampling_rate_hz: int,
        encoder_model_name: str = "laion/clap-htsat-unfused",
        encoder_sampling_rate_hz: Optional[int] = None,
        prototype_window_sec: float = 1.0,
        prototype_hop_sec: float = 0.5,
        prototype_min_window_rms: float = 0.01,
        prototype_window_rms_ratio: float = 0.25,
        device: Optional[torch.device] = None,
    ) -> None:
        self._clean_sound_dir = Path(clean_sound_dir)
        self._observation_sampling_rate_hz = max(int(observation_sampling_rate_hz), 1)
        self._encoder_model_name = str(encoder_model_name)
        self._device = device or torch.device("cpu")
        self._prototype_window_sec = max(float(prototype_window_sec), 0.1)
        self._prototype_hop_sec = max(float(prototype_hop_sec), 0.05)
        self._prototype_min_window_rms = max(float(prototype_min_window_rms), 0.0)
        self._prototype_window_rms_ratio = max(float(prototype_window_rms_ratio), 0.0)
        self._encoder_sampling_rate_hz = max(
            int(encoder_sampling_rate_hz or observation_sampling_rate_hz),
            1,
        )

        self._alias_to_category: Dict[str, str] = {}
        for category, aliases in DEFAULT_CATEGORY_ALIASES.items():
            canonical = canonicalize_audio_category(category)
            self._alias_to_category[canonical] = canonical
            self._alias_to_category[canonical.replace("_", " ")] = canonical
            for alias in aliases:
                alias_key = canonicalize_audio_category(alias)
                if alias_key:
                    self._alias_to_category[alias_key] = canonical
                    self._alias_to_category[alias_key.replace("_", " ")] = canonical

        self._backend_name = "pending"
        self._clap_processor = None
        self._clap_model = None
        self._librosa = load_librosa_module()
        self._prototype_labels: Tuple[str, ...] = ()
        self._prototype_matrix: Optional[np.ndarray] = None
        self._category_to_index: Dict[str, int] = {}
        self._loaded = False

    @property
    def backend_name(self) -> str:
        return str(self._backend_name)

    @property
    def clean_sound_dir(self) -> str:
        return str(self._clean_sound_dir)

    @property
    def prototype_labels(self) -> Tuple[str, ...]:
        return self._prototype_labels

    @property
    def prototype_matrix(self) -> Optional[np.ndarray]:
        return None if self._prototype_matrix is None else np.asarray(self._prototype_matrix, dtype=np.float32)

    @property
    def category_to_index(self) -> Dict[str, int]:
        self.ensure_loaded()
        return dict(self._category_to_index)

    @property
    def alias_to_category(self) -> Dict[str, str]:
        return dict(self._alias_to_category)

    def ensure_loaded(self) -> None:
        if self._loaded:
            return
        self._loaded = True
        reference_files = sorted(self._clean_sound_dir.glob("*.wav"))
        self._ensure_encoder_backend()

        labels: List[str] = []
        embeddings: List[np.ndarray] = []
        for wav_path in reference_files:
            category = canonicalize_audio_category(wav_path.stem)
            embedding = self._build_file_prototype(wav_path)
            labels.append(category)
            embeddings.append(np.asarray(embedding, dtype=np.float32))
            self._alias_to_category.setdefault(category, category)
            self._alias_to_category.setdefault(category.replace("_", " "), category)

        self._prototype_labels = tuple(labels)
        self._prototype_matrix = np.stack(embeddings, axis=0).astype(np.float32)
        self._category_to_index = {
            str(category): int(index)
            for index, category in enumerate(self._prototype_labels)
        }

    def resolve_category(self, value: Any) -> str:
        normalized = canonicalize_audio_category(value)
        if normalized in self._alias_to_category:
            return self._alias_to_category[normalized]
        spaced = normalized.replace("_", " ")
        if spaced in self._alias_to_category:
            return self._alias_to_category[spaced]
        return normalized

    def prototype_for_category(self, value: Any) -> np.ndarray:
        self.ensure_loaded()
        category = self.resolve_category(value)
        index = self._prototype_labels.index(category)
        return np.asarray(self._prototype_matrix[index], dtype=np.float32)

    def goal_prototypes(self, goal_labels: Sequence[Any]) -> Dict[str, np.ndarray]:
        self.ensure_loaded()
        prototypes: Dict[str, np.ndarray] = {}
        for label in goal_labels:
            prototype = self.prototype_for_category(label)
            category = self.resolve_category(label)
            prototypes[category] = np.asarray(prototype, dtype=np.float32)
        return prototypes

    def encode_audio_observation(self, audio: np.ndarray) -> np.ndarray:
        mono = self._to_mono_observation(audio)
        self.ensure_loaded()
        return self._encode_waveform(mono, sampling_rate_hz=self._observation_sampling_rate_hz)

    def score_embedding(self, embedding: np.ndarray) -> Dict[str, float]:
        self.ensure_loaded()
        query = np.asarray(embedding, dtype=np.float32)
        scores: Dict[str, float] = {}
        for index, category in enumerate(self._prototype_labels):
            scores[str(category)] = float(cosine_similarity(query, self._prototype_matrix[index]))
        return scores

    def _ensure_encoder_backend(self) -> None:
        if self._backend_name != "pending":
            return
        clap_processor, clap_model = load_clap_backend(
            encoder_model_name=self._encoder_model_name,
            device=self._device,
        )
        self._clap_processor = clap_processor
        self._clap_model = clap_model
        self._encoder_sampling_rate_hz = int(clap_processor.feature_extractor.sampling_rate)
        self._backend_name = f"clap:{self._encoder_model_name}"

    def _build_file_prototype(self, wav_path: Path) -> np.ndarray:
        waveform, sampling_rate_hz = load_audio_file(
            wav_path,
            target_sr=self._encoder_sampling_rate_hz,
            librosa_module=self._librosa,
        )
        mono = self._prepare_waveform(waveform)

        window_size = int(round(self._prototype_window_sec * self._encoder_sampling_rate_hz))
        hop_size = int(round(self._prototype_hop_sec * self._encoder_sampling_rate_hz))
        windows = self._sliding_windows(mono, window_size=window_size, hop_size=hop_size)
        windows = self._select_informative_windows(windows)

        embeddings: List[np.ndarray] = []
        for window in windows:
            embedding = self._encode_waveform(window, sampling_rate_hz=self._encoder_sampling_rate_hz)
            embeddings.append(np.asarray(embedding, dtype=np.float32))

        prototype = np.mean(np.stack(embeddings, axis=0), axis=0).astype(np.float32)
        norm = float(np.linalg.norm(prototype))
        if norm <= 1e-8:
            return prototype.astype(np.float32)
        prototype = prototype / norm
        return prototype.astype(np.float32)

    def _to_mono_observation(self, audio: np.ndarray) -> np.ndarray:
        array = np.asarray(audio, dtype=np.float32)
        if array.ndim == 1:
            return self._prepare_waveform(array)
        if array.shape[0] == 2:
            return self._prepare_waveform(np.mean(array[:2], axis=0))
        if array.shape[-1] == 2:
            return self._prepare_waveform(np.mean(array[:, :2], axis=1))
        return self._prepare_waveform(np.mean(array, axis=0))

    def _prepare_waveform(self, waveform: np.ndarray) -> np.ndarray:
        mono = np.asarray(waveform, dtype=np.float32).reshape(-1)
        mono = mono - float(np.mean(mono))
        peak = float(np.max(np.abs(mono)))
        if peak <= 1e-8:
            return np.zeros_like(mono, dtype=np.float32)
        mono = mono / peak
        return mono.astype(np.float32)

    def _sliding_windows(self, waveform: np.ndarray, *, window_size: int, hop_size: int) -> List[np.ndarray]:
        audio = np.asarray(waveform, dtype=np.float32).reshape(-1)
        if audio.size <= window_size:
            return [self._pad_or_trim(audio, window_size)]

        windows: List[np.ndarray] = []
        for start_index in range(0, audio.size - window_size + 1, hop_size):
            end_index = start_index + window_size
            windows.append(self._pad_or_trim(audio[start_index:end_index], window_size))
        return windows

    def _select_informative_windows(self, windows: Sequence[np.ndarray]) -> List[np.ndarray]:
        if not windows:
            return []
        rms_values = np.asarray(
            [float(np.sqrt(np.mean(np.square(np.asarray(window, dtype=np.float32))))) for window in windows],
            dtype=np.float32,
        )
        max_rms = float(np.max(rms_values)) if rms_values.size > 0 else 0.0
        threshold = max(
            float(self._prototype_min_window_rms),
            float(self._prototype_window_rms_ratio) * max_rms,
        )
        selected = [
            np.asarray(window, dtype=np.float32)
            for window, rms in zip(windows, rms_values)
            if float(rms) >= threshold
        ]
        if selected:
            return selected
        best_index = int(np.argmax(rms_values)) if rms_values.size > 0 else 0
        return [np.asarray(windows[best_index], dtype=np.float32)]

    def _pad_or_trim(self, waveform: np.ndarray, target_size: int) -> np.ndarray:
        audio = np.asarray(waveform, dtype=np.float32).reshape(-1)
        if audio.size == target_size:
            return audio.astype(np.float32)
        if audio.size > target_size:
            return audio[:target_size].astype(np.float32)
        return np.pad(audio, (0, target_size - audio.size), mode="constant").astype(np.float32)

    def _encode_waveform(
        self,
        waveform: np.ndarray,
        *,
        sampling_rate_hz: int,
    ) -> np.ndarray:
        mono = self._prepare_waveform(waveform)
        self._ensure_encoder_backend()

        clap_mono = resample_audio(
            mono,
            orig_sr=int(sampling_rate_hz),
            target_sr=int(self._encoder_sampling_rate_hz),
            librosa_module=self._librosa,
        )
        embedding = encode_clap_audio(
            clap_processor=self._clap_processor,
            clap_model=self._clap_model,
            mono=clap_mono,
            sampling_rate_hz=int(self._encoder_sampling_rate_hz),
            device=self._device,
        )
        return self._normalize_embedding(embedding)

    def _normalize_embedding(self, embedding: np.ndarray) -> np.ndarray:
        array = np.asarray(embedding, dtype=np.float32).reshape(-1)
        norm = float(np.linalg.norm(array))
        if norm <= 1e-6:
            return array.astype(np.float32)
        return (array / norm).astype(np.float32)
