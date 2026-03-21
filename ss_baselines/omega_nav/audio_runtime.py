from __future__ import annotations

import importlib
import importlib.util
from pathlib import Path
from typing import Any, Optional, Tuple

import numpy as np
import torch
from scipy.io import wavfile


def load_librosa_module() -> Optional[Any]:
    if importlib.util.find_spec("librosa") is None:
        return None
    return importlib.import_module("librosa")


def load_clap_backend(
    *,
    encoder_model_name: str,
    device: torch.device,
) -> Tuple[Any, Any]:
    transformers = importlib.import_module("transformers")
    clap_processor = transformers.AutoProcessor.from_pretrained(str(encoder_model_name))
    clap_model = transformers.ClapModel.from_pretrained(str(encoder_model_name))
    clap_model.eval()
    clap_model.to(device)
    return clap_processor, clap_model


def resample_audio(
    waveform: np.ndarray,
    *,
    orig_sr: int,
    target_sr: int,
    librosa_module: Optional[Any],
) -> np.ndarray:
    audio = np.asarray(waveform, dtype=np.float32).reshape(-1)
    if int(orig_sr) == int(target_sr):
        return audio.astype(np.float32)
    if librosa_module is not None:
        return np.asarray(
            librosa_module.resample(audio, orig_sr=int(orig_sr), target_sr=int(target_sr)),
            dtype=np.float32,
        )

    duration = float(audio.size) / float(orig_sr)
    target_size = int(round(duration * int(target_sr)))
    source_positions = np.linspace(0.0, 1.0, num=audio.size, endpoint=False, dtype=np.float32)
    target_positions = np.linspace(0.0, 1.0, num=target_size, endpoint=False, dtype=np.float32)
    return np.interp(target_positions, source_positions, audio).astype(np.float32)


def load_audio_file(
    wav_path: Path,
    *,
    target_sr: int,
    librosa_module: Optional[Any],
) -> Tuple[np.ndarray, int]:
    if librosa_module is not None:
        waveform, sampling_rate_hz = librosa_module.load(
            str(wav_path),
            sr=int(target_sr),
            mono=True,
        )
        return np.asarray(waveform, dtype=np.float32), int(sampling_rate_hz)

    sampling_rate_hz, waveform = wavfile.read(str(wav_path))
    array = np.asarray(waveform)
    if array.ndim == 2:
        array = np.mean(array.astype(np.float32), axis=1)
    if np.issubdtype(array.dtype, np.integer):
        max_value = max(abs(np.iinfo(array.dtype).min), np.iinfo(array.dtype).max)
        array = array.astype(np.float32) / float(max_value)
    else:
        array = array.astype(np.float32)

    resampled = resample_audio(
        array,
        orig_sr=int(sampling_rate_hz),
        target_sr=int(target_sr),
        librosa_module=None,
    )
    return resampled.astype(np.float32), int(target_sr)


def encode_clap_audio(
    *,
    clap_processor: Any,
    clap_model: Any,
    mono: np.ndarray,
    sampling_rate_hz: int,
    device: torch.device,
) -> np.ndarray:
    with torch.inference_mode():
        inputs = clap_processor(
            audio=[mono],
            sampling_rate=int(sampling_rate_hz),
            padding=True,
            return_tensors="pt",
        )
        inputs = {
            key: value.to(device)
            for key, value in inputs.items()
        }
        features = clap_model.get_audio_features(**inputs)
    feature_tensor = features if torch.is_tensor(features) else features.pooler_output
    return np.asarray(feature_tensor.detach().cpu().numpy()[0], dtype=np.float32)
