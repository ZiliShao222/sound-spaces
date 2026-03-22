from ss_baselines.omega_nav.perception.base import (
    AudioBearingEstimate,
    AudioObservationPacket,
    AudioPerceptionState,
    AudioProtocolState,
    MapObservation,
    PerceptionOutput,
    SemanticVoxelMapState,
)

__all__ = [
    "AudioBearingEstimate",
    "AudioObservationPacket",
    "AudioPerceptionState",
    "AudioProtocolState",
    "MapObservation",
    "PerceptionEncoder",
    "PerceptionOutput",
    "SemanticVoxelMapState",
]


def __getattr__(name: str):
    if name == "PerceptionEncoder":
        from ss_baselines.omega_nav.perception.encoder import PerceptionEncoder

        return PerceptionEncoder
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
