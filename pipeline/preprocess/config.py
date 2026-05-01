"""Chapter 3 preprocessing configuration defaults and helpers."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PreprocessConfig:
    target_sampling_rate_hz: float = 50.0
    window_size_seconds: float = 2.56
    overlap_ratio: float = 0.5
    interpolation_method: str = "linear"
    max_missing_ratio_per_window: float = 0.20
    derive_acc_magnitude: bool = True
    derive_gyro_magnitude: bool = True

    def __post_init__(self) -> None:
        if self.target_sampling_rate_hz <= 0:
            raise ValueError("target_sampling_rate_hz must be > 0")
        if self.window_size_seconds <= 0:
            raise ValueError("window_size_seconds must be > 0")
        if not (0.0 <= self.overlap_ratio < 1.0):
            raise ValueError("overlap_ratio must be in [0.0, 1.0)")
        if not (0.0 <= self.max_missing_ratio_per_window <= 1.0):
            raise ValueError("max_missing_ratio_per_window must be in [0.0, 1.0]")


    @property
    def window_size_samples(self) -> int:
        return int(round(self.target_sampling_rate_hz * self.window_size_seconds))

    @property
    def step_size_samples(self) -> int:
        step = int(round(self.window_size_samples * (1.0 - self.overlap_ratio)))
        return max(1, step)


def default_preprocess_config() -> PreprocessConfig:
    return PreprocessConfig()
