from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RealtimeTuning:
    smoothing: int
    min_confidence: float
    dual_arm_agree_threshold: float
    dual_arm_single_threshold: float | None
    output_hysteresis: bool
    hysteresis_active_enter_threshold: float
    hysteresis_active_exit_threshold: float
    hysteresis_active_switch_threshold: float
    hysteresis_neutral_enter_threshold: float
    hysteresis_enter_confirm_frames: int
    hysteresis_switch_confirm_frames: int
    hysteresis_neutral_confirm_frames: int
    softmax_reject_enabled: bool
    softmax_reject_min_confidence: float
    softmax_reject_min_margin: float
    prototype_reject_min_confidence: float
    prototype_reject_min_margin: float

    @property
    def resolved_dual_arm_single_threshold(self) -> float:
        if self.dual_arm_single_threshold is None:
            return float(self.min_confidence)
        return float(self.dual_arm_single_threshold)


@dataclass(frozen=True)
class CarlaTuning:
    gesture_max_age_s: float
    active_steer_dwell_frames: int
    neutral_steer_dwell_frames: int
    steer_left: float
    steer_right: float
    steer_left_strong: float
    steer_right_strong: float
    steer_neutral: float


# Used only for logging/CSV metadata. Edit the tuning objects below directly.
RUNTIME_TUNING_NAME = "manual"


# Edit these values directly when tuning realtime behavior.
REALTIME_TUNING = RealtimeTuning(
    smoothing=1,
    min_confidence=0.80,
    dual_arm_agree_threshold=0.55,
    dual_arm_single_threshold=None,
    output_hysteresis=False,
    hysteresis_active_enter_threshold=0.85,
    hysteresis_active_exit_threshold=0.60,
    hysteresis_active_switch_threshold=0.88,
    hysteresis_neutral_enter_threshold=0.75,
    hysteresis_enter_confirm_frames=1,
    hysteresis_switch_confirm_frames=1,
    hysteresis_neutral_confirm_frames=1,
    softmax_reject_enabled=False,
    softmax_reject_min_confidence=0.55,
    softmax_reject_min_margin=0.08,
    prototype_reject_min_confidence=0.55,
    prototype_reject_min_margin=0.08,
)

# Edit these values directly when tuning CARLA control behavior.
CARLA_TUNING = CarlaTuning(
    gesture_max_age_s=0.75,
    active_steer_dwell_frames=1,
    neutral_steer_dwell_frames=1,
    steer_left=-0.08,
    steer_right=0.08,
    steer_left_strong=-0.30,
    steer_right_strong=0.30,
    steer_neutral=0.0,
)
