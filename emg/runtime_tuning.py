from __future__ import annotations

from dataclasses import dataclass, replace


# Edit this one name to switch the active runtime tuning preset.
ACTIVE_RUNTIME_TUNING_PRESET = "baseline"


@dataclass(frozen=True)
class RealtimeTuning:
    smoothing: int = 1
    min_confidence: float = 0.80
    dual_arm_agree_threshold: float = 0.55
    dual_arm_single_threshold: float | None = None
    output_hysteresis: bool = False
    hysteresis_active_enter_threshold: float = 0.85
    hysteresis_active_exit_threshold: float = 0.60
    hysteresis_active_switch_threshold: float = 0.88
    hysteresis_neutral_enter_threshold: float = 0.75
    hysteresis_enter_confirm_frames: int = 1
    hysteresis_switch_confirm_frames: int = 1
    hysteresis_neutral_confirm_frames: int = 1
    softmax_reject_enabled: bool = False
    softmax_reject_min_confidence: float = 0.55
    softmax_reject_min_margin: float = 0.08
    prototype_reject_min_confidence: float = 0.55
    prototype_reject_min_margin: float = 0.08

    @property
    def resolved_dual_arm_single_threshold(self) -> float:
        if self.dual_arm_single_threshold is None:
            return float(self.min_confidence)
        return float(self.dual_arm_single_threshold)


@dataclass(frozen=True)
class CarlaTuning:
    gesture_max_age_s: float = 0.75
    active_steer_dwell_frames: int = 1
    neutral_steer_dwell_frames: int = 1
    steer_left: float = -0.08
    steer_right: float = 0.08
    steer_left_strong: float = -0.4
    steer_right_strong: float = 0.4
    steer_neutral: float = 0.0


@dataclass(frozen=True)
class RuntimeTuningPreset:
    name: str
    description: str
    realtime: RealtimeTuning
    carla: CarlaTuning


_BASE_REALTIME = RealtimeTuning()
_BASE_CARLA = CarlaTuning()
_CARLA_DWELL2 = replace(
    _BASE_CARLA,
    active_steer_dwell_frames=2,
    neutral_steer_dwell_frames=2,
)
_FLICKER_MILD_REALTIME = replace(
    _BASE_REALTIME,
    smoothing=3,
    output_hysteresis=True,
    hysteresis_neutral_enter_threshold=0.80,
    hysteresis_enter_confirm_frames=2,
    hysteresis_switch_confirm_frames=2,
    hysteresis_neutral_confirm_frames=2,
)
_FLICKER_STRONG_REALTIME = replace(
    _BASE_REALTIME,
    smoothing=5,
    min_confidence=0.82,
    output_hysteresis=True,
    hysteresis_active_enter_threshold=0.88,
    hysteresis_active_exit_threshold=0.62,
    hysteresis_active_switch_threshold=0.90,
    hysteresis_neutral_enter_threshold=0.82,
    hysteresis_enter_confirm_frames=3,
    hysteresis_switch_confirm_frames=2,
    hysteresis_neutral_confirm_frames=2,
)

_PRESETS: dict[str, RuntimeTuningPreset] = {
    "baseline": RuntimeTuningPreset(
        name="baseline",
        description="Current repo defaults before flicker-tuning changes.",
        realtime=_BASE_REALTIME,
        carla=_BASE_CARLA,
    ),
    "flicker_mild": RuntimeTuningPreset(
        name="flicker_mild",
        description="Mild smoothing and hysteresis for the first neutral-flicker pass.",
        realtime=_FLICKER_MILD_REALTIME,
        carla=_BASE_CARLA,
    ),
    "flicker_mild_margin": RuntimeTuningPreset(
        name="flicker_mild_margin",
        description="Mild smoothing and hysteresis plus softmax ambiguity rejection.",
        realtime=replace(
            _FLICKER_MILD_REALTIME,
            softmax_reject_enabled=True,
            softmax_reject_min_confidence=0.80,
            softmax_reject_min_margin=0.10,
        ),
        carla=_BASE_CARLA,
    ),
    "flicker_strong": RuntimeTuningPreset(
        name="flicker_strong",
        description="Stronger latching for aggressive neutral-flicker suppression.",
        realtime=_FLICKER_STRONG_REALTIME,
        carla=_BASE_CARLA,
    ),
    "flicker_strong_margin": RuntimeTuningPreset(
        name="flicker_strong_margin",
        description="Strong hysteresis plus softmax ambiguity rejection.",
        realtime=replace(
            _FLICKER_STRONG_REALTIME,
            softmax_reject_enabled=True,
            softmax_reject_min_confidence=0.82,
            softmax_reject_min_margin=0.12,
        ),
        carla=_BASE_CARLA,
    ),
    "flicker_mild_margin_dwell2": RuntimeTuningPreset(
        name="flicker_mild_margin_dwell2",
        description="Mild realtime gating plus 2-frame CARLA steering dwell.",
        realtime=replace(
            _FLICKER_MILD_REALTIME,
            softmax_reject_enabled=True,
            softmax_reject_min_confidence=0.80,
            softmax_reject_min_margin=0.10,
        ),
        carla=_CARLA_DWELL2,
    ),
    "flicker_strong_margin_dwell2": RuntimeTuningPreset(
        name="flicker_strong_margin_dwell2",
        description="Strong realtime gating plus 2-frame CARLA steering dwell.",
        realtime=replace(
            _FLICKER_STRONG_REALTIME,
            softmax_reject_enabled=True,
            softmax_reject_min_confidence=0.82,
            softmax_reject_min_margin=0.12,
        ),
        carla=_CARLA_DWELL2,
    ),
}


def list_runtime_tuning_presets() -> tuple[str, ...]:
    return tuple(sorted(_PRESETS))


def get_runtime_tuning_preset(name: str | None = None) -> RuntimeTuningPreset:
    preset_name = str(name or ACTIVE_RUNTIME_TUNING_PRESET).strip()
    try:
        return _PRESETS[preset_name]
    except KeyError as exc:
        available = ", ".join(list_runtime_tuning_presets())
        raise ValueError(
            f"Unknown runtime tuning preset {preset_name!r}. Available presets: {available}"
        ) from exc


ACTIVE_RUNTIME_TUNING = get_runtime_tuning_preset()
