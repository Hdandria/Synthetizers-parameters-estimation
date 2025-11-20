from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

from loguru import logger

try:
    # Only for typing/context; pedalboard is required at runtime where this is used
    from pedalboard import VST3Plugin
except Exception:
    VST3Plugin = Any  # type: ignore

from src.data.vst.param_spec import (
    CategoricalParameter,
    ContinuousParameter,
    DiscreteLiteralParameter,
    Parameter,
)
from src.data.vst.vital_param_spec import VITAL_PARAM_SPEC
from src.data.vst.vital_details import get_min_max_for_plugin_name


CORE_COMPONENT_MAP: Dict[str, str] = {
    # Support both legacy and canonical prefixes
    "osc": "osc",
    "oscillator": "osc",
    "env": "env",
    "envelope": "env",
    "random": "random",
    "random_lfo": "random",
    "reverb": "reverb",
    "chorus": "chorus",
    "phaser": "phaser",
    "flanger": "flanger",
    "eq": "eq",
    "delay": "delay",
    "compressor": "compressor",
    "sample": "sample",
    "modulation": "modulation",
}

COMMON_SUFFIX_MAP: Dict[str, str] = {
    # Map pedalboard naming back to original Vital internal names (reverse direction for preset keys)
    "_switch": "_on",
    "_phase_randomization": "_random_phase",
    "_frequency_morph_type": "_spectral_morph_type",
    "_frequency_morph_amount": "_spectral_morph_amount",
    "_frequency_morph_spread": "_spectral_morph_spread",
}

# Explicit overrides from legacy preset key -> pedalboard key
EXPLICIT_KEY_MAP: Dict[str, str] = {
    "delay_dry_wet": "delay_mix",
    "reverb_dry_wet": "reverb_mix",
    "chorus_dry_wet": "chorus_mix",
    "flanger_dry_wet": "flanger_mix",
    "phaser_dry_wet": "phaser_mix",
    "delay_on": "delay_switch",
    "reverb_on": "reverb_switch",
    "chorus_on": "chorus_switch",
    "flanger_on": "flanger_switch",
    "phaser_on": "phaser_switch",
    "distortion_on": "distortion_switch",
    "compressor_on": "compressor_switch",
    "filter_1_on": "filter_1_switch",
    "filter_2_on": "filter_2_switch",
    "filter_fx_on": "filter_fx_switch",
    "sample_on": "sample_switch",
    "macro_control_1": "macro_1",
    "macro_control_2": "macro_2",
    "macro_control_3": "macro_3",
    "macro_control_4": "macro_4",
    # Missed chorus mappings from preset -> spec/pedalboard
    "chorus_cutoff": "chorus_filter_cutoff",
    "chorus_spread": "chorus_filter_spread",
    # Compressor detailed thresholds/ratios (preset includes compressor_* prefixes)
    "compressor_low_upper_threshold": "low_upper_threshold",
    "compressor_low_lower_threshold": "low_lower_threshold",
    "compressor_low_upper_ratio": "low_upper_ratio",
    "compressor_low_lower_ratio": "low_lower_ratio",
    "compressor_band_upper_threshold": "band_upper_threshold",
    "compressor_band_lower_threshold": "band_lower_threshold",
    "compressor_band_upper_ratio": "band_upper_ratio",
    "compressor_band_lower_ratio": "band_lower_ratio",
    "compressor_high_upper_threshold": "high_upper_threshold",
    "compressor_high_lower_threshold": "high_lower_threshold",
    "compressor_high_upper_ratio": "high_upper_ratio",
    "compressor_high_lower_ratio": "high_lower_ratio",
}

_PARAM_REGISTRY: Dict[str, Parameter] = {
    param.name: param for param in VITAL_PARAM_SPEC.synth_params
}


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _normalize_continuous(param: ContinuousParameter, value: Any) -> float:
    """Normalize internal Vital engine value to plugin raw_value.

    Preferred path: use precise min/max from VITAL_PARAM_DETAILS (linear transform). This mirrors
    ValueBridge::convertToPluginValue: (engine - min) / (max - min). Display skew is intentionally
    ignored because the plugin expects unskewed 0..1.

    Fallback path: if details missing, revert to legacy heuristics (simple clamping / percentage scaling).
    """
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        numeric = param.min

    min_max = get_min_max_for_plugin_name(param.name)
    if min_max is not None:
        lo, hi = min_max
        span = hi - lo if hi != lo else 1.0
        raw = (numeric - lo) / span
        return _clamp(raw, 0.0, 1.0)

    # Fallback heuristic (retain simple handling for parameters not yet in details map)
    if 0.0 <= numeric <= 1.0:
        return _clamp(numeric, param.min, param.max)
    if 1.0 < numeric <= 100.0:
        return _clamp(numeric / 100.0, param.min, param.max)
    return _clamp(numeric, param.min, param.max)


def _normalize_categorical(param: CategoricalParameter, value: Any) -> float:
    raw_values = param.raw_values or []
    if isinstance(value, bool):
        candidate = float(value)
    else:
        try:
            candidate = float(value)
        except (TypeError, ValueError):
            candidate = None

    if candidate is not None and raw_values:
        return min(raw_values, key=lambda rv: abs(rv - candidate))

    if value in param.values:
        idx = param.values.index(value)
        return raw_values[idx]

    return raw_values[0] if raw_values else 0.0


def _normalize_value_from_spec(param_name: str, value: Any) -> Any:
    param = _PARAM_REGISTRY.get(param_name)
    if param is None:
        return _legacy_coerce(value)

    if isinstance(param, ContinuousParameter):
        return _normalize_continuous(param, value)
    if isinstance(param, CategoricalParameter):
        return _normalize_categorical(param, value)
    if isinstance(param, DiscreteLiteralParameter):
        return _legacy_coerce(value)
    return value


def _remap_key(old_key: str) -> str:
    """Map a legacy Vital preset parameter name to a pedalboard parameter name.

    Order of operations:
    1. Explicit mapping dictionary.
    2. Structured group conversions (osc/env/lfo/random/filter/modulation).
    3. Generic suffix heuristics (on->switch, dry_wet->mix, spectral->frequency).
    4. Fallback: return original key.
    """
    if old_key in EXPLICIT_KEY_MAP:
        return EXPLICIT_KEY_MAP[old_key]

    # Oscillators: osc_<n>_* -> oscillator_<n>_* ; spectral_morph_* -> frequency_morph_*
    if old_key.startswith("osc_"):
        parts = old_key.split("_")
        if len(parts) >= 3 and parts[1].isdigit():
            idx = parts[1]
            tail = "_".join(parts[2:])
            tail = tail.replace("spectral_morph_", "frequency_morph_")
            return f"oscillator_{idx}_{tail}"

    # Envelopes
    if old_key.startswith("env_"):
        parts = old_key.split("_")
        if len(parts) >= 3 and parts[1].isdigit():
            idx = parts[1]
            tail = "_".join(parts[2:])
            return f"envelope_{idx}_{tail}"

    # LFOs
    if old_key.startswith("lfo_"):
        parts = old_key.split("_")
        if len(parts) >= 3 and parts[1].isdigit():
            idx = parts[1]
            tail = "_".join(parts[2:])
            return f"lfo_{idx}_{tail}"

    # Random LFOs
    if old_key.startswith("random_"):
        parts = old_key.split("_")
        if len(parts) >= 3 and parts[1].isdigit():
            idx = parts[1]
            tail = "_".join(parts[2:])
            return f"random_lfo_{idx}_{tail}"

    # Filters: filter_<n>_* keep naming; *_on -> *_switch handled below
    if old_key.startswith("filter_") and old_key.endswith("_on"):
        return old_key.replace("_on", "_switch")

    # Sample switch
    if old_key == "sample_on":
        return "sample_switch"

    # Generic FX suffixes
    if old_key.endswith("dry_wet"):
        return old_key.replace("dry_wet", "mix")
    if old_key.endswith("_on"):
        return old_key.replace("_on", "_switch")
    if "spectral_morph_" in old_key:
        return old_key.replace("spectral_morph_", "frequency_morph_")

    # Heuristic prefix replacement attempt
    for prefix, mapped_prefix in CORE_COMPONENT_MAP.items():
        if old_key.startswith(prefix):
            candidate = old_key.replace(prefix, mapped_prefix, 1)
            for suffix, mapped_suffix in COMMON_SUFFIX_MAP.items():
                if candidate.endswith(suffix):
                    candidate = candidate[: -len(suffix)] + mapped_suffix
                    break
            return candidate

    # Suffix-only heuristic
    for suffix, mapped_suffix in COMMON_SUFFIX_MAP.items():
        if old_key.endswith(suffix):
            return old_key[: -len(suffix)] + mapped_suffix

    return old_key


def _legacy_coerce(value: Any) -> Any:
    """Fallback coercion when no parameter metadata is available."""
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        v = float(value)
        if 1.0 < v <= 100.0:
            return v / 100.0
        return v
    return value


def convert_vital_preset_to_params(preset_path, plugin):
    """Convert a Vital .vital preset into a dict of plugin parameter raw_values.

    This reads the legacy JSON format, remaps keys to Pedalboard Vital parameter
    names, applies light heuristics to coerce values to valid raw_value ranges,
    and filters to keys actually supported by the loaded plugin.

    It does not set values on the plugin; the returned dict can be passed to
    the existing set_params() call path.
    """
    preset_path = str(preset_path)
    with open(preset_path, "rb") as f:
        try:
            data = json.load(f)
        except Exception as e:  # pragma: no cover - tight scope utility
            logger.error(f"Failed to read .vital preset JSON at {preset_path}: {e}")
            raise

    settings = data.get("settings", {})
    if not isinstance(settings, dict):
        logger.warning(
            "Unexpected .vital format: 'settings' is missing or not a dict; nothing to apply."
        )
        return {}

    # Remap keys using explicit + heuristic rules, track unmapped originals
    remapped: Dict[str, Any] = {}
    unmapped_original: list[str] = []
    for key, value in settings.items():
        new_key = _remap_key(key)
        if new_key == key:
            # If unchanged and not a plugin param, mark as unmapped candidate
            if key not in plugin.parameters:
                unmapped_original.append(key)
        remapped[new_key] = value

    plugin_params = set(plugin.parameters.keys())
    intersect = plugin_params.intersection(remapped.keys())
    if not intersect:
        logger.warning(
            "No overlapping parameters between .vital preset and plugin; check mapping rules."
        )
    else:
        skipped = len(remapped) - len(intersect)
        if skipped > 0:
            sample_skipped = list(sorted(set(remapped.keys()) - intersect))[:10]
            logger.debug(
                f"Vital preset coverage: applying {len(intersect)}/{len(remapped)}. Sample skipped: {sample_skipped}"
            )
        if unmapped_original:
            logger.debug(
                f"Original keys unmapped (first 10): {unmapped_original[:10]}"
            )

    result: Dict[str, Any] = {}
    exact = 0
    fallback = 0
    for k in intersect:
        before = remapped[k]
        has_exact = get_min_max_for_plugin_name(k) is not None
        normalized = _normalize_value_from_spec(k, before)
        result[k] = normalized
        if has_exact:
            exact += 1
        else:
            fallback += 1

    logger.info(
        f"Converted .vital preset: mapped {len(settings)} keys -> {len(remapped)}; applying {len(result)} parameters. "
        f"Normalization coverage exact={exact} fallback={fallback} ({exact/(exact+fallback):.1%} exact)."
    )
    return result
