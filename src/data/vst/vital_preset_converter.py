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


CORE_COMPONENT_MAP: Dict[str, str] = {
    # Component Mapping
    "osc": "oscillator",
    "env": "envelope",
    "random": "random_lfo",

    # Macro Control Mapping
    "macro_control": "macro",

    # Effect Mapping
    "reverb": "reverb",
    "chorus": "chorus",
    "phaser": "phaser",
    "flanger": "flanger",
    "eq": "eq",
    "delay": "delay",
    "compressor": "compressor",

    # Other Core Modules
    "sample": "sample",
    "wavetables": "wavetables",
    "lfos": "lfos",
    "modulations": "modulations",
}

COMMON_SUFFIX_MAP: Dict[str, str] = {
    # On/Off
    "_on": "_switch",

    # Envelope Parameter Mapping
    "_attack": "_attack",
    "_decay": "_decay",
    "_sustain": "_sustain",
    "_release": "_release",
    "_hold": "_hold",

    # LFO Keytracking
    "_keytrack_tune": "_tune",
    "_keytrack_transpose": "_transpose",
    "_fade_time": "_fade_in",

    # Oscillator Specific
    "_spectral_morph": "_frequency_morph",
    "_random_phase": "_phase_randomization",
    "_frame_spread": "_frame_spread",

    # Effect Mix
    "_dry_wet": "_mix",

    # Filter/Effect Keytracking
    "_keytrack": "_key_track",
}


def _remap_key(old_key: str) -> str:
    """Apply prefix and suffix mappings to transform old .vital keys to plugin keys."""
    new_key = old_key
    # Prefix mapping (replace only at the start)
    for prefix, mapped_prefix in CORE_COMPONENT_MAP.items():
        if new_key.startswith(prefix):
            new_key = new_key.replace(prefix, mapped_prefix, 1)
            break
    # Suffix mapping (replace only once at the end)
    for suffix, mapped_suffix in COMMON_SUFFIX_MAP.items():
        if new_key.endswith(suffix):
            new_key = new_key[: -len(suffix)] + mapped_suffix
            break
    return new_key


def _coerce_value_for_param(value: Any, current_raw_value: Any) -> Any:
    """Heuristically coerce a .vital value to something acceptable for plugin.parameters[*].raw_value.

    Rules:
    - booleans pass-through
    - ints/floats: if > 1 and <= 100, assume percent and divide by 100
    - strings and others: pass-through as-is (plugin will reject if incompatible)
    """
    # Preserve bools
    if isinstance(value, bool):
        return value

    # Normalize common percent-like values (0..100 -> 0..1)
    if isinstance(value, (int, float)):
        v = float(value)
        if v > 1.0 and v <= 100.0:
            return v / 100.0
        return v

    # Fallback: return unchanged
    return value


def convert_vital_preset_to_params(
    preset_path: str | Path, plugin: VST3Plugin
) -> dict[str, Any]:
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

    # Remap keys using rules from exploration
    remapped: Dict[str, Any] = {}
    for key, value in settings.items():
        new_key = _remap_key(key)
        remapped[new_key] = value

    plugin_params = set(plugin.parameters.keys())
    intersect = plugin_params.intersection(remapped.keys())
    if not intersect:
        logger.warning(
            "No overlapping parameters between .vital preset and plugin; check mapping rules."
        )

    result: Dict[str, Any] = {}
    for k in intersect:
        try:
            current_raw = plugin.parameters[k].raw_value
        except Exception:
            current_raw = None
        coerced = _coerce_value_for_param(remapped[k], current_raw)
        result[k] = coerced

    logger.info(
        f"Converted .vital preset: mapped {len(settings)} keys -> {len(remapped)}; "
        f"applying {len(result)} parameters recognized by the plugin."
    )
    return result
