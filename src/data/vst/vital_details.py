"""Vital engine parameter min/max ranges for normalization."""

from __future__ import annotations

from typing import Dict, Tuple

# Base (non-grouped) parameters.
BASE_DETAILS: Dict[str, Tuple[float, float]] = {
    "bypass": (0.0, 1.0),
    "beats_per_minute": (0.333333333, 5.0),
    "delay_dry_wet": (0.0, 1.0),
    "delay_feedback": (-1.0, 1.0),
    "delay_frequency": (-2.0, 9.0),
    "delay_aux_frequency": (-2.0, 9.0),
    "delay_on": (0.0, 1.0),
    "delay_style": (0.0, 3.0),
    "delay_filter_cutoff": (8.0, 136.0),
    "delay_filter_spread": (0.0, 1.0),
    "delay_sync": (0.0, 3.0),
    "delay_tempo": (4.0, 12.0),
    "delay_aux_sync": (0.0, 3.0),
    "delay_aux_tempo": (4.0, 12.0),
    "distortion_on": (0.0, 1.0),
    "distortion_type": (0.0, 5.0),
    "distortion_drive": (0.0, 20.0),
    "distortion_mix": (0.0, 1.0),
    "distortion_filter_order": (0.0, 2.0),
    "distortion_filter_cutoff": (8.0, 136.0),
    "distortion_filter_resonance": (0.0, 1.0),
    "distortion_filter_blend": (0.0, 2.0),
    "legato": (0.0, 1.0),
    "pitch_bend_range": (0.0, 48.0),
    "polyphony": (1.0, 32.0),
    "voice_tune": (-1.0, 1.0),
    "voice_transpose": (-48.0, 48.0),
    "voice_amplitude": (0.0, 1.0),
    "stereo_routing": (0.0, 1.0),
    "stereo_mode": (0.0, 1.0),
    "portamento_time": (-10.0, 4.0),
    "portamento_slope": (-8.0, 8.0),
    "portamento_force": (0.0, 1.0),
    "portamento_scale": (0.0, 1.0),
    "reverb_pre_low_cutoff": (0.0, 128.0),
    "reverb_pre_high_cutoff": (0.0, 128.0),
    "reverb_low_shelf_cutoff": (0.0, 128.0),
    "reverb_low_shelf_gain": (-6.0, 0.0),
    "reverb_high_shelf_cutoff": (0.0, 128.0),
    "reverb_high_shelf_gain": (-6.0, 0.0),
    "reverb_dry_wet": (0.0, 1.0),
    "reverb_delay": (0.0, 0.3),
    "reverb_decay_time": (-6.0, 6.0),
    "reverb_size": (0.0, 1.0),
    "reverb_chorus_amount": (0.0, 1.0),
    "reverb_chorus_frequency": (-8.0, 3.0),
    "reverb_on": (0.0, 1.0),
    "sub_on": (0.0, 1.0),
    "sub_direct_out": (0.0, 1.0),
    "sub_transpose": (-48.0, 48.0),
    "sub_transpose_quantize": (0.0, 8191.0),
    "sub_tune": (-1.0, 1.0),
    "sub_level": (0.0, 1.0),
    "sub_pan": (-1.0, 1.0),
    "sub_waveform": (0.0, 5.0),
    "sample_on": (0.0, 1.0),
    "sample_random_phase": (0.0, 1.0),
    "sample_keytrack": (0.0, 1.0),
    "sample_loop": (0.0, 1.0),
    "sample_bounce": (0.0, 1.0),
    "sample_transpose": (-48.0, 48.0),
    "sample_transpose_quantize": (0.0, 8191.0),
    "sample_tune": (-1.0, 1.0),
    "sample_level": (0.0, 1.0),
    "sample_destination": (0.0, 14.0),
    "sample_pan": (-1.0, 1.0),
    "velocity_track": (-1.0, 1.0),
    "volume": (0.0, 7399.4404),
    "phaser_on": (0.0, 1.0),
    "phaser_dry_wet": (0.0, 1.0),
    "phaser_feedback": (0.0, 1.0),
    "phaser_frequency": (-5.0, 2.0),
    "phaser_sync": (0.0, 3.0),
    "phaser_tempo": (0.0, 10.0),
    "phaser_center": (8.0, 136.0),
    "phaser_blend": (0.0, 2.0),
    "phaser_mod_depth": (0.0, 48.0),
    "phaser_phase_offset": (0.0, 1.0),
    "flanger_on": (0.0, 1.0),
    "flanger_dry_wet": (0.0, 0.5),
    "flanger_feedback": (-1.0, 1.0),
    "flanger_frequency": (-5.0, 2.0),
    "flanger_sync": (0.0, 3.0),
    "flanger_tempo": (0.0, 10.0),
    "flanger_center": (8.0, 136.0),
    "flanger_mod_depth": (0.0, 1.0),
    "flanger_phase_offset": (0.0, 1.0),
    "chorus_on": (0.0, 1.0),
    "chorus_dry_wet": (0.0, 1.0),
    "chorus_feedback": (-0.95, 0.95),
    "chorus_cutoff": (8.0, 136.0),
    "chorus_spread": (0.0, 1.0),
    "chorus_voices": (1.0, 4.0),
    "chorus_frequency": (-6.0, 3.0),
    "chorus_sync": (0.0, 3.0),
    "chorus_tempo": (0.0, 10.0),
    "chorus_mod_depth": (0.0, 1.0),
    "chorus_delay_1": (-10.0, -5.64386),
    "chorus_delay_2": (-10.0, -5.64386),
    "compressor_on": (0.0, 1.0),
    "compressor_low_upper_threshold": (-80.0, 0.0),
    "compressor_band_upper_threshold": (-80.0, 0.0),
    "compressor_high_upper_threshold": (-80.0, 0.0),
    "compressor_low_lower_threshold": (-80.0, 0.0),
    "compressor_band_lower_threshold": (-80.0, 0.0),
    "compressor_high_lower_threshold": (-80.0, 0.0),
    "compressor_low_upper_ratio": (0.0, 1.0),
    "compressor_band_upper_ratio": (0.0, 1.0),
    "compressor_high_upper_ratio": (0.0, 1.0),
    "compressor_low_lower_ratio": (-1.0, 1.0),
    "compressor_band_lower_ratio": (-1.0, 1.0),
    "compressor_high_lower_ratio": (-1.0, 1.0),
    "compressor_low_gain": (-30.0, 30.0),
    "compressor_band_gain": (-30.0, 30.0),
    "compressor_high_gain": (-30.0, 30.0),
    "compressor_attack": (0.0, 1.0),
    "compressor_release": (0.0, 1.0),
    "compressor_enabled_bands": (0.0, 2.0),
    "compressor_mix": (0.0, 1.0),
	"eq_on": (0.0, 1.0),
	"eq_low_mode": (0.0, 1.0),
	"eq_low_cutoff": (8.0, 136.0),
	"eq_low_gain": (-30.0, 30.0),
	"eq_low_resonance": (0.0, 1.0),
	"eq_band_mode": (0.0, 1.0),
	"eq_band_cutoff": (8.0, 136.0),
	"eq_band_gain": (-30.0, 30.0),
	"eq_band_resonance": (0.0, 1.0),
	"eq_high_mode": (0.0, 1.0),
	"eq_high_cutoff": (8.0, 136.0),
	"eq_high_gain": (-30.0, 30.0),
	"eq_high_resonance": (0.0, 1.0),
	"effect_chain_order": (0.0, 362879.0),
	"voice_priority": (0.0, 3.0),
	"voice_override": (0.0, 3.0),
	"oversampling": (0.0, 3.0),
	"pitch_wheel": (-1.0, 1.0),
	"mod_wheel": (0.0, 1.0),
	"mpe_enabled": (0.0, 1.0),
	"view_spectrogram": (0.0, 2.0),
	"chorus_filter_cutoff": (8.0, 136.0),
	"reverb_low_cutoff": (8.0, 136.0),
	"reverb_high_cutoff": (8.0, 136.0),
	"filter_fx_cutoff": (8.0, 136.0),
	"filter_fx_resonance": (0.0, 1.0),
	"filter_fx_drive": (0.0, 20.0),
	"filter_fx_blend": (0.0, 2.0),
	"filter_fx_style": (0.0, 9.0),
	"filter_fx_formant_x": (0.0, 1.0),
	"filter_fx_formant_y": (0.0, 1.0),
	"filter_fx_formant_transpose": (-12.0, 12.0),
	"filter_fx_formant_resonance": (0.3, 1.0),
	"filter_fx_formant_spread": (-1.0, 1.0),
	"lfo_1_delay": (0.0, 1.4142135624),
	"lfo_2_delay": (0.0, 1.4142135624),
	"lfo_3_delay": (0.0, 1.4142135624),
	"lfo_4_delay": (0.0, 1.4142135624),
	"lfo_5_delay": (0.0, 1.4142135624),
	"lfo_6_delay": (0.0, 1.4142135624),
	"lfo_7_delay": (0.0, 1.4142135624),
	"lfo_8_delay": (0.0, 1.4142135624),
	"lfo_1_transpose": (-48.0, 48.0),
	"lfo_2_transpose": (-48.0, 48.0),
	"lfo_3_transpose": (-48.0, 48.0),
	"lfo_4_transpose": (-48.0, 48.0),
	"lfo_5_transpose": (-48.0, 48.0),
	"lfo_6_transpose": (-48.0, 48.0),
	"lfo_7_transpose": (-48.0, 48.0),
	"lfo_8_transpose": (-48.0, 48.0),
	"lfo_1_tune": (-1.0, 1.0),
	"lfo_2_tune": (-1.0, 1.0),
	"lfo_3_tune": (-1.0, 1.0),
	"lfo_4_tune": (-1.0, 1.0),
	"lfo_5_tune": (-1.0, 1.0),
	"lfo_6_tune": (-1.0, 1.0),
	"lfo_7_tune": (-1.0, 1.0),
	"lfo_8_tune": (-1.0, 1.0),
	"osc_1_blend": (0.0, 2.0),
	"osc_2_blend": (0.0, 2.0),
	"osc_3_blend": (0.0, 2.0),
	"macro_control_1": (0.0, 1.0),
	"macro_control_2": (0.0, 1.0),
	"macro_control_3": (0.0, 1.0),
	"macro_control_4": (0.0, 1.0),
	"band_lower_ratio": (-1.0, 1.0),
	"band_lower_threshold": (-80.0, 0.0),
	"band_upper_ratio": (0.0, 1.0),
	"band_upper_threshold": (-80.0, 0.0),
	"high_lower_ratio": (-1.0, 1.0),
	"high_lower_threshold": (-80.0, 0.0),
	"high_upper_ratio": (0.0, 1.0),
	"high_upper_threshold": (-80.0, 0.0),
	"low_lower_ratio": (-1.0, 1.0),
	"low_lower_threshold": (-80.0, 0.0),
	"low_upper_ratio": (0.0, 1.0),
	"low_upper_threshold": (-80.0, 0.0),
	"macro_1": (0.0, 1.0),
	"macro_2": (0.0, 1.0),
	"macro_3": (0.0, 1.0),
	"macro_4": (0.0, 1.0),
	"chorus_filter_spread": (0.0, 1.0),
	"delay_frequency_2": (-2.0, 9.0),
	"delay_sync_2": (0.0, 3.0),
	"delay_tempo_2": (4.0, 12.0),
	"filter_1_comb_blend_offset": (0.0, 1.0),
	"filter_1_key_track": (-1.0, 1.0),
	"filter_1_mix": (0.0, 1.0),
	"filter_2_comb_blend_offset": (0.0, 1.0),
	"filter_2_key_track": (-1.0, 1.0),
	"filter_2_mix": (0.0, 1.0),
	"filter_fx_comb_blend_offset": (0.0, 1.0),
	"filter_fx_key_track": (-1.0, 1.0),
	"filter_fx_mix": (0.0, 1.0),
	"filter_fx_model": (0.0, 7.0),
	"filter_fx_switch": (0.0, 1.0),
	"note_start_and_end": (0.0, 1.0),
	"oscillator_1_frequency_morph_phase": (0.0, 1.0),
	"oscillator_1_unison_frame_spread": (-128.0, 128.0),
	"oscillator_2_frequency_morph_phase": (0.0, 1.0),
	"oscillator_2_unison_frame_spread": (-128.0, 128.0),
	"oscillator_3_frequency_morph_phase": (0.0, 1.0),
	"oscillator_3_unison_frame_spread": (-128.0, 128.0),
	"pitch": (-1.0, 1.0),
	"random_lfo_1_transpose": (-60.0, 36.0),
	"random_lfo_1_tune": (-1.0, 1.0),
	"random_lfo_2_transpose": (-60.0, 36.0),
	"random_lfo_2_tune": (-1.0, 1.0),
	"random_lfo_3_transpose": (-60.0, 36.0),
	"random_lfo_3_tune": (-1.0, 1.0),
	"random_lfo_4_transpose": (-60.0, 36.0),
	"random_lfo_4_tune": (-1.0, 1.0),
	"reverb_high_gain": (-6.0, 0.0),
	"reverb_low_gain": (-6.0, 0.0),
}

ENV_DETAILS = {
	"delay": (0.0, 1.4142135624),
	"attack": (0.0, 2.37842),
	"hold": (0.0, 1.4142135624),
	"decay": (0.0, 2.37842),
	"release": (0.0, 2.37842),
	"attack_power": (-20.0, 20.0),
	"decay_power": (-20.0, 20.0),
	"release_power": (-20.0, 20.0),
	"sustain": (0.0, 1.0),
}

LFO_DETAILS = {
	"phase": (0.0, 1.0),
	"sync_type": (0.0, 3.0),
	"frequency": (-7.0, 9.0),
	"sync": (0.0, 3.0),
	"tempo": (0.0, 12.0),
	"fade_time": (0.0, 8.0),
	"smooth_mode": (0.0, 1.0),
	"smooth_time": (-10.0, 4.0),
	"delay_time": (0.0, 4.0),
	"stereo": (-0.5, 0.5),
	"keytrack_transpose": (-60.0, 36.0),
	"keytrack_tune": (-1.0, 1.0),
}

RANDOM_LFO_DETAILS = {
	"style": (0.0, 5.0),
	"frequency": (-7.0, 9.0),
	"sync": (0.0, 3.0),
	"tempo": (0.0, 12.0),
	"stereo": (0.0, 1.0),
	"sync_type": (0.0, 1.0),
	"keytrack_transpose": (-60.0, 36.0),
	"keytrack_tune": (-1.0, 1.0),
}

FILTER_DETAILS = {
	"mix": (0.0, 1.0),
	"cutoff": (8.0, 136.0),
	"resonance": (0.0, 1.0),
	"drive": (0.0, 20.0),
	"blend": (0.0, 2.0),
	"style": (0.0, 9.0),
	"model": (0.0, 7.0),
	"on": (0.0, 1.0),
	"blend_transpose": (0.0, 84.0),
	"keytrack": (-1.0, 1.0),
	"formant_x": (0.0, 1.0),
	"formant_y": (0.0, 1.0),
	"formant_transpose": (-12.0, 12.0),
	"formant_resonance": (0.3, 1.0),
	"formant_spread": (-1.0, 1.0),
	"osc1_input": (0.0, 1.0),
	"osc2_input": (0.0, 1.0),
	"osc3_input": (0.0, 1.0),
	"sample_input": (0.0, 1.0),
	"filter_input": (0.0, 1.0),
}

OSC_DETAILS = {
	"on": (0.0, 1.0),
	"transpose": (-48.0, 48.0),
	"transpose_quantize": (0.0, 8191.0),
	"tune": (-1.0, 1.0),
	"pan": (-1.0, 1.0),
	"stack_style": (0.0, 3.0),
	"unison_detune": (0.0, 10.0),
	"unison_voices": (1.0, 16.0),
	"unison_blend": (0.0, 1.0),
	"detune_power": (-5.0, 5.0),
	"detune_range": (0.0, 48.0),
	"level": (0.0, 1.0),
	"midi_track": (0.0, 1.0),
	"smooth_interpolation": (0.0, 1.0),
	"spectral_unison": (0.0, 1.0),
	"wave_frame": (0.0, 256.0),
	"frame_spread": (-128.0, 128.0),
	"stereo_spread": (0.0, 1.0),
	"phase": (0.0, 1.0),
	"distortion_phase": (0.0, 1.0),
	"random_phase": (0.0, 1.0),
	"distortion_type": (0.0, 5.0),
	"distortion_amount": (0.0, 1.0),
	"distortion_spread": (-0.5, 0.5),
	"spectral_morph_type": (0.0, 5.0),
	"spectral_morph_amount": (0.0, 1.0),
	"spectral_morph_spread": (-0.5, 0.5),
	"destination": (0.0, 14.0),
	"view_2d": (0.0, 2.0),
}

MOD_DETAILS = {
	"amount": (-1.0, 1.0),
	"power": (-10.0, 10.0),
	"bipolar": (0.0, 1.0),
	"stereo": (0.0, 1.0),
	"bypass": (0.0, 1.0),
}

GROUP_PREFIX_MAP = {
	"envelope": ("env", ENV_DETAILS),
	"lfo": ("lfo", LFO_DETAILS),
	"random_lfo": ("random", RANDOM_LFO_DETAILS),
	"oscillator": ("osc", OSC_DETAILS),
	"filter": ("filter", FILTER_DETAILS),
	"modulation": ("modulation", MOD_DETAILS),
}

SYNONYMS: dict[str, str] = {
	"delay_switch": "delay_on",
	"delay_mix": "delay_dry_wet",
	"delay_aux_frequency": "delay_frequency_2",
	"delay_aux_sync": "delay_sync_2",
	"delay_aux_tempo": "delay_tempo_2",
	"chorus_switch": "chorus_on",
	"chorus_mix": "chorus_dry_wet",
	"flanger_switch": "flanger_on",
	"flanger_mix": "flanger_dry_wet",
	"phaser_switch": "phaser_on",
	"phaser_mix": "phaser_dry_wet",
	"distortion_switch": "distortion_on",
	"compressor_switch": "compressor_on",
	"eq_switch": "eq_on",
	"reverb_switch": "reverb_on",
	"reverb_mix": "reverb_dry_wet",
	"sample_switch": "sample_on",
	"filter_1_keytrack": "filter_1_key_track",
	"filter_2_keytrack": "filter_2_key_track",
	"filter_fx_keytrack": "filter_fx_key_track",
	"lfo_1_delay_time": "lfo_1_delay",
	"lfo_2_delay_time": "lfo_2_delay",
	"lfo_3_delay_time": "lfo_3_delay",
	"lfo_4_delay_time": "lfo_4_delay",
	"lfo_5_delay_time": "lfo_5_delay",
	"lfo_6_delay_time": "lfo_6_delay",
	"lfo_7_delay_time": "lfo_7_delay",
	"lfo_8_delay_time": "lfo_8_delay",
}

def plugin_name_to_internal(name: str) -> str:
	if name in SYNONYMS:
		return SYNONYMS[name]
	if name in BASE_DETAILS:
		return name
	if name.startswith("macro_"):
		parts = name.split("_")
		if len(parts) == 2 and parts[1].isdigit():
			return f"macro_control_{parts[1]}"
	for plugin_prefix, (internal_prefix, _) in GROUP_PREFIX_MAP.items():
		if name.startswith(plugin_prefix + "_"):
			rest = name[len(plugin_prefix) + 1 :]
			if "_" in rest:
				idx, tail = rest.split("_", 1)
				if idx.isdigit():
					tail = tail.replace("switch", "on")
					tail = tail.replace("mix", "dry_wet")
					tail = tail.replace("fade_in", "fade_time")
					tail = tail.replace("phase_randomization", "random_phase")
					tail = tail.replace("frequency_morph", "spectral_morph")
					return f"{internal_prefix}_{idx}_{tail}"
	parts = name.split("_")
	if len(parts) >= 3 and parts[1].isdigit():
		group = parts[0]
		if group in GROUP_PREFIX_MAP:
			internal_prefix, _ = GROUP_PREFIX_MAP[group]
			idx = parts[1]
			tail = "_".join(parts[2:])
			tail = tail.replace("switch", "on")
			tail = tail.replace("mix", "dry_wet")
			tail = tail.replace("fade_in", "fade_time")
			tail = tail.replace("phase_randomization", "random_phase")
			tail = tail.replace("frequency_morph", "spectral_morph")
			return f"{internal_prefix}_{idx}_{tail}"
	return name

def get_min_max_for_plugin_name(name: str) -> Tuple[float, float] | None:
	if name in BASE_DETAILS:
		return BASE_DETAILS[name]
	internal = plugin_name_to_internal(name)
	if internal in BASE_DETAILS:
		return BASE_DETAILS[internal]
	if internal == name:
		parts = internal.split("_")
		if len(parts) >= 3 and parts[1].isdigit():
			base = "_".join(parts[2:])
			prefix = parts[0]
			for plugin_prefix, (internal_prefix, details) in GROUP_PREFIX_MAP.items():
				if prefix == internal_prefix and base in details:
					return details[base]
		return BASE_DETAILS.get(internal)
	parts = internal.split("_")
	if len(parts) >= 3 and parts[1].isdigit():
		base = "_".join(parts[2:])
		internal_prefix = parts[0]
		for plugin_prefix, (prefix_match, details) in GROUP_PREFIX_MAP.items():
			if internal_prefix == prefix_match and base in details:
				return details[base]
	return None
