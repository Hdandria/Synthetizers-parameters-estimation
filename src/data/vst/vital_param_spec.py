from src.data.vst.param_spec import (
    CategoricalParameter,
    ContinuousParameter,
    DiscreteLiteralParameter,
    NoteDurationParameter,
    ParamSpec,
)


def _evenly_spaced(n: int) -> list[float]:
    if n <= 1:
        return [0.0]
    if n == 2:
        return [0.0, 1.0]
    step = 1.0 / (n - 1)
    return [round(i * step, 6) for i in range(n)]


VITAL_PARAM_SPEC = ParamSpec(
    [
        # Global / transport
        ContinuousParameter(name="beats_per_minute", min=0.0, max=1.0),
        ContinuousParameter(name="volume", min=0.0, max=1.0),
        CategoricalParameter(
            name="bypass", values=[0, 1], raw_values=[0.0, 1.0], encoding="onehot"
        ),

        # Delay
        CategoricalParameter(
            name="delay_switch", values=[False, True], raw_values=[0.0, 1.0], encoding="onehot"
        ),
        ContinuousParameter(name="delay_mix", min=0.0, max=1.0),
        ContinuousParameter(name="delay_feedback", min=0.0, max=1.0),
        ContinuousParameter(name="delay_filter_cutoff", min=0.0, max=1.0),
        ContinuousParameter(name="delay_filter_spread", min=0.0, max=1.0),
        ContinuousParameter(name="delay_frequency", min=0.0, max=1.0),
        CategoricalParameter(
            name="delay_style", values=list(range(4)), raw_values=_evenly_spaced(4), encoding="onehot"
        ),
        CategoricalParameter(
            name="delay_sync", values=list(range(4)), raw_values=_evenly_spaced(4), encoding="onehot"
        ),
        CategoricalParameter(
            name="delay_tempo", values=list(range(9)), raw_values=_evenly_spaced(9), encoding="onehot"
        ),
        # Dual/aux delay
        ContinuousParameter(name="delay_frequency_2", min=0.0, max=1.0),
        CategoricalParameter(
            name="delay_sync_2", values=list(range(4)), raw_values=_evenly_spaced(4), encoding="onehot"
        ),
        CategoricalParameter(
            name="delay_tempo_2", values=list(range(9)), raw_values=_evenly_spaced(9), encoding="onehot"
        ),

        # Chorus
        CategoricalParameter(
            name="chorus_switch", values=[False, True], raw_values=[0.0, 1.0], encoding="onehot"
        ),
        ContinuousParameter(name="chorus_mix", min=0.0, max=1.0),
        ContinuousParameter(name="chorus_feedback", min=0.0, max=1.0),
        ContinuousParameter(name="chorus_filter_cutoff", min=0.0, max=1.0),
        ContinuousParameter(name="chorus_frequency", min=0.0, max=1.0),
        CategoricalParameter(
            name="chorus_sync", values=list(range(4)), raw_values=_evenly_spaced(4), encoding="onehot"
        ),
        CategoricalParameter(
            name="chorus_tempo", values=list(range(11)), raw_values=_evenly_spaced(11), encoding="onehot"
        ),
        ContinuousParameter(name="chorus_mod_depth", min=0.0, max=1.0),
        ContinuousParameter(name="chorus_delay_1", min=0.0, max=1.0),
        ContinuousParameter(name="chorus_delay_2", min=0.0, max=1.0),
        ContinuousParameter(name="chorus_filter_spread", min=0.0, max=1.0),
        CategoricalParameter(
            name="chorus_voices", values=[4, 8, 12, 16], raw_values=_evenly_spaced(4), encoding="onehot"
        ),

        # Flanger
        CategoricalParameter(
            name="flanger_switch", values=[False, True], raw_values=[0.0, 1.0], encoding="onehot"
        ),
        ContinuousParameter(name="flanger_mix", min=0.0, max=1.0),
        ContinuousParameter(name="flanger_feedback", min=0.0, max=1.0),
        ContinuousParameter(name="flanger_frequency", min=0.0, max=1.0),
        ContinuousParameter(name="flanger_mod_depth", min=0.0, max=1.0),
        ContinuousParameter(name="flanger_phase_offset", min=0.0, max=1.0),
        CategoricalParameter(
            name="flanger_sync", values=list(range(4)), raw_values=_evenly_spaced(4), encoding="onehot"
        ),
        CategoricalParameter(
            name="flanger_tempo", values=list(range(11)), raw_values=_evenly_spaced(11), encoding="onehot"
        ),
        ContinuousParameter(name="flanger_center", min=0.0, max=1.0),

        # Phaser
        CategoricalParameter(
            name="phaser_switch", values=[False, True], raw_values=[0.0, 1.0], encoding="onehot"
        ),
        ContinuousParameter(name="phaser_mix", min=0.0, max=1.0),
        ContinuousParameter(name="phaser_feedback", min=0.0, max=1.0),
        ContinuousParameter(name="phaser_frequency", min=0.0, max=1.0),
        CategoricalParameter(
            name="phaser_sync", values=list(range(4)), raw_values=_evenly_spaced(4), encoding="onehot"
        ),
        CategoricalParameter(
            name="phaser_tempo", values=list(range(11)), raw_values=_evenly_spaced(11), encoding="onehot"
        ),
        ContinuousParameter(name="phaser_center", min=0.0, max=1.0),
        ContinuousParameter(name="phaser_blend", min=0.0, max=1.0),
        ContinuousParameter(name="phaser_mod_depth", min=0.0, max=1.0),
        ContinuousParameter(name="phaser_phase_offset", min=0.0, max=1.0),

        # Distortion
        CategoricalParameter(
            name="distortion_switch", values=[False, True], raw_values=[0.0, 1.0], encoding="onehot"
        ),
        CategoricalParameter(
            name="distortion_type", values=list(range(6)), raw_values=_evenly_spaced(6), encoding="onehot"
        ),
        ContinuousParameter(name="distortion_drive", min=0.0, max=1.0),
        ContinuousParameter(name="distortion_mix", min=0.0, max=1.0),
        CategoricalParameter(
            name="distortion_filter_order", values=list(range(3)), raw_values=_evenly_spaced(3), encoding="onehot"
        ),
        ContinuousParameter(name="distortion_filter_cutoff", min=0.0, max=1.0),
        ContinuousParameter(name="distortion_filter_resonance", min=0.0, max=1.0),
        ContinuousParameter(name="distortion_filter_blend", min=0.0, max=1.0),

        # Compressor
        CategoricalParameter(
            name="compressor_switch", values=[False, True], raw_values=[0.0, 1.0], encoding="onehot"
        ),
        ContinuousParameter(name="compressor_attack", min=0.0, max=1.0),
        ContinuousParameter(name="compressor_release", min=0.0, max=1.0),
        CategoricalParameter(
            name="compressor_enabled_bands", values=list(range(4)), raw_values=_evenly_spaced(4), encoding="onehot"
        ),
        ContinuousParameter(name="compressor_low_gain", min=0.0, max=1.0),
        ContinuousParameter(name="compressor_band_gain", min=0.0, max=1.0),
        ContinuousParameter(name="compressor_high_gain", min=0.0, max=1.0),
        ContinuousParameter(name="low_upper_threshold", min=0.0, max=1.0),
        ContinuousParameter(name="band_upper_threshold", min=0.0, max=1.0),
        ContinuousParameter(name="high_upper_threshold", min=0.0, max=1.0),
        ContinuousParameter(name="low_lower_threshold", min=0.0, max=1.0),
        ContinuousParameter(name="band_lower_threshold", min=0.0, max=1.0),
        ContinuousParameter(name="high_lower_threshold", min=0.0, max=1.0),
        ContinuousParameter(name="low_upper_ratio", min=0.0, max=1.0),
        ContinuousParameter(name="band_upper_ratio", min=0.0, max=1.0),
        ContinuousParameter(name="high_upper_ratio", min=0.0, max=1.0),
        ContinuousParameter(name="low_lower_ratio", min=0.0, max=1.0),
        ContinuousParameter(name="band_lower_ratio", min=0.0, max=1.0),
        ContinuousParameter(name="high_lower_ratio", min=0.0, max=1.0),
        ContinuousParameter(name="compressor_mix", min=0.0, max=1.0),

        # EQ
        CategoricalParameter(
            name="eq_switch", values=[False, True], raw_values=[0.0, 1.0], encoding="onehot"
        ),
        CategoricalParameter(
            name="eq_low_mode", values=[0, 1], raw_values=[0.0, 1.0], encoding="onehot"
        ),
        ContinuousParameter(name="eq_low_cutoff", min=0.0, max=1.0),
        ContinuousParameter(name="eq_low_gain", min=0.0, max=1.0),
        ContinuousParameter(name="eq_low_resonance", min=0.0, max=1.0),
        CategoricalParameter(
            name="eq_band_mode", values=[0, 1], raw_values=[0.0, 1.0], encoding="onehot"
        ),
        ContinuousParameter(name="eq_band_cutoff", min=0.0, max=1.0),
        ContinuousParameter(name="eq_band_gain", min=0.0, max=1.0),
        ContinuousParameter(name="eq_band_resonance", min=0.0, max=1.0),
        CategoricalParameter(
            name="eq_high_mode", values=[0, 1], raw_values=[0.0, 1.0], encoding="onehot"
        ),
        ContinuousParameter(name="eq_high_cutoff", min=0.0, max=1.0),
        ContinuousParameter(name="eq_high_gain", min=0.0, max=1.0),
        ContinuousParameter(name="eq_high_resonance", min=0.0, max=1.0),

        # Reverb
        CategoricalParameter(
            name="reverb_switch", values=[False, True], raw_values=[0.0, 1.0], encoding="onehot"
        ),
        ContinuousParameter(name="reverb_mix", min=0.0, max=1.0),
        ContinuousParameter(name="reverb_delay", min=0.0, max=1.0),
        ContinuousParameter(name="reverb_decay_time", min=0.0, max=1.0),
        ContinuousParameter(name="reverb_size", min=0.0, max=1.0),
        ContinuousParameter(name="reverb_low_cutoff", min=0.0, max=1.0),
        ContinuousParameter(name="reverb_low_gain", min=0.0, max=1.0),
        ContinuousParameter(name="reverb_high_cutoff", min=0.0, max=1.0),
        ContinuousParameter(name="reverb_high_gain", min=0.0, max=1.0),
        ContinuousParameter(name="reverb_pre_low_cutoff", min=0.0, max=1.0),
        ContinuousParameter(name="reverb_pre_high_cutoff", min=0.0, max=1.0),
        ContinuousParameter(name="reverb_chorus_amount", min=0.0, max=1.0),
        ContinuousParameter(name="reverb_chorus_frequency", min=0.0, max=1.0),

        # Filters (1, 2, fx)
        # -- Filter 1
        CategoricalParameter(
            name="filter_1_switch", values=[False, True], raw_values=[0.0, 1.0], encoding="onehot"
        ),
        ContinuousParameter(name="filter_1_mix", min=0.0, max=1.0),
        ContinuousParameter(name="filter_1_cutoff", min=0.0, max=1.0),
        ContinuousParameter(name="filter_1_resonance", min=0.0, max=1.0),
        ContinuousParameter(name="filter_1_drive", min=0.0, max=1.0),
        ContinuousParameter(name="filter_1_blend", min=0.0, max=1.0),
        CategoricalParameter(
            name="filter_1_style", values=list(range(6)), raw_values=_evenly_spaced(6), encoding="onehot"
        ),
        CategoricalParameter(
            name="filter_1_model", values=list(range(8)), raw_values=_evenly_spaced(8), encoding="onehot"
        ),
        CategoricalParameter(
            name="filter_1_filter_input", values=[False, True], raw_values=[0.0, 1.0], encoding="onehot"
        ),
        ContinuousParameter(name="filter_1_key_track", min=0.0, max=1.0),
        ContinuousParameter(name="filter_1_formant_x", min=0.0, max=1.0),
        ContinuousParameter(name="filter_1_formant_y", min=0.0, max=1.0),
        ContinuousParameter(name="filter_1_formant_transpose", min=0.0, max=1.0),
        ContinuousParameter(name="filter_1_formant_resonance", min=0.0, max=1.0),
        ContinuousParameter(name="filter_1_formant_spread", min=0.0, max=1.0),
        ContinuousParameter(name="filter_1_comb_blend_offset", min=0.0, max=1.0),

        # -- Filter 2
        CategoricalParameter(
            name="filter_2_switch", values=[False, True], raw_values=[0.0, 1.0], encoding="onehot"
        ),
        ContinuousParameter(name="filter_2_mix", min=0.0, max=1.0),
        ContinuousParameter(name="filter_2_cutoff", min=0.0, max=1.0),
        ContinuousParameter(name="filter_2_resonance", min=0.0, max=1.0),
        ContinuousParameter(name="filter_2_drive", min=0.0, max=1.0),
        ContinuousParameter(name="filter_2_blend", min=0.0, max=1.0),
        CategoricalParameter(
            name="filter_2_style", values=list(range(6)), raw_values=_evenly_spaced(6), encoding="onehot"
        ),
        CategoricalParameter(
            name="filter_2_model", values=list(range(8)), raw_values=_evenly_spaced(8), encoding="onehot"
        ),
        CategoricalParameter(
            name="filter_2_filter_input", values=[False, True], raw_values=[0.0, 1.0], encoding="onehot"
        ),
        ContinuousParameter(name="filter_2_key_track", min=0.0, max=1.0),
        ContinuousParameter(name="filter_2_formant_x", min=0.0, max=1.0),
        ContinuousParameter(name="filter_2_formant_y", min=0.0, max=1.0),
        ContinuousParameter(name="filter_2_formant_transpose", min=0.0, max=1.0),
        ContinuousParameter(name="filter_2_formant_resonance", min=0.0, max=1.0),
        ContinuousParameter(name="filter_2_formant_spread", min=0.0, max=1.0),
        ContinuousParameter(name="filter_2_comb_blend_offset", min=0.0, max=1.0),

        # -- Filter FX
        CategoricalParameter(
            name="filter_fx_switch", values=[False, True], raw_values=[0.0, 1.0], encoding="onehot"
        ),
        ContinuousParameter(name="filter_fx_mix", min=0.0, max=1.0),
        ContinuousParameter(name="filter_fx_cutoff", min=0.0, max=1.0),
        ContinuousParameter(name="filter_fx_resonance", min=0.0, max=1.0),
        ContinuousParameter(name="filter_fx_drive", min=0.0, max=1.0),
        ContinuousParameter(name="filter_fx_blend", min=0.0, max=1.0),
        CategoricalParameter(
            name="filter_fx_style", values=list(range(6)), raw_values=_evenly_spaced(6), encoding="onehot"
        ),
        CategoricalParameter(
            name="filter_fx_model", values=list(range(8)), raw_values=_evenly_spaced(8), encoding="onehot"
        ),
        ContinuousParameter(name="filter_fx_key_track", min=0.0, max=1.0),
        ContinuousParameter(name="filter_fx_formant_x", min=0.0, max=1.0),
        ContinuousParameter(name="filter_fx_formant_y", min=0.0, max=1.0),
        ContinuousParameter(name="filter_fx_formant_transpose", min=0.0, max=1.0),
        ContinuousParameter(name="filter_fx_formant_resonance", min=0.0, max=1.0),
        ContinuousParameter(name="filter_fx_formant_spread", min=0.0, max=1.0),
        ContinuousParameter(name="filter_fx_comb_blend_offset", min=0.0, max=1.0),

        # Envelopes (1..6)
        # Attack/decay/sustain/release, powers, plus delay/hold where available.
        *[
            ContinuousParameter(name=f"envelope_{i}_delay", min=0.0, max=1.0)
            for i in range(1, 7)
        ],
        *[
            ContinuousParameter(name=f"envelope_{i}_attack", min=0.0, max=1.0)
            for i in range(1, 7)
        ],
        *[
            ContinuousParameter(name=f"envelope_{i}_hold", min=0.0, max=1.0)
            for i in range(1, 7)
        ],
        *[
            ContinuousParameter(name=f"envelope_{i}_decay", min=0.0, max=1.0)
            for i in range(1, 7)
        ],
        *[
            ContinuousParameter(name=f"envelope_{i}_release", min=0.0, max=1.0)
            for i in range(1, 7)
        ],
        *[
            ContinuousParameter(name=f"envelope_{i}_attack_power", min=0.0, max=1.0)
            for i in range(1, 7)
        ],
        *[
            ContinuousParameter(name=f"envelope_{i}_decay_power", min=0.0, max=1.0)
            for i in range(1, 7)
        ],
        *[
            ContinuousParameter(name=f"envelope_{i}_release_power", min=0.0, max=1.0)
            for i in range(1, 7)
        ],
        *[
            ContinuousParameter(name=f"envelope_{i}_sustain", min=0.0, max=1.0)
            for i in range(1, 7)
        ],

        # LFOs (1..8)
        *[
            ContinuousParameter(name=f"lfo_{i}_phase", min=0.0, max=1.0)
            for i in range(1, 9)
        ],
        *[
            CategoricalParameter(
                name=f"lfo_{i}_sync_type",
                values=list(range(7)),
                raw_values=_evenly_spaced(7),
                encoding="onehot",
            )
            for i in range(1, 5)
        ],
        *[
            ContinuousParameter(name=f"lfo_{i}_frequency", min=0.0, max=1.0)
            for i in range(1, 9)
        ],
        *[
            CategoricalParameter(
                name=f"lfo_{i}_sync",
                values=list(range(5)),
                raw_values=_evenly_spaced(5),
                encoding="onehot",
            )
            for i in range(1, 9)
        ],
        *[
            CategoricalParameter(
                name=f"lfo_{i}_tempo",
                values=list(range(13)),
                raw_values=_evenly_spaced(13),
                encoding="onehot",
            )
            for i in range(1, 9)
        ],
        *[
            ContinuousParameter(name=f"lfo_{i}_fade_in", min=0.0, max=1.0)
            for i in range(1, 9)
        ],
        *[
            ContinuousParameter(name=f"lfo_{i}_delay", min=0.0, max=1.0)
            for i in range(1, 9)
        ],
        *[
            CategoricalParameter(
                name=f"lfo_{i}_smooth_mode",
                values=[False, True],
                raw_values=[0.0, 1.0],
                encoding="onehot",
            )
            for i in range(1, 9)
        ],
        *[
            ContinuousParameter(name=f"lfo_{i}_smooth_time", min=0.0, max=1.0)
            for i in range(1, 9)
        ],
        *[
            ContinuousParameter(name=f"lfo_{i}_stereo", min=0.0, max=1.0)
            for i in range(1, 9)
        ],
        *[
            ContinuousParameter(name=f"lfo_{i}_transpose", min=0.0, max=1.0)
            for i in range(1, 9)
        ],
        *[
            ContinuousParameter(name=f"lfo_{i}_tune", min=0.0, max=1.0)
            for i in range(1, 9)
        ],

        # Random LFOs (1..4)
        *[
            CategoricalParameter(
                name=f"random_lfo_{i}_style",
                values=list(range(4)),
                raw_values=_evenly_spaced(4),
                encoding="onehot",
            )
            for i in range(1, 5)
        ],
        *[
            ContinuousParameter(name=f"random_lfo_{i}_frequency", min=0.0, max=1.0)
            for i in range(1, 5)
        ],
        *[
            CategoricalParameter(
                name=f"random_lfo_{i}_sync",
                values=list(range(5)),
                raw_values=_evenly_spaced(5),
                encoding="onehot",
            )
            for i in range(1, 5)
        ],
        *[
            CategoricalParameter(
                name=f"random_lfo_{i}_tempo",
                values=list(range(13)),
                raw_values=_evenly_spaced(13),
                encoding="onehot",
            )
            for i in range(1, 5)
        ],
        *[
            CategoricalParameter(
                name=f"random_lfo_{i}_stereo",
                values=[False, True],
                raw_values=[0.0, 1.0],
                encoding="onehot",
            )
            for i in range(1, 5)
        ],
        *[
            CategoricalParameter(
                name=f"random_lfo_{i}_sync_type",
                values=[False, True],
                raw_values=[0.0, 1.0],
                encoding="onehot",
            )
            for i in range(1, 5)
        ],
        *[
            ContinuousParameter(name=f"random_lfo_{i}_transpose", min=0.0, max=1.0)
            for i in range(1, 5)
        ],
        *[
            ContinuousParameter(name=f"random_lfo_{i}_tune", min=0.0, max=1.0)
            for i in range(1, 5)
        ],

        # Oscillators (1..3) - core controls
        *[
            CategoricalParameter(
                name=f"oscillator_{i}_switch",
                values=[False, True],
                raw_values=[0.0, 1.0],
                encoding="onehot",
            )
            for i in range(1, 4)
        ],
        *[
            ContinuousParameter(name=f"oscillator_{i}_level", min=0.0, max=1.0)
            for i in range(1, 4)
        ],
        *[
            ContinuousParameter(name=f"oscillator_{i}_pan", min=0.0, max=1.0)
            for i in range(1, 4)
        ],
        *[
            ContinuousParameter(name=f"oscillator_{i}_phase", min=0.0, max=1.0)
            for i in range(1, 4)
        ],
        *[
            ContinuousParameter(name=f"oscillator_{i}_phase_randomization", min=0.0, max=1.0)
            for i in range(1, 4)
        ],
        *[
            CategoricalParameter(
                name=f"oscillator_{i}_smooth_interpolation",
                values=[False, True],
                raw_values=[0.0, 1.0],
                encoding="onehot",
            )
            for i in range(1, 4)
        ],
        *[
            ContinuousParameter(name=f"oscillator_{i}_transpose", min=0.0, max=1.0)
            for i in range(1, 4)
        ],
        *[
            ContinuousParameter(name=f"oscillator_{i}_tune", min=0.0, max=1.0)
            for i in range(1, 4)
        ],
        *[
            ContinuousParameter(name=f"oscillator_{i}_blend", min=0.0, max=1.0)
            for i in range(1, 4)
        ],
        *[
            ContinuousParameter(name=f"oscillator_{i}_unison_detune", min=0.0, max=1.0)
            for i in range(1, 4)
        ],
        *[
            CategoricalParameter(
                name=f"oscillator_{i}_unison_voices",
                values=list(range(1, 17)),
                raw_values=_evenly_spaced(16),
                encoding="onehot",
            )
            for i in range(1, 4)
        ],
        *[
            CategoricalParameter(
                name=f"oscillator_{i}_stack_style",
                values=list(range(13)),
                raw_values=_evenly_spaced(13),
                encoding="onehot",
            )
            for i in range(1, 4)
        ],
        *[
            CategoricalParameter(
                name=f"oscillator_{i}_spectral_unison",
                values=[False, True],
                raw_values=[0.0, 1.0],
                encoding="onehot",
            )
            for i in range(1, 4)
        ],
        *[
            CategoricalParameter(
                name=f"oscillator_{i}_destination",
                values=list(range(15)),
                raw_values=_evenly_spaced(15),
                encoding="onehot",
            )
            for i in range(1, 4)
        ],
        *[
            ContinuousParameter(name=f"oscillator_{i}_wave_frame", min=0.0, max=1.0)
            for i in range(1, 4)
        ],
        *[
            ContinuousParameter(name=f"oscillator_{i}_unison_frame_spread", min=0.0, max=1.0)
            for i in range(1, 4)
        ],
        *[
            ContinuousParameter(name=f"oscillator_{i}_stereo_spread", min=0.0, max=1.0)
            for i in range(1, 4)
        ],
        *[
            CategoricalParameter(
                name=f"oscillator_{i}_midi_track",
                values=[False, True],
                raw_values=[0.0, 1.0],
                encoding="onehot",
            )
            for i in range(1, 4)
        ],

        # Oscillator frequency morphing
        *[
            CategoricalParameter(
                name=f"oscillator_{i}_frequency_morph_type",
                values=list(range(17)),
                raw_values=_evenly_spaced(17),
                encoding="onehot",
            )
            for i in range(1, 4)
        ],
        *[
            ContinuousParameter(name=f"oscillator_{i}_frequency_morph_amount", min=0.0, max=1.0)
            for i in range(1, 4)
        ],
        *[
            ContinuousParameter(name=f"oscillator_{i}_frequency_morph_spread", min=0.0, max=1.0)
            for i in range(1, 4)
        ],
        *[
            ContinuousParameter(name=f"oscillator_{i}_frequency_morph_phase", min=0.0, max=1.0)
            for i in range(1, 4)
        ],

        # Sample section
        CategoricalParameter(
            name="sample_switch", values=[False, True], raw_values=[0.0, 1.0], encoding="onehot"
        ),
        CategoricalParameter(
            name="sample_loop", values=[False, True], raw_values=[0.0, 1.0], encoding="onehot"
        ),
        CategoricalParameter(
            name="sample_bounce", values=[False, True], raw_values=[0.0, 1.0], encoding="onehot"
        ),
        CategoricalParameter(
            name="sample_keytrack", values=[False, True], raw_values=[0.0, 1.0], encoding="onehot"
        ),
        ContinuousParameter(name="sample_level", min=0.0, max=1.0),
        ContinuousParameter(name="sample_pan", min=0.0, max=1.0),
        ContinuousParameter(name="sample_transpose", min=0.0, max=1.0),
        ContinuousParameter(name="sample_tune", min=0.0, max=1.0),
        CategoricalParameter(
            name="sample_destination",
            values=list(range(15)),
            raw_values=_evenly_spaced(15),
            encoding="onehot",
        ),
        ContinuousParameter(name="sample_transpose_quantize", min=0.0, max=1.0),

        # Voice / global performance
        CategoricalParameter(
            name="legato", values=[False, True], raw_values=[0.0, 1.0], encoding="onehot"
        ),
        ContinuousParameter(name="stereo_routing", min=0.0, max=1.0),
        CategoricalParameter(
            name="stereo_mode", values=[0, 1], raw_values=[0.0, 1.0], encoding="onehot"
        ),
        CategoricalParameter(
            name="voice_override", values=[0, 1], raw_values=[0.0, 1.0], encoding="onehot"
        ),
        CategoricalParameter(
            name="voice_priority", values=list(range(5)), raw_values=_evenly_spaced(5), encoding="onehot"
        ),
        ContinuousParameter(name="voice_amplitude", min=0.0, max=1.0),
        ContinuousParameter(name="voice_transpose", min=0.0, max=1.0),
        ContinuousParameter(name="voice_tune", min=0.0, max=1.0),
        CategoricalParameter(
            name="oversampling", values=list(range(4)), raw_values=_evenly_spaced(4), encoding="onehot"
        ),
        CategoricalParameter(
            name="polyphony", values=list(range(1, 33)), raw_values=_evenly_spaced(32), encoding="onehot"
        ),
        CategoricalParameter(
            name="pitch_bend_range", values=list(range(49)), raw_values=_evenly_spaced(49), encoding="onehot"
        ),
        ContinuousParameter(name="portamento_time", min=0.0, max=1.0),
        ContinuousParameter(name="portamento_slope", min=0.0, max=1.0),
        CategoricalParameter(
            name="portamento_force", values=[False, True], raw_values=[0.0, 1.0], encoding="onehot"
        ),
        CategoricalParameter(
            name="portamento_scale", values=[False, True], raw_values=[0.0, 1.0], encoding="onehot"
        ),
        ContinuousParameter(name="velocity_track", min=0.0, max=1.0),
        ContinuousParameter(name="mod_wheel", min=0.0, max=1.0),
        ContinuousParameter(name="pitch_wheel", min=0.0, max=1.0),
        CategoricalParameter(
            name="mpe_enabled", values=[False, True], raw_values=[0.0, 1.0], encoding="onehot"
        ),
        CategoricalParameter(
            name="view_spectrogram", values=list(range(3)), raw_values=_evenly_spaced(3), encoding="onehot"
        ),
        # Macros (pedalboard exposes macro_1..macro_4)
        ContinuousParameter(name="macro_1", min=0.0, max=1.0),
        ContinuousParameter(name="macro_2", min=0.0, max=1.0),
        ContinuousParameter(name="macro_3", min=0.0, max=1.0),
        ContinuousParameter(name="macro_4", min=0.0, max=1.0),

        # Effect chain ordering
        ContinuousParameter(name="effect_chain_order", min=0.0, max=1.0),

        # Sample: random phase (present in pedalboard list but missing in spec)
        CategoricalParameter(
            name="sample_random_phase", values=[False, True], raw_values=[0.0, 1.0], encoding="onehot"
        ),

        # Additional oscillator parameters exposed by pedalboard
        *[
            ContinuousParameter(name=f"oscillator_{i}_detune_power", min=0.0, max=1.0)
            for i in range(1, 4)
        ],
        *[
            ContinuousParameter(name=f"oscillator_{i}_detune_range", min=0.0, max=1.0)
            for i in range(1, 4)
        ],
        *[
            ContinuousParameter(name=f"oscillator_{i}_distortion_amount", min=0.0, max=1.0)
            for i in range(1, 4)
        ],
        *[
            ContinuousParameter(name=f"oscillator_{i}_distortion_phase", min=0.0, max=1.0)
            for i in range(1, 4)
        ],
        *[
            ContinuousParameter(name=f"oscillator_{i}_distortion_spread", min=0.0, max=1.0)
            for i in range(1, 4)
        ],
        *[
            CategoricalParameter(
                name=f"oscillator_{i}_distortion_type",
                values=list(range(7)),  # matches SynthOscillator::kNumDistortionTypes (approximate)
                raw_values=_evenly_spaced(7),
                encoding="onehot",
            )
            for i in range(1, 4)
        ],
        *[
            ContinuousParameter(name=f"oscillator_{i}_transpose_quantize", min=0.0, max=1.0)
            for i in range(1, 4)
        ],
        *[
            CategoricalParameter(
                name=f"oscillator_{i}_view_2d",
                values=list(range(3)),
                raw_values=_evenly_spaced(3),
                encoding="onehot",
            )
            for i in range(1, 4)
        ],

        # Extend modulation matrix coverage to 64 (pedalboard exposes up to modulation_64_*)
        *[
            param
            for i in range(1, 65)
            for param in [
                ContinuousParameter(name=f"modulation_{i}_amount", min=0.0, max=1.0),
                ContinuousParameter(name=f"modulation_{i}_power", min=0.0, max=1.0),
                CategoricalParameter(
                    name=f"modulation_{i}_bipolar",
                    values=[False, True],
                    raw_values=[0.0, 1.0],
                    encoding="onehot",
                ),
                CategoricalParameter(
                    name=f"modulation_{i}_stereo",
                    values=[False, True],
                    raw_values=[0.0, 1.0],
                    encoding="onehot",
                ),
                CategoricalParameter(
                    name=f"modulation_{i}_bypass",
                    values=[False, True],
                    raw_values=[0.0, 1.0],
                    encoding="onehot",
                ),
            ]
        ],
    ],
    [
        # Note-level parameters
        DiscreteLiteralParameter(name="pitch", min=48, max=72),
        NoteDurationParameter(name="note_start_and_end", max_note_duration_seconds=4.0),
    ],
)

VITAL_SIMPLE_PARAM_SPEC = ParamSpec(
    [
        # Global / transport
        ContinuousParameter(name="beats_per_minute", min=0.0, max=1.0),
        ContinuousParameter(name="volume", min=0.0, max=1.0),

        # Delay
        CategoricalParameter(
            name="delay_switch",
            values=[False, True],
            raw_values=[0.0, 1.0],
            encoding="onehot",
        ),
        ContinuousParameter(name="delay_mix", min=0.0, max=1.0),
        ContinuousParameter(name="delay_feedback", min=0.0, max=1.0),
        ContinuousParameter(name="delay_frequency", min=0.0, max=1.0),
        CategoricalParameter(
            name="delay_sync",
            values=list(range(4)),
            raw_values=_evenly_spaced(4),
            encoding="onehot",
        ),
        CategoricalParameter(
            name="delay_tempo",
            values=list(range(9)),
            raw_values=_evenly_spaced(9),
            encoding="onehot",
        ),

        # Chorus
        CategoricalParameter(
            name="chorus_switch",
            values=[False, True],
            raw_values=[0.0, 1.0],
            encoding="onehot",
        ),
        ContinuousParameter(name="chorus_mix", min=0.0, max=1.0),
        ContinuousParameter(name="chorus_feedback", min=0.0, max=1.0),
        ContinuousParameter(name="chorus_frequency", min=0.0, max=1.0),
        CategoricalParameter(
            name="chorus_sync",
            values=list(range(4)),
            raw_values=_evenly_spaced(4),
            encoding="onehot",
        ),
        CategoricalParameter(
            name="chorus_tempo",
            values=list(range(11)),
            raw_values=_evenly_spaced(11),
            encoding="onehot",
        ),
        ContinuousParameter(name="chorus_mod_depth", min=0.0, max=1.0),

        # Flanger
        CategoricalParameter(
            name="flanger_switch",
            values=[False, True],
            raw_values=[0.0, 1.0],
            encoding="onehot",
        ),
        ContinuousParameter(name="flanger_mix", min=0.0, max=1.0),
        ContinuousParameter(name="flanger_feedback", min=0.0, max=1.0),
        ContinuousParameter(name="flanger_frequency", min=0.0, max=1.0),
        ContinuousParameter(name="flanger_mod_depth", min=0.0, max=1.0),
        CategoricalParameter(
            name="flanger_sync",
            values=list(range(4)),
            raw_values=_evenly_spaced(4),
            encoding="onehot",
        ),
        CategoricalParameter(
            name="flanger_tempo",
            values=list(range(11)),
            raw_values=_evenly_spaced(11),
            encoding="onehot",
        ),

        # Phaser
        CategoricalParameter(
            name="phaser_switch",
            values=[False, True],
            raw_values=[0.0, 1.0],
            encoding="onehot",
        ),
        ContinuousParameter(name="phaser_mix", min=0.0, max=1.0),
        ContinuousParameter(name="phaser_feedback", min=0.0, max=1.0),
        ContinuousParameter(name="phaser_frequency", min=0.0, max=1.0),
        CategoricalParameter(
            name="phaser_sync",
            values=list(range(4)),
            raw_values=_evenly_spaced(4),
            encoding="onehot",
        ),
        CategoricalParameter(
            name="phaser_tempo",
            values=list(range(11)),
            raw_values=_evenly_spaced(11),
            encoding="onehot",
        ),
        ContinuousParameter(name="phaser_mod_depth", min=0.0, max=1.0),

        # Distortion
        CategoricalParameter(
            name="distortion_switch",
            values=[False, True],
            raw_values=[0.0, 1.0],
            encoding="onehot",
        ),
        CategoricalParameter(
            name="distortion_type",
            values=list(range(6)),
            raw_values=_evenly_spaced(6),
            encoding="onehot",
        ),
        ContinuousParameter(name="distortion_drive", min=0.0, max=1.0),
        ContinuousParameter(name="distortion_mix", min=0.0, max=1.0),

        # Compressor
        CategoricalParameter(
            name="compressor_switch",
            values=[False, True],
            raw_values=[0.0, 1.0],
            encoding="onehot",
        ),
        ContinuousParameter(name="compressor_attack", min=0.0, max=1.0),
        ContinuousParameter(name="compressor_release", min=0.0, max=1.0),
        ContinuousParameter(name="compressor_mix", min=0.0, max=1.0),
        ContinuousParameter(name="band_upper_threshold", min=0.0, max=1.0),
        ContinuousParameter(name="band_lower_threshold", min=0.0, max=1.0),
        ContinuousParameter(name="band_upper_ratio", min=0.0, max=1.0),
        ContinuousParameter(name="band_lower_ratio", min=0.0, max=1.0),

        # EQ
        CategoricalParameter(
            name="eq_switch",
            values=[False, True],
            raw_values=[0.0, 1.0],
            encoding="onehot",
        ),
        ContinuousParameter(name="eq_low_gain", min=0.0, max=1.0),
        ContinuousParameter(name="eq_band_gain", min=0.0, max=1.0),
        ContinuousParameter(name="eq_high_gain", min=0.0, max=1.0),

        # Reverb
        CategoricalParameter(
            name="reverb_switch",
            values=[False, True],
            raw_values=[0.0, 1.0],
            encoding="onehot",
        ),
        ContinuousParameter(name="reverb_mix", min=0.0, max=1.0),
        ContinuousParameter(name="reverb_decay_time", min=0.0, max=1.0),
        ContinuousParameter(name="reverb_size", min=0.0, max=1.0),
        ContinuousParameter(name="reverb_chorus_amount", min=0.0, max=1.0),

        # Filters (1, 2, fx)
        *[
            param
            for i in [1, 2]
            for param in [
                CategoricalParameter(
                    name=f"filter_{i}_switch",
                    values=[False, True],
                    raw_values=[0.0, 1.0],
                    encoding="onehot",
                ),
                ContinuousParameter(name=f"filter_{i}_mix", min=0.0, max=1.0),
                ContinuousParameter(name=f"filter_{i}_cutoff", min=0.0, max=1.0),
                ContinuousParameter(name=f"filter_{i}_resonance", min=0.0, max=1.0),
                ContinuousParameter(name=f"filter_{i}_drive", min=0.0, max=1.0),
                ContinuousParameter(name=f"filter_{i}_blend", min=0.0, max=1.0),
                CategoricalParameter(
                    name=f"filter_{i}_style",
                    values=list(range(6)),
                    raw_values=_evenly_spaced(6),
                    encoding="onehot",
                ),
            ]
        ],
        # Filter FX
        CategoricalParameter(
            name="filter_fx_switch",
            values=[False, True],
            raw_values=[0.0, 1.0],
            encoding="onehot",
        ),
        ContinuousParameter(name="filter_fx_mix", min=0.0, max=1.0),
        ContinuousParameter(name="filter_fx_cutoff", min=0.0, max=1.0),
        ContinuousParameter(name="filter_fx_resonance", min=0.0, max=1.0),
        ContinuousParameter(name="filter_fx_drive", min=0.0, max=1.0),
        ContinuousParameter(name="filter_fx_blend", min=0.0, max=1.0),
        CategoricalParameter(
            name="filter_fx_style",
            values=list(range(6)),
            raw_values=_evenly_spaced(6),
            encoding="onehot",
        ),

        # Envelopes (1..5)
        # Env 1: ADSR
        ContinuousParameter(name="envelope_1_attack", min=0.0, max=1.0),
        ContinuousParameter(name="envelope_1_decay", min=0.0, max=1.0),
        ContinuousParameter(name="envelope_1_sustain", min=0.0, max=1.0),
        ContinuousParameter(name="envelope_1_release", min=0.0, max=1.0),
        # Env 2-5: ADSR + Delay
        *[
            ContinuousParameter(name=f"envelope_{i}_{p}", min=0.0, max=1.0)
            for i in range(2, 6)
            for p in ["delay", "attack", "decay", "sustain", "release"]
        ],

        # LFOs (1..6)
        *[
            param
            for i in range(1, 7)
            for param in [
                ContinuousParameter(name=f"lfo_{i}_frequency", min=0.0, max=1.0),
                CategoricalParameter(
                    name=f"lfo_{i}_sync",
                    values=list(range(5)),
                    raw_values=_evenly_spaced(5),
                    encoding="onehot",
                ),
                CategoricalParameter(
                    name=f"lfo_{i}_tempo",
                    values=list(range(13)),
                    raw_values=_evenly_spaced(13),
                    encoding="onehot",
                ),
                ContinuousParameter(name=f"lfo_{i}_fade_in", min=0.0, max=1.0),
                ContinuousParameter(name=f"lfo_{i}_smooth_time", min=0.0, max=1.0),
            ]
        ],

        # Oscillators (1..3)
        *[
            param
            for i in range(1, 4)
            for param in [
                CategoricalParameter(
                    name=f"oscillator_{i}_switch",
                    values=[False, True],
                    raw_values=[0.0, 1.0],
                    encoding="onehot",
                ),
                ContinuousParameter(name=f"oscillator_{i}_level", min=0.0, max=1.0),
                ContinuousParameter(name=f"oscillator_{i}_pan", min=0.0, max=1.0),
                ContinuousParameter(name=f"oscillator_{i}_transpose", min=0.0, max=1.0),
                ContinuousParameter(name=f"oscillator_{i}_tune", min=0.0, max=1.0),
                ContinuousParameter(name=f"oscillator_{i}_unison_detune", min=0.0, max=1.0),
                CategoricalParameter(
                    name=f"oscillator_{i}_unison_voices",
                    values=list(range(1, 17)),
                    raw_values=_evenly_spaced(16),
                    encoding="onehot",
                ),
                ContinuousParameter(name=f"oscillator_{i}_blend", min=0.0, max=1.0),
                ContinuousParameter(name=f"oscillator_{i}_wave_frame", min=0.0, max=1.0),
                ContinuousParameter(
                    name=f"oscillator_{i}_frequency_morph_amount", min=0.0, max=1.0
                ),
                CategoricalParameter(
                    name=f"oscillator_{i}_frequency_morph_type",
                    values=list(range(17)),
                    raw_values=_evenly_spaced(17),
                    encoding="onehot",
                ),
            ]
        ],

        # Sample section
        CategoricalParameter(
            name="sample_switch",
            values=[False, True],
            raw_values=[0.0, 1.0],
            encoding="onehot",
        ),
        CategoricalParameter(
            name="sample_keytrack",
            values=[False, True],
            raw_values=[0.0, 1.0],
            encoding="onehot",
        ),
        ContinuousParameter(name="sample_level", min=0.0, max=1.0),
        ContinuousParameter(name="sample_pan", min=0.0, max=1.0),
        ContinuousParameter(name="sample_tune", min=0.0, max=1.0),

        # Voice / global performance
        CategoricalParameter(
            name="voice_priority",
            values=list(range(5)),
            raw_values=_evenly_spaced(5),
            encoding="onehot",
        ),
        CategoricalParameter(
            name="polyphony",
            values=list(range(1, 33)),
            raw_values=_evenly_spaced(32),
            encoding="onehot"
        ),
        CategoricalParameter(
            name="pitch_bend_range",
            values=list(range(49)),
            raw_values=_evenly_spaced(49),
            encoding="onehot"
        ),
        ContinuousParameter(name="portamento_time", min=0.0, max=1.0),
        ContinuousParameter(name="velocity_track", min=0.0, max=1.0),

        # Macros (1..4)
        *[
            ContinuousParameter(name=f"macro_control_{i}", min=0.0, max=1.0)
            for i in range(1, 5)
        ],

        # Modulation Matrix (1..32)
        *[
            param
            for i in range(1, 33)
            for param in [
                ContinuousParameter(name=f"modulation_{i}_amount", min=0.0, max=1.0),
                ContinuousParameter(name=f"modulation_{i}_power", min=0.0, max=1.0),
            ]
        ],
    ],
    [
        # Note-level parameters
        DiscreteLiteralParameter(name="pitch", min=48, max=72),
        NoteDurationParameter(name="note_start_and_end", max_note_duration_seconds=4.0),
    ],
)
