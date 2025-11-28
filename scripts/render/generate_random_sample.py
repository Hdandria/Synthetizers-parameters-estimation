#!/usr/bin/env python3
import time
from pathlib import Path

import click
import numpy as np
import rootutils

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
from src.data.vst import load_plugin, param_specs, render_params  # noqa: E402
from src.data.vst.core import write_wav


@click.command()
@click.option("--plugin-key", type=click.Choice(list(param_specs.keys())), default="vital")
@click.option("--plugin-path", type=click.Path(path_type=Path), default=None)
@click.option("--preset", type=click.Path(path_type=Path), default=None)
@click.option("--duration", type=float, default=4.0)
@click.option("--sr", type=int, default=44100)
@click.option("--channels", type=int, default=2)
@click.option("--velocity", type=int, default=100)
@click.option("--out", type=click.Path(path_type=Path), default=None)
@click.option("--pitch", type=int, default=None)
def main(plugin_key, plugin_path, preset, duration, sr, channels, velocity, out, pitch):
    # Pick a default plugin if none provided
    if plugin_path is None:
        root = Path(__file__).resolve().parents[2]
        plugin_path = {
            "vital": root / "plugins" / "Vital.vst3",
            "surge_xt": root / "plugins" / "Surge XT.vst3",
            "surge_simple": root / "plugins" / "Surge XT.vst3",
        }[plugin_key]
    # Common mistake: passing a .vital as the plugin path
    if str(plugin_path).lower().endswith(".vital"):
        raise click.ClickException("--plugin-path must be a VST3. Pass .vital via --preset.")

    # Default output path
    if out is None:
        ts = time.strftime("%Y%m%d-%H%M%S")
        out = Path.cwd() / "outputs" / "random" / f"random_{plugin_key}_{ts}.wav"
    out.parent.mkdir(parents=True, exist_ok=True)

    plugin = load_plugin(str(plugin_path))
    spec = param_specs[plugin_key]

    # If preset is provided, keep synth params untouched; only sample note window and pitch
    synth_params, note_params = spec.sample()
    if preset is not None:
        synth_params = {}
    if pitch is not None:
        note_params["pitch"] = int(pitch)

    # Ensure note starts early enough to be heard
    window = note_params["note_start_and_end"]
    if not isinstance(window, tuple):
        window = (0.0, float(window))
    ns, ne = window
    if ns > 0.5:
        window = (0.2, max(0.4, ne - (ns - 0.2)))

    audio = render_params(
        plugin=plugin,
        params=synth_params,
        midi_note=int(note_params["pitch"]),
        velocity=int(velocity),
        note_start_and_end=window,
        signal_duration_seconds=float(duration),
        sample_rate=float(sr),
        channels=int(channels),
        preset_path=str(preset) if preset is not None else None,
    )

    # Clean NaN/Inf, fail if silent; then normalize
    audio = np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0)
    peak = float(np.max(np.abs(audio)))
    if peak == 0.0:
        raise click.ClickException(
            "Rendered audio is silent or invalid (peak==0 after cleanup). Aborting without writing."
        )
    audio = audio / peak

    write_wav(audio, str(out), float(sr), int(channels))
    print(f"Wrote: {out} (peak={peak:.3f})")


if __name__ == "__main__":
    main()
