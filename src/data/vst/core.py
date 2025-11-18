from __future__ import annotations

import _thread
import os
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Optional, Tuple

import mido
import numpy as np
import rootutils
from loguru import logger
from pedalboard import VST3Plugin
from pedalboard.io import AudioFile

root = rootutils.find_root(search_from=os.path.dirname(os.path.abspath(__file__)), indicator=".project-root")
from src.data.vst.vital_preset_converter import convert_vital_preset_to_params

FLUSH_DURATION_SECONDS = 0.5


def _enforce_minimal_audible_params(plugin: VST3Plugin, params: dict[str, float]) -> dict[str, float]:
    """Light guard rails to avoid silent random patches.

    Only touches parameters that exist on the plugin:
    - bypass off
    - some output volume
    - amp env (env1) fast attack, some sustain, some release
    - osc1 on with some level
    - avoid fully wet filter mix by default
    """
    out = dict(params)
    keys = set(plugin.parameters.keys())

    def set_if_present(k: str, v: float):
        if k in keys and k not in out:
            out[k] = v

    def clamp_min(k: str, vmin: float, default: float | None = None):
        if k in keys:
            if k in out:
                try:
                    if float(out[k]) < vmin:
                        out[k] = vmin
                except Exception:
                    pass
            elif default is not None:
                out[k] = default

    if "bypass" in keys:
        out["bypass"] = 0.0
    clamp_min("volume", 0.3, default=0.7)
    clamp_min("envelope_1_sustain", 0.3, default=0.8)
    if "envelope_1_attack" in keys and ("envelope_1_attack" not in out or out["envelope_1_attack"] > 0.3):
        out["envelope_1_attack"] = 0.01
    set_if_present("envelope_1_release", 0.2)
    set_if_present("oscillator_1_switch", 1.0)
    clamp_min("oscillator_1_level", 0.3, default=0.7)
    if "filter_1_mix" in keys and out.get("filter_1_mix", 0.0) > 0.9:
        out["filter_1_mix"] = 0.2

    return out


def _call_with_interrupt(fn: Callable, sleep_time: float = 2.0):
    """Calls the function fn on the main thread, while another thread sends a KeyboardInterrupt
    (SIGINT) to the main thread."""

    def send_interrupt():
        # Brief sleep so that fn starts before we send the interrupt
        time.sleep(sleep_time)
        _thread.interrupt_main()

    # Create and start the thread that sends the interrupt
    t = threading.Thread(target=send_interrupt)
    t.start()

    try:
        fn()
    except KeyboardInterrupt:
        print("Interrupted main thread.")
    finally:
        t.join()


def _prepare_plugin(plugin: VST3Plugin) -> None:
    _call_with_interrupt(plugin.show_editor, sleep_time=2.0)


def load_plugin(plugin_path: str) -> VST3Plugin:
    logger.info(f"Loading plugin {plugin_path}")
    p = VST3Plugin(plugin_path)
    logger.info(f"Plugin {plugin_path} loaded")
    logger.info("Preparing plugin for preset load...")
    # _prepare_plugin(p) # NOTE: commented out to avoid GUI
    logger.info("Plugin ready")
    return p


def load_preset(plugin: VST3Plugin, preset_path: str) -> None:
    logger.info(f"Loading preset {preset_path}")
    # Handle legacy Vital JSON presets
    if preset_path.lower().endswith(".vital"):
        try:
            params = convert_vital_preset_to_params(preset_path, plugin)
            if params:
                set_params(plugin, params)
                logger.info("Applied .vital preset via parameter mapping")
            else:
                logger.warning(".vital preset produced no applicable parameters")
        except Exception as e:
            logger.error(f"Failed to convert/apply .vital preset: {e}")
            raise
    else:
        plugin.load_preset(preset_path)
        logger.info(f"Preset {preset_path} loaded")


def set_params(plugin: VST3Plugin, params: dict[str, float]) -> None:
    for k, v in params.items():
        try:
            plugin.parameters[k].raw_value = v
        except KeyError:
            logger.warning(
                f"Parameter '{k}' not found in plugin. Available parameters: {list(plugin.parameters.keys())}"
            )
            raise


def write_wav(audio: np.ndarray, path: str, sample_rate: float, channels: int) -> None:
    with AudioFile(str(path), "w", sample_rate, channels) as f:
        f.write(audio.T)


def render_params(
    plugin: VST3Plugin,
    params: dict[str, float],
    midi_note: int,
    velocity: int,
    note_start_and_end: tuple[float, float],
    signal_duration_seconds: float,
    sample_rate: float,
    channels: int,
    preset_path: str | None = None,
) -> np.ndarray:
    if preset_path is not None:
        load_preset(plugin, preset_path)

    logger.debug("post-load flush")
    plugin.process([], FLUSH_DURATION_SECONDS, sample_rate, channels, 2048, True)  # flush
    plugin.reset()

    logger.debug("setting params")
    params = _enforce_minimal_audible_params(plugin, params)
    set_params(plugin, params)
    # plugin.reset()

    logger.debug("post-param flush")
    plugin.process([], FLUSH_DURATION_SECONDS, sample_rate, channels, 2048, True)  # flush
    plugin.reset()

    midi_events = make_midi_events(midi_note, velocity, *note_start_and_end)

    logger.debug("rendering audio")
    output = plugin.process(
        midi_events, signal_duration_seconds, sample_rate, channels, 2048, True
    )

    logger.debug("post-render flush")
    plugin.process([], FLUSH_DURATION_SECONDS, sample_rate, channels, 2048, True)  # flush
    plugin.reset()
    # If silent, just log and return as requested
    if np.max(np.abs(output)) == 0:
        logger.warning("Rendered audio is silent (peak==0). Skipping.")
    return output


def make_midi_events(pitch: int, velocity: int, note_start: float, note_end: float):
    events = []
    note_on = mido.Message("note_on", note=pitch, velocity=velocity, time=0)
    events.append((note_on.bytes(), note_start))
    note_off = mido.Message("note_off", note=pitch, velocity=velocity, time=0)
    events.append((note_off.bytes(), note_end))

    return tuple(events)
