import math

import pytest

from src.data.vst.vital_details import (
    plugin_name_to_internal,
    get_min_max_for_plugin_name,
)


@pytest.mark.parametrize(
    "plugin_name,expected_internal,expected_range",
    [
        ("chorus_delay_1", "chorus_delay_1", (-10.0, -5.64386)),
        ("delay_aux_frequency", "delay_frequency_2", (-2.0, 9.0)),
        ("filter_1_cutoff", "filter_1_cutoff", (8.0, 136.0)),
        ("filter_fx_switch", "filter_fx_switch", (0.0, 1.0)),
        ("lfo_3_delay_time", "lfo_3_delay", (0.0, 1.4142135624)),
    ],
)
def test_plugin_name_conversion_and_ranges(plugin_name, expected_internal, expected_range):
    internal = plugin_name_to_internal(plugin_name)
    assert internal == expected_internal
    rng = get_min_max_for_plugin_name(plugin_name)
    assert rng is not None
    # Compare floats with tolerance to avoid flakiness.
    assert math.isclose(rng[0], expected_range[0], rel_tol=1e-9, abs_tol=1e-9)
    assert math.isclose(rng[1], expected_range[1], rel_tol=1e-9, abs_tol=1e-9)


def test_unknown_parameter_returns_none():
    assert get_min_max_for_plugin_name("nonexistent_param_xyz") is None
