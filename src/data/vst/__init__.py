from src.data.vst.core import load_plugin as load_plugin
from src.data.vst.core import load_preset as load_preset
from src.data.vst.core import render_params as render_params
from src.data.vst.surge_xt_param_spec import (
    SURGE_SIMPLE_PARAM_SPEC,
    SURGE_XT_PARAM_SPEC,
)
from src.data.vst.vital_param_spec import VITAL_PARAM_SPEC

param_specs = {
    "surge_xt": SURGE_XT_PARAM_SPEC,
    "surge_simple": SURGE_SIMPLE_PARAM_SPEC,
    "vital": VITAL_PARAM_SPEC,
}
