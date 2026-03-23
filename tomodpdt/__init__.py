"""Top-level package exports for TomoDpDt.

This module keeps imports lazy so that utility modules remain importable even
when some optional heavy dependencies are unavailable, while still preserving
the notebook-friendly API such as ``tomodpdt.Tomography``.
"""

from importlib import import_module

_MODULE_EXPORTS = {
    "application": "application",
    "fft_loader": "fft_loader",
    "forward_module": "forward_module",
    "helpers": "helpers",
    "image_modalities_dt": "image_modalities_dt",
    "imaging_modality_torch": "imaging_modality_torch",
    "plotting": "plotting",
    "rotations": "rotations",
    "simulate": "simulate",
    "volumes": "volumes",
}

_SYMBOL_EXPORTS = {
    "Tomography": ("application", "Tomography"),
    "StageSpec": ("application", "StageSpec"),
    "ForwardModelSimple": ("forward_module", "ForwardModelSimple"),
    "create_data": ("simulate", "create_data"),
    "vec_to_field_multi": ("fft_loader", "vec_to_field_multi"),
    "vec_to_field": ("fft_loader", "vec_to_field"),
    "field_to_vec_multi": ("fft_loader", "field_to_vec_multi"),
    "field_to_vec": ("fft_loader", "field_to_vec"),
}

__all__ = sorted([*_MODULE_EXPORTS.keys(), *_SYMBOL_EXPORTS.keys()])


def __getattr__(name):
    if name in _MODULE_EXPORTS:
        module = import_module(f".{_MODULE_EXPORTS[name]}", __name__)
        globals()[name] = module
        return module

    if name in _SYMBOL_EXPORTS:
        module_name, symbol_name = _SYMBOL_EXPORTS[name]
        module = import_module(f".{module_name}", __name__)
        value = getattr(module, symbol_name)
        globals()[name] = value
        return value

    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")


def __dir__():
    return sorted(set(globals()) | set(__all__))
