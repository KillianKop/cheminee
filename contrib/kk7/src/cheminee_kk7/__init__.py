"""Optional add-on for cheminee: thermal climatology from thermal.kk7.ch blended into the lift model.

Kept outside the `cheminee` package on purpose: kk7 data are CC BY-NC-SA 4.0 (non-commercial), so cheminee
itself and its flight package never depend on them.
"""

from .lift import Kk7Lift, kk7_factory, load_layers

__all__ = ["Kk7Lift", "kk7_factory", "load_layers"]
