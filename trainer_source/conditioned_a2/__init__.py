"""Conditioned NAM A2 training and export support.

This package deliberately lives beside, rather than inside, the historical
PANAMA trainer fork.  It targets the pinned upstream ``neural-amp-modeler``
0.13 A2 model shape and exports DessMetal's block-constant global controls.
"""

from .model import (
    A2_CHANNELS,
    ConditionedA2Net,
    build_conditioned_a2,
    export_child_dict,
    pack_slimmable_container,
)

__all__ = [
    "A2_CHANNELS",
    "ConditionedA2Net",
    "build_conditioned_a2",
    "export_child_dict",
    "pack_slimmable_container",
]
