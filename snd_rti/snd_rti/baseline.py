"""
One-to-one baseline model.

For every full-RTI route (i,j) the baseline forces an empty-return
route (j,i) with the same daily volume; no hubs can be activated and
no cross-route consolidation is allowed. This produces the reference
total cost against which the optimised SND-RTI solution is compared.
"""
from __future__ import annotations

from .instance import InstanceData
from .model import build_model


def build_baseline_model(inst: InstanceData, verbose: bool = False):
    """Return a ConcreteModel representing the one-to-one baseline."""
    return build_model(
        inst,
        allow_hubs=False,
        one_to_one_empties=True,
        verbose=verbose,
    )
