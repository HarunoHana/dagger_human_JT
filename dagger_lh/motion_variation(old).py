"""Small, reproducible motion variations for scripted PickPlace.

Only modifies approach and transport waypoints and grasp pause duration.
Never changes the final grasp pose, target bin, or wrist orientation.
"""
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class MotionVariation:
    name: str
    approach_offset_xy: tuple[float, float]
    transport_offset_xy: tuple[float, float]
    grasp_close_steps: int
    transit_gain: float


VARIANTS = (
    MotionVariation("direct", (0.0, 0.0), (0.0, 0.0), 35, 5.0),
    MotionVariation("left_arc", (-0.035, 0.0), (-0.025, 0.0), 40, 4.5),
    MotionVariation("right_arc", (0.035, 0.0), (0.025, 0.0), 45, 4.8),
    MotionVariation("side_arc", (0.0, 0.035), (0.0, 0.025), 38, 5.0),
)


def choose_motion(rng=None):
    """Select one movement style once for this rollout."""
    rng = rng if rng is not None else np.random.default_rng()
    return VARIANTS[int(rng.integers(len(VARIANTS)))]
