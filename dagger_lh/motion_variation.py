"""Conservative motion variety for the existing ScriptedPickPlace controller.

Only modifies a high approach waypoint, a high transport waypoint,
the gripper-close hold time, and transit gain. The final grasp pose,
wrist alignment, release target, contact checks, and recovery remain unchanged.
Each rollout chooses exactly one profile, even if grasp recovery occurs.
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


# Each waypoint offset is in meters, relative to a safe elevated waypoint.
# Stay within about 3.5 cm to minimize impact on baseline reliability.
# 24 preset combinations, with different approach and transport directions.
VARIANTS = (
    MotionVariation("direct_standard", (0, 0), (0, 0), 35, 5.00),
    MotionVariation("direct_patient", (0, 0), (0, 0), 43, 4.50),
    MotionVariation("direct_brisk", (0, 0), (0, 0), 37, 5.15),
    MotionVariation("direct_careful", (0, 0), (0, 0), 48, 4.20),
    MotionVariation("approach_east", (0.030, 0), (0, 0), 36, 4.80),
    MotionVariation("approach_west", (-0.030, 0), (0, 0), 42, 4.60),
    MotionVariation("approach_north", (0, 0.030), (0, 0), 39, 5.00),
    MotionVariation("approach_south", (0, -0.030), (0, 0), 45, 4.50),
    MotionVariation("approach_ne", (0.022, 0.022), (0, 0), 38, 4.90),
    MotionVariation("approach_nw", (-0.022, 0.022), (0, 0), 44, 4.60),
    MotionVariation("approach_se", (0.022, -0.022), (0, 0), 40, 5.05),
    MotionVariation("approach_sw", (-0.022, -0.022), (0, 0), 46, 4.40),
    MotionVariation("transport_east", (0, 0), (0.025, 0), 36, 4.70),
    MotionVariation("transport_west", (0, 0), (-0.025, 0), 42, 4.85),
    MotionVariation("transport_north", (0, 0), (0, 0.025), 39, 4.55),
    MotionVariation("transport_south", (0, 0), (0, -0.025), 45, 4.95),
    MotionVariation("mixed_east_north", (0.030, 0), (0, 0.025), 35, 4.75),
    MotionVariation("mixed_west_south", (-0.030, 0), (0, -0.025), 41, 4.60),
    MotionVariation("mixed_north_west", (0, 0.030), (-0.025, 0), 38, 5.00),
    MotionVariation("mixed_south_east", (0, -0.030), (0.025, 0), 44, 4.55),
    MotionVariation("mixed_ne_sw", (0.022, 0.022), (-0.018, -0.018), 37, 4.80),
    MotionVariation("mixed_nw_se", (-0.022, 0.022), (0.018, -0.018), 43, 4.65),
    MotionVariation("mixed_se_nw", (0.022, -0.022), (-0.018, 0.018), 40, 5.05),
    MotionVariation("mixed_sw_ne", (-0.022, -0.022), (0.018, 0.018), 47, 4.40),
)


def choose_motion(rng=None):
    """Choose a motion style once per rollout (not each control step)."""
    rng = rng if rng is not None else np.random.default_rng()
    return VARIANTS[int(rng.integers(len(VARIANTS)))]
