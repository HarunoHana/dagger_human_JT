
import numpy as np
from scipy.spatial.transform import Rotation

from dagger_lh.config import Config
from dagger_lh.envs import make_env

cfg = Config.load("configs/study_pickplace.yaml")
env = make_env(cfg.env)

try:
    env.reset()

    raw = env._last_raw_obs
    start_quat = np.asarray(raw["robot0_eef_quat"])

    # Quaternion order: XYZW
    start_rotation = Rotation.from_quat(start_quat)

    # Small test rotation about world X axis
    target_rotation = (
        Rotation.from_euler("x", 15, degrees=True)
        * start_rotation
    )

    for step in range(100):
        current_quat = np.asarray(
            env._last_raw_obs["robot0_eef_quat"]
        )
        current_rotation = Rotation.from_quat(current_quat)

        error_rotation = target_rotation * current_rotation.inv()
        rotvec = error_rotation.as_rotvec()

        action = np.zeros(7)
        action[3:6] = np.clip(rotvec * 2.0, -0.15, 0.15)
        action[6] = -1.0

        env.step(action)

    final_rotation = Rotation.from_quat(
        env._last_raw_obs["robot0_eef_quat"]
    )

    remaining = (
        target_rotation * final_rotation.inv()
    ).magnitude()

    print("Target rotation: 15 degrees")
    print("Remaining error:", np.degrees(remaining))

finally:
    env.close()
