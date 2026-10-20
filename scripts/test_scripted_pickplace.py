
import numpy as np

from dagger_lh.config import Config
from dagger_lh.envs import make_env
from dagger_lh.scripted_pickplace import ScriptedPickPlace

cfg = Config.load("configs/study_pickplace.yaml")
env = make_env(cfg.env)

try:
    env.reset()
    robot = ScriptedPickPlace(env)

    success = False

    for step in range(600):
        action = robot.act()
        _, _, done, info = env.step(action)

        if step % 50 == 0:
            print(
                f"Step: {step} | "
                f"Stage: {robot.stage} | "
                f"Success: {info['success']}"
            )

        if info["success"]:
            success = True
            print("PICK AND PLACE SUCCESS!")
            break

    print("Final success:", success)

finally:
    env.close()
