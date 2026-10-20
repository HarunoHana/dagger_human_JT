
from dagger_lh.config import Config
from dagger_lh.envs import make_env

cfg = Config.load("configs/study_pickplace.yaml")
env = make_env(cfg.env)

try:
    env.reset()

    raw_env = env
    for _ in range(10):
        if hasattr(raw_env, "target_bin_placements"):
            break
        if hasattr(raw_env, "env"):
            raw_env = raw_env.env
        else:
            break

    print("\n=== TARGET INFORMATION ===")

    for name in (
        "target_bin_placements",
        "bin2_pos",
        "bin1_pos",
        "object_to_id",
    ):
        if hasattr(raw_env, name):
            print(name, getattr(raw_env, name))

    print("\n=== CAN POSITION ===")
    print(env._last_raw_obs.get("Can_pos"))

finally:
    env.close()
