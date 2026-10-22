"""Interactive session loops: demonstration collection and HG-DAgger rounds.

Two entry points, selected by `Config.mode`:

  ``demo``    pure teleoperation. The human drives every step; episodes are saved
              as robomimic demos. Use this to build the seed dataset.

  ``dagger``  Human-Gated DAgger (Kelly et al. 2019). The policy drives; the
              operator watches and takes control (TAB) when it goes wrong. Only
              human-labelled states enter the aggregated training set, which is
              what makes keyboard teleoperation tractable -- the alternative
              (labelling every state) is not something a person can do at 20 Hz.

After each round the policy is LoRA-finetuned on everything collected so far,
under the contingency regime that `StudyController` dictates for that round.
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .config import Config
from .data import DemoWriter, Episode, EpisodeBuffer, load_episodes
from .envs import RobosuiteEnv, make_env, silence_robosuite_logs
from .study import (RoundMetrics, SessionLogger, StudyController, SurveyRunner,
                    helplessness_index)
from .teleop import KeyboardDevice, OperatorDisplay, TeleopState
from .trainer import PolicyTrainer


@dataclass
class EpisodeResult:
    success: bool = False
    steps: int = 0
    seconds: float = 0.0
    human_steps: int = 0
    engaged_steps: int = 0
    idle_steps: int = 0
    keypresses: int = 0
    n_interventions: int = 0
    first_intervention_step: Optional[int] = None
    time_to_first_intervention: Optional[float] = None
    gave_up: bool = False
    discarded: bool = False
    quit: bool = False
    buffer: Optional[EpisodeBuffer] = None


class _Aborted(Exception):
    """Operator asked to quit the session."""


class SessionRunner:
    """Shared setup for both modes: env, display, teleop, policy, logging."""

    def __init__(self, cfg: Config, display=None, teleop=None, env=None,
                 survey=None):
        """`display`/`teleop`/`env`/`survey` may be injected to drive the loop
        from something other than a local pygame window -- headlessly for tests
        (scripts/smoke_test.py) or over a WebSocket for the online study
        (dagger_lh/web/worker.py). Defaults build the local pygame front-end."""
        silence_robosuite_logs()
        self.cfg = cfg
        self.env: RobosuiteEnv = env if env is not None else make_env(cfg.env)
        self.display = display if display is not None else OperatorDisplay(
            cfg.teleop, main_size=cfg.env.render_height)
        self.teleop = teleop if teleop is not None else KeyboardDevice(
            cfg.teleop, action_dim=self.env.action_dim)
        self.headless = display is not None

        session_root = os.path.join(cfg.paths.sessions,
                                    f"{cfg.study.participant_id}_{cfg.mode}_"
                                    f"{time.strftime('%Y%m%d_%H%M%S')}")
        self.logger = SessionLogger(session_root, cfg)
        self.session_dir = session_root

        demo_path = cfg.demo_dataset or os.path.join(session_root, "demos.hdf5")
        self.writer = DemoWriter(demo_path, self.env.env_meta())
        self.demo_path = demo_path
        self.obs_keys = self.env.obs_keys
        self.episodes: List[Episode] = []
        self.study = StudyController(cfg.study)
        self.survey = survey if survey is not None else SurveyRunner(self.display)
        self._round_demo_names: Dict[int, List[str]] = {}
        self.ladder = None
        if cfg.study.ladder_path:
            from .ladder import PolicyLadder
            self.ladder = PolicyLadder(cfg.study.ladder_path)

    # ------------------------------------------------------------------ helpers
    def _render(self) -> Tuple[np.ndarray, List[np.ndarray]]:
        if getattr(self, "headless", False) and not getattr(
                self.display, "wants_frames", False):
            return np.zeros((8, 8, 3), dtype=np.uint8), []
        main = self.env.render_operator()
        sides: List[np.ndarray] = []
        if self.cfg.teleop.show_second_view:
            for cam in self.cfg.env.camera_names:
                if cam != self.cfg.env.render_camera:
                    sides.append(self.env.render_operator(
                    camera=cam,
                    height=self.cfg.env.render_height,
                    width=self.cfg.env.render_width))
                    break
        return main, sides

    def _handle_global_events(self, st: TeleopState) -> None:
        if st.event_toggle_help:
            self.display.show_help = not self.display.show_help
        if st.event_quit:
            raise _Aborted()
        if st.event_pause:
            self.display.message("Paused", "", sub="press ENTER to resume")
            self.display.wait_for_key()

    def _commit_episode(self, ep: EpisodeBuffer, round_index: int,
                        phase: int, condition: str, success: bool,
                        source: str) -> Optional[str]:
        ep.meta.update(
            round=int(round_index), phase=int(phase), condition=str(condition),
            success=bool(success), source=source,
            participant_id=self.cfg.study.participant_id,
            timestamp=time.time(),
        )
        name = self.writer.write_episode(ep)
        if name is not None:
            self._round_demo_names.setdefault(round_index, []).append(name)
            self.writer.write_filter_key(
                f"round_{round_index}", self._round_demo_names[round_index])
            # Reload just this episode so the trainer sees exactly what is on disk.
            self.episodes.extend(load_episodes(self.demo_path, self.obs_keys, [name]))
        return name

    def close(self) -> None:
        self.display.close()
        self.env.close()

    # ------------------------------------------------------------------ episodes
    def run_episode(
        self,
        policy=None,
        round_index: int = 0,
        phase: int = 1,
        condition: str = "responsive",
        action_noise_std: float = 0.0,
        hud_extra: Optional[Dict[str, object]] = None,
        record_all_steps: bool = True,
    ) -> EpisodeResult:
        """One episode. With `policy=None` the human drives every step (demo mode);
        otherwise the policy drives until the operator takes control (HG-DAgger).
        """
        cfg = self.cfg
        res = EpisodeResult()
        buf = EpisodeBuffer()
        buf.model_xml = self.env.get_model_xml()

        if policy is not None and self.cfg.env.env_name == "PickPlaceCan":
            from .evaluate import sample_init_states

            if not hasattr(self, "_study_init_states"):
                self._study_init_states = sample_init_states(
                    self.env,
                    n=max(self.study.total_rounds, 50),
                    seed=9901,
                )

            obs = self.env.reset_to_sim_state(
                self._study_init_states[round_index]
            )
        else:
            obs = self.env.reset()
        self.teleop.reset_episode()
        if policy is not None:
            policy.reset_rollout()
        if policy is None:
            # In demo mode the human always has authority.
            setter = getattr(self.teleop, "set_authority", None)
            if callable(setter):
                setter(True)

        rng = np.random.default_rng(cfg.study.seed + round_index)
        keys_before = self.teleop.total_keypresses
        was_intervening = False
        t_start = time.time()
        success = False
        step = 0

        # AI만 타이머 되게끔
        ai_elapsed = 0.0
        ai_limit = 60.0
        ai_timed_out = False
        last_tick = time.monotonic()
        previous_human = False
        last_robot_gripper_closed = False

        while True:
            st = self.teleop.poll()
            self._handle_global_events(st)

            now = time.monotonic()
            delta = now - last_tick
            last_tick = now

            # Count time only when AI was controlling
            if not previous_human and policy is not None:
                ai_elapsed += delta

            if ai_elapsed >= ai_limit:
                ai_timed_out = True
                self.logger.log(
                    "ai_timeout",
                    {"round": round_index, "ai_seconds": ai_elapsed}
                )
                break

            if st.event_giveup:
                res.gave_up = True
                self.logger.log("giveup", {"round": round_index, "step": step})
                break
            if st.event_discard:
                res.discarded = True
                self.logger.log("discard", {"round": round_index, "step": step})
                break
            if st.event_success:
                if policy is not None and self.ladder is not None:
                    # In the I/W/S study, Enter must not fabricate a robot success.
                    # Only info["success"] from the environment can mark success.
                    self.logger.log("manual_success_ignored", {
                        "round": round_index, "step": step,
                    })
                    continue
                success = True
                break

            human_in_control = st.intervening or policy is None
            previous_human = human_in_control

            if human_in_control and not was_intervening:
                if policy is not None:
                    self.teleop.set_gripper_closed(
                        last_robot_gripper_closed
                    )

                    # poll() already created st.action.
                    # Update it immediately for this frame.
                    st.action[6] = (
                        1.0 if last_robot_gripper_closed else -1.0
                    )
                    st.gripper_closed = last_robot_gripper_closed

                res.n_interventions += 1
                if res.first_intervention_step is None:
                    res.first_intervention_step = step
                    res.time_to_first_intervention = time.time() - t_start
                self.logger.log(
                    "intervention_start",
                    {"round": round_index, "step": step}
                )

            elif was_intervening and not human_in_control:
                self.logger.log(
                    "intervention_end",
                    {"round": round_index, "step": step}
                )

                if hasattr(policy, "resume_from_human"):
                    policy.resume_from_human()

            was_intervening = human_in_control


            if human_in_control:
                action = st.action
                res.human_steps += 1
                if not st.engaged:
                    res.idle_steps += 1
            else:
                action = np.asarray(policy.act(obs), dtype=np.float64)
                last_robot_gripper_closed = bool(action[6] > 0)
                if action_noise_std > 0:
                    # `degraded` condition: the policy visibly worsens regardless
                    # of what the operator does.
                    action = action + rng.normal(0.0, action_noise_std, action.shape)
                    action = np.clip(action, -1.0, 1.0)
            res.engaged_steps += int(st.engaged)

            state = self.env.get_sim_state()
            next_obs, reward, done, info = self.env.step(action)

            if record_all_steps or human_in_control:
                buf.add(obs, action, reward, done, state=state,
                        actor=int(human_in_control),
                        # In demo mode the human holds authority throughout, even
                        # though no TAB toggle was involved.
                        intervention=int(st.intervening or policy is None))
            obs = next_obs
            step += 1

            if info.get("success"):
                success = True

            main, sides = self._render()
            hud = {
                "mode": ("DEMONSTRATION" if policy is None else "DAgger"),
                "controller": "YOU" if human_in_control else "ROBOT",
                "round": f"{round_index + 1}/{self.study.total_rounds}",
                "step": f"{step}/{cfg.env.horizon}",
                "time": f"{max(0.0, ai_limit - ai_elapsed):.1f}s LEFT",
                "gripper": "closed" if st.gripper_closed else "open",
                "status": ("SUCCESS - press ENTER" if success else ""),
            }
            hud.update(hud_extra or {})
            self.display.draw(main, hud, sides)

            if success or done:
                break

        res.steps = step
        res.seconds = time.time() - t_start
        res.success = success
        res.keypresses = self.teleop.total_keypresses - keys_before
        res.buffer = buf
        return res

    # ------------------------------------------------------------------ demo mode
    
    def run_scripted_test(self):
        from .scripted_pickplace import ScriptedPickPlace

        robot = ScriptedPickPlace(self.env)

        self.display.message(
            "SCRIPTED ROBOT TEST",
            "The robot will perform Pick and Place.\n\n"
            "Press TAB to take control.\n"
            "Press TAB again to return control to the robot.\n\n"
            "Press ENTER to begin.",
        )

        if self.display.wait_for_key() == self.display.K_ESCAPE:
            return

        robot.reset()

        result = self.run_episode(
            policy=robot,
            round_index=0,
            condition="scripted",
        )

        print("Success:", result.success)
        print("Human steps:", result.human_steps)
        print("Interventions:", result.n_interventions)
        print("Total seconds:", result.seconds)

    def run_demo_mode(self, num_demos: Optional[int] = None) -> Dict[str, Any]:
        """Pure teleoperation collection: no policy in the loop."""
        cfg = self.cfg
        target = int(num_demos or cfg.study.num_rounds)
        self.display.message(
            "DEMONSTRATION MODE",
            f"You will teleoperate the robot to complete: {cfg.env.env_name}\n"
            f"Target: {target} successful demonstrations\n\n"
            "You have full control the whole time.\n"
            "Press ENTER when the task is done, BACKSPACE to redo,\n"
            "X if you want to give up on an attempt.\n\n"
            "Controls are listed on the right during the episode.",
        )
        if self.display.wait_for_key() == self.display.K_ESCAPE:
            return {"aborted": True, "collected": 0}

        collected = 0
        attempts = 0
        try:
            while collected < target:
                attempts += 1
                res = self.run_episode(policy=None, round_index=collected)
                if res.discarded:
                    self.logger.log("demo_discarded", {"attempt": attempts})
                    continue
                if res.gave_up:
                    self.logger.log("demo_gaveup", {"attempt": attempts})
                    continue
                name = self._commit_episode(
                    res.buffer, collected, 1, "demo", res.success, "human_demo")
                collected += 1
                self.logger.log("demo_saved", {
                    "name": name, "steps": res.steps, "success": res.success,
                    "seconds": res.seconds, "keypresses": res.keypresses,
                })
                self.display.message(
                    f"Saved demonstration {collected}/{target}",
                    f"steps: {res.steps}   time: {res.seconds:.1f}s   "
                    f"success: {res.success}",
                )
                if self.display.wait_for_key() == self.display.K_ESCAPE:
                    break
        except _Aborted:
            self.logger.log("aborted", {"where": "demo_mode"})

        summary = {
            "mode": "demo", "collected": collected, "attempts": attempts,
            "dataset": self.demo_path,
        }
        self.logger.write_summary(summary)
        return summary

    # ------------------------------------------------------------------ dagger
    def run_dagger_mode(self, trainer: PolicyTrainer) -> Dict[str, Any]:
        """Round loop: rollout with intervention, then a contingency-gated update."""
        cfg = self.cfg
        policy = trainer.policy

        self.display.message(
            "ROBOT TEACHING SESSION",
            f"Task: {cfg.env.env_name}\n\n"
            "The robot will try the task on its own.\n"
            "Watch it. When it goes wrong, press TAB to TAKE CONTROL\n"
            "and show it what to do. Press TAB again to hand control back.\n\n"
            "After each attempt, the robot policy may change.\n"
            "BACKSPACE = retry   X = give up (move to survey)\n"
        )
        if self.display.wait_for_key() == self.display.K_ESCAPE:
            return {"aborted": True}

        feedback = ""
        try:
            for r in range(self.study.total_rounds):
                plan = self.study.plan_round(r)
                self.logger.log("round_start", {
                    "round": r, "phase": plan.phase, "condition": plan.condition,
                    "commit_update": plan.commit_update,
                    "action_noise_std": plan.action_noise_std,
                    "reason": plan.reason,
                })

                metrics = RoundMetrics(
                    round_index=r, phase=plan.phase, condition=plan.condition,
                    contingent=self.study.is_contingent(r),
                )

                # Under a scripted schedule the behaviour the operator watches
                # comes from a ladder rung of known competence, not from the
                # policy being trained. The trainable policy is still updated and
                # rolled back below, so the wall-clock cost is unchanged.
                rollout_policy = policy
                round_noise = plan.action_noise_std
                if plan.ladder_target is not None and self.ladder is not None:
                    rollout_policy, rung = self.ladder.load_rung(
                        plan.ladder_target, trainer.device, cfg.policy)
                    # A rung may encode its competence as calibrated rollout
                    # noise rather than as distinct weights. When the ladder bakes
                    # that noise into the checkpoint, the loaded policy already
                    # applies it -- adding it again here would double it.
                    if not self.ladder.manifest.get("noise_baked"):
                        round_noise += float(rung.get("action_noise_std", 0.0) or 0.0)
                    self.logger.log("ladder_rung", {
                        "round": r, "target": plan.ladder_target,
                        "rung_success": rung["verified_success"],
                        "rung_noise": rung.get("action_noise_std", 0.0),
                        "checkpoint": rung["checkpoint"],
                    })

                if r == 0 and plan.ladder_target is not None and self.ladder is not None:
                    first_rung = self.ladder.nearest(plan.ladder_target)
                    self.display.message(
                        "Robot Policy — Attempt 1 of 4",
                        f"Current policy success rate: {first_rung['verified_success']:.0%}\n\n"
                        "Press ENTER to begin.",
                    )
                    if self.display.wait_for_key() == self.display.K_ESCAPE:
                        raise _Aborted()

                # ---- rollouts ----
                for k in range(int(cfg.study.rollouts_per_round)):
                    # BACKSPACE discards the attempt and retries THIS round.
                    # It must not trigger the policy-update/survey flow.
                    while True:
                        res = self.run_episode(
                            policy=rollout_policy, round_index=r, phase=plan.phase,
                            condition=plan.condition,
                            action_noise_std=round_noise,
                            hud_extra={"phase": f"{plan.phase}", "feedback": feedback},
                        )
                        if res.discarded:
                            self.logger.log("round_retry", {
                                "round": r, "rollout": k,
                                "reason": "backspace_discard",
                            })
                            self.display.message(
                                "Retrying Attempt",
                                f"Attempt {r + 1} of {self.study.total_rounds}"
                                "\n\nPress ENTER to retry.",
                            )
                            if self.display.wait_for_key() == self.display.K_ESCAPE:
                                raise _Aborted()
                            continue
                        break
                    metrics.n_rollouts += 1
                    metrics.steps_total += res.steps
                    metrics.human_steps += res.human_steps
                    metrics.engaged_steps += res.engaged_steps
                    metrics.idle_steps += res.idle_steps
                    metrics.keypresses += res.keypresses
                    metrics.n_interventions += res.n_interventions
                    metrics.episode_seconds += res.seconds
                    metrics.giveup_events += int(res.gave_up)
                    metrics.discard_events += int(res.discarded)
                    metrics.n_success += int(res.success)
                    if (metrics.first_intervention_step is None
                            and res.first_intervention_step is not None):
                        metrics.first_intervention_step = res.first_intervention_step
                        metrics.time_to_first_intervention = res.time_to_first_intervention

                    if not res.discarded and res.buffer is not None and len(res.buffer):
                        self._commit_episode(res.buffer, r, plan.phase,
                                             plan.condition, res.success, "dagger")

                # IWS_THREE_SETS_V2: Only swap pretrained policies for I/W/S.
                # Official probabilities: Michelle's verified_success in ladder.json.
                # Local cross-check: 10 checkpoints x 50 rollouts = 500 rollouts.
                # Individual local rates differed, but the overall trend was similar.
                # No participant-dependent training is committed for these sets.
                if plan.ladder_target is not None:
                    self._wait_showing(
                        "Updating Robot Policy",
                        "Preparing the next robot policy...",
                        max(1.0, float(cfg.train.finetune_min_seconds)),
                    )
                    metrics.train_report = {
                        "committed": False,
                        "note": "pretrained checkpoint switching; no model training",
                    }
                    self.logger.log("policy_update", {
                        "round": r, **metrics.train_report,
                    })
                else:
                    # Retain original training behavior for other experiments.
                    self.display.message(
                        "Updating the robot's policy",
                        "Learning from your corrections...\n\nplease wait", sub="")
                    t_update = time.time()
                    report = trainer.finetune(
                        self.episodes, commit=plan.commit_update,
                        progress_cb=lambda i, n, l: self._train_progress(i, n, l),
                    )
                    floor = float(cfg.train.finetune_min_seconds)
                    if floor > 0:
                        remaining = floor - (time.time() - t_update)
                        if remaining > 0:
                            self._wait_showing(
                                "Updating the robot's policy",
                                "Learning from your corrections...\n\nplease wait",
                                remaining)
                    metrics.train_report = report.as_dict()
                    self.logger.log("policy_update", {"round": r, **report.as_dict()})

                if plan.ladder_target is not None and self.ladder is not None:
                    # Show NEXT policy's rate before survey, not the previous
                    # round's measured success. Last round shows current rate.
                    if r + 1 < self.study.total_rounds:
                        shown_plan = self.study.plan_round(r + 1)
                        shown_rung = self.ladder.nearest(shown_plan.ladder_target)
                        subtitle = f"Next attempt: {r + 2} of 4"
                    else:
                        shown_rung = self.ladder.nearest(plan.ladder_target)
                        subtitle = "Final attempt complete"
                    self.display.message(
                        "Robot Policy Updated",
                        f"{subtitle}\n\n"
                        f"Current policy success rate: {shown_rung['verified_success']:.0%}\n\n"
                        "Press ENTER to answer questions.",
                    )
                    if self.display.wait_for_key() == self.display.K_ESCAPE:
                        raise _Aborted()
                    metrics.survey = self._iws_placeholder_survey(r)
                    self.logger.log("survey", {"round": r, **metrics.survey})
                elif (cfg.study.survey_enabled
                      and (r + 1) % max(1, cfg.study.survey_every_n_rounds) == 0):
                    metrics.survey = self.survey.run(
                        header=f"Check-in after attempt {r + 1}")
                    self.logger.log("survey", {"round": r, **metrics.survey})

                self.study.record(metrics)
                self.logger.log_round(metrics)

                feedback = self.study.progress_feedback(
                    r, metrics.success_rate, metrics.train_report)

                ckpt = os.path.join(self.session_dir, f"adapter_round_{r}.pth")
                trainer.save_adapter(ckpt, extra={"round": r,
                                                  "condition": plan.condition})
        except _Aborted:
            self.logger.log("aborted", {"where": "dagger_mode"})

        summary = self.study.summary()
        summary.update(mode="dagger", dataset=self.demo_path,
                       session_dir=self.session_dir)
        self.logger.write_summary(summary)
        self.display.message(
            "Session complete",
            "Thank you.\n\nYour data has been saved.",
            sub="press ENTER to exit")
        self.display.wait_for_key()
        return summary

    def _iws_placeholder_survey(self, round_index: int) -> Dict[str, int]:
        """Temporary survey: Question 1..5, scale 1..10 (0 = 10)."""
        import pygame
        answers = {}
        keys = [getattr(pygame, f"K_{n}") for n in range(1, 10)] + [pygame.K_0]
        for n in range(1, 6):
            while True:
                self.display.message(
                    f"Survey — Attempt {round_index + 1} of 4",
                    f"Question {n}\n\nTEST QUESTION\n\n"
                    "Please rate from 1 to 10",
                    sub="",
                )
                key = self.display.wait_for_key(keys + [pygame.K_ESCAPE])
                if key == pygame.K_ESCAPE:
                    raise _Aborted()
                if key in keys:
                    answers[f"question{n}"] = keys.index(key) + 1
                    break
        return answers

    def _train_progress(self, i: int, n: int, loss: float) -> None:
        bar = int(28 * i / max(n, 1))
        self.display.message(
            "Updating the robot's policy",
            "Learning from your corrections...\n\n"
            f"[{'#' * bar}{'.' * (28 - bar)}]",
            sub="")
        self._pump()

    def _pump(self) -> None:
        pump = getattr(self.display, "pump", None)
        if pump is not None:
            pump()

    def _wait_showing(self, title: str, body: str, seconds: float) -> None:
        """Hold a message on screen for `seconds`, keeping the window responsive."""
        end = time.time() + seconds
        while time.time() < end:
            self.display.message(title, body, sub="")
            self._pump()
            tick = getattr(getattr(self.display, "clock", None), "tick", None)
            if tick is not None:
                tick(20)
            else:
                time.sleep(0.02)
