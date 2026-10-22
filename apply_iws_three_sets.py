"""Patch existing dagger_lh/runner.py for three independent four-round I/W/S sets.

Run in project root. Creates backup; can be run repeatedly without duplicate edits.
"""
from pathlib import Path
import ast
import shutil

path = Path('dagger_lh/runner.py')
if not path.is_file():
    raise SystemExit('Run this from the dagger_human_effects project root.')
s = path.read_text(encoding='utf-8')
if 'IWS_THREE_SETS_V2' in s:
    print('Already installed; no change.')
    raise SystemExit(0)

# Pre-existing policy pages, if installed, must be removed to avoid duplicate prompts.
for marker in ('IWS_POLICY_SCREEN_V1', 'PICKPLACE_LADDER_UI_V1'):
    if marker in s:
        start = s.index('                # ' + marker)
        end = s.index('                # ---- rollouts ----', start)
        s = s[:start] + s[end:]

start_tag = '                # ---- policy update (contingency manipulation lives here) ----'
end_tag = '                self.study.record(metrics)'
start = s.find(start_tag)
end = s.find(end_tag, start)
if start < 0 or end < 0:
    raise SystemExit('Could not identify update/survey section; no files changed.')
replacement = '''                # IWS_THREE_SETS_V2: Only swap pretrained policies for I/W/S.
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
                        "Learning from your corrections...\\n\\nplease wait", sub="")
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
                                "Learning from your corrections...\\n\\nplease wait",
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
                        f"{subtitle}\\n\\n"
                        f"Current policy success rate: {shown_rung['verified_success']:.0%}\\n\\n"
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

'''
s = s[:start] + replacement + s[end:]

# Strip existing inter-round page (including previous UI patch), since policy
# page and the survey now precede the next round.
inter_start = '                if r < self.study.total_rounds - 1:'
inter_end = '                ckpt = os.path.join(self.session_dir, f"adapter_round_{r}.pth")'
a = s.find(inter_start, s.find('                self.study.record(metrics)'))
b = s.find(inter_end, a)
if a < 0 or b < 0:
    raise SystemExit('Could not identify old inter-round message; no files changed.')
s = s[:a] + s[b:]

# First-round policy information screen and survey helper.
marker = '                # ---- rollouts ----'
if s.count(marker) != 1:
    raise SystemExit('Missing unique rollout anchor; no files changed.')
s = s.replace(marker, '''                if r == 0 and plan.ladder_target is not None and self.ladder is not None:
                    first_rung = self.ladder.nearest(plan.ladder_target)
                    self.display.message(
                        "Robot Policy — Attempt 1 of 4",
                        f"Current policy success rate: {first_rung['verified_success']:.0%}\\n\\n"
                        "Press ENTER to begin.",
                    )
                    if self.display.wait_for_key() == self.display.K_ESCAPE:
                        raise _Aborted()

''' + marker, 1)
helper_anchor = '    def _train_progress(self, i: int, n: int, loss: float) -> None:'
if s.count(helper_anchor) != 1:
    raise SystemExit('Missing survey insertion anchor; no files changed.')
helper = '''    def _iws_placeholder_survey(self, round_index: int) -> Dict[str, int]:
        """Temporary survey: Question 1..5, scale 1..10 (0 = 10)."""
        import pygame
        answers = {}
        keys = [getattr(pygame, f"K_{n}") for n in range(1, 10)] + [pygame.K_0]
        for n in range(1, 6):
            while True:
                self.display.message(
                    f"Survey — Attempt {round_index + 1} of 4",
                    f"Question {n}\\n\\nRate from 1 to 10.\\n\\n"
                    "Rate this question on a scale from 1 to 10, where 1 is the lowest and 10 is the highest.\\n\\n",
                    sub="",
                )
                key = self.display.wait_for_key(keys + [pygame.K_ESCAPE])
                if key == pygame.K_ESCAPE:
                    raise _Aborted()
                if key in keys:
                    answers[f"question{n}"] = keys.index(key) + 1
                    break
        return answers

'''
s = s.replace(helper_anchor, helper + helper_anchor, 1)
ast.parse(s, filename=str(path))
backup = path.with_name('runner.py.before_iws_three_sets.bak')
if backup.exists():
    raise SystemExit(f'Backup exists already: {backup}. No files changed.')
shutil.copy2(path, backup)
path.write_text(s, encoding='utf-8')
print('Patched', path, '| backup:', backup)
