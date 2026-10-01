"""Tests for systemone.loop — See > Decide > Act, the one agent loop.

No model needed — judges and envs are scripted.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from systemone.loop import (  # noqa: E402
    ActResult,
    DecisionLoop,
    Observation,
    run_loop,
)


def _answers(choice="right", conf=0.9, mass=None, latency=1.0):
    return {
        "action": {"type": "choice", "choice": choice,
                   "probabilities": {choice: conf},
                   "confidence": conf, "label_mass": mass},
        "_meta": {"backend": "stub", "latency_ms": latency},
    }


class ScriptEnv:
    """Env scripted per tick: (progressed, done) pairs."""

    def __init__(self, script, obs_text="here"):
        self.script = list(script)
        self.obs_text = obs_text
        self.actions = []

    def observe(self):
        return Observation(text=self.obs_text)

    def act(self, action):
        self.actions.append(action)
        progressed, done = self.script.pop(0) if self.script else (True, True)
        return ActResult(progressed=progressed, done=done)


def _judge(choice="right", conf=0.9, mass=None):
    def judge(state_text, questions, images, videos):
        assert isinstance(state_text, str) and state_text
        assert isinstance(images, list) and isinstance(videos, list)
        return _answers(choice, conf, mass)
    return judge


def _questions():
    return [{"name": "action", "type": "choice",
             "options": ["up", "right", "wait"]}]


def test_loop_reaches_done():
    env = ScriptEnv([(True, False), (True, True)])
    result = run_loop(env, _judge(), _questions(), budget=5)
    assert result.outcome == "done"
    assert result.n_ticks == 2
    assert env.actions == ["right", "right"]
    assert len(result.memory) == 2
    assert "I saw" in result.memory[0]


def test_uncertainty_gate_falls_back_to_wait():
    env = ScriptEnv([(True, True)])
    result = run_loop(env, _judge(conf=0.1), _questions(), budget=5)
    assert result.steps[0].gated is True
    assert env.actions == ["wait"]


def test_label_mass_gate_trips_for_sglang_style_judges():
    env = ScriptEnv([(True, True)])
    result = run_loop(env, _judge(conf=0.9, mass=0.2), _questions(), budget=5)
    assert result.steps[0].gated is True
    assert env.actions == ["wait"]


def test_stall_guard_stops_spin():
    env = ScriptEnv([(False, False)] * 10)
    result = run_loop(env, _judge(), _questions(), budget=10)
    assert result.outcome == "stalled"
    assert result.n_ticks == 3  # max_stalls default


def test_budget_exhaustion_recorded():
    env = ScriptEnv([(True, False)] * 10)
    result = run_loop(env, _judge(), _questions(), budget=4)
    assert result.outcome == "budget"
    assert result.n_ticks == 4


def test_escalation_without_system2_is_recorded_not_fatal():
    env = ScriptEnv([(True, True)])
    result = run_loop(env, _judge(conf=0.5), _questions(), budget=5)
    step = result.steps[0]
    assert step.escalated is False
    assert "no system2" in (step.escalation or "")
    assert env.actions == ["right"]  # System 1 action kept


def test_system2_override_wins():
    env = ScriptEnv([(True, True)])

    def system2(state_text, options):
        assert "Memory (compressed turns)" in state_text
        return "up"

    result = run_loop(env, _judge(conf=0.5), _questions(), budget=5,
                      system2=system2)
    step = result.steps[0]
    assert step.escalated is True
    assert env.actions == ["up"]


def test_system2_failure_fails_open():
    env = ScriptEnv([(True, True)])

    def system2(state_text, options):
        raise RuntimeError("boom")

    result = run_loop(env, _judge(conf=0.5), _questions(), budget=5,
                      system2=system2)
    assert result.steps[0].escalated is False
    assert "failed" in (result.steps[0].escalation or "")
    assert env.actions == ["right"]


def test_gated_steps_skip_escalation():
    seen = []

    def system2(state_text, options):
        seen.append(True)
        return "up"

    env = ScriptEnv([(True, True)])
    run_loop(env, _judge(conf=0.1), _questions(), budget=5, system2=system2)
    assert seen == []  # gate tripped first; no escalation call


def test_state_text_prefix_cache_shape():
    loop = DecisionLoop(_judge(), system_prompt="SYS", memory_lines=2)
    text = loop.build_state_text(["m1", "m2", "m3"], "OBS", 7)
    assert text.startswith("SYS\n\nMemory (compressed turns):\nm2\nm3\n\n")
    assert text.endswith("Tick 7 — current observation:\nOBS")


def test_videos_reach_judge():
    seen = []

    def judge(state_text, questions, images, videos):
        seen.append((images, videos))
        return _answers()

    class VidEnv(ScriptEnv):
        def observe(self):
            return Observation(text="clip", videos=["https://e.com/v.mp4"])

    run_loop(VidEnv([(True, True)]), judge, _questions(), budget=2)
    assert seen[0] == ([], ["https://e.com/v.mp4"])


def test_step_records_telemetry():
    env = ScriptEnv([(True, True)])
    result = run_loop(env, _judge(), _questions(), budget=5)
    step = result.steps[0]
    assert step.latency_ms == 1.0
    assert step.backend == "stub"
    assert step.confidence == 0.9
    assert step.status == "ok"
