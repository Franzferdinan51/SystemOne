"""Tests for systemone.model_lifecycle."""

import time

from systemone import model_lifecycle as ml


def _reset():
    ml._usage.clear()
    ml._sessions.clear()


def test_track_model_use():
    _reset()
    ml.track_model_use("model-a")
    assert "model-a" in ml._usage
    assert ml._usage["model-a"] <= time.time()


def test_track_with_session():
    _reset()
    ml.track_model_use("model-a", session_id="s1")
    ml.track_model_use("model-b", session_id="s1")
    assert ml._sessions["s1"] == {"model-a", "model-b"}


def test_release_session_unloads_orphans(monkeypatch):
    _reset()
    unloaded = []
    monkeypatch.setattr(ml, "_unload_model", lambda m: unloaded.append(m) or True)
    ml.track_model_use("model-a", session_id="s1")
    ml.track_model_use("model-b", session_id="s2")
    # Make model-a idle
    ml._usage["model-a"] = time.time() - 120
    ml.release_session_models("s1")
    assert "s1" not in ml._sessions
    assert "model-a" in unloaded  # only s1 used it, now idle
    assert "model-b" not in unloaded  # s2 still needs it


def test_release_keeps_shared_models(monkeypatch):
    _reset()
    unloaded = []
    monkeypatch.setattr(ml, "_unload_model", lambda m: unloaded.append(m) or True)
    ml.track_model_use("shared", session_id="s1")
    ml.track_model_use("shared", session_id="s2")
    ml._usage["shared"] = time.time() - 120
    ml.release_session_models("s1")
    assert "shared" not in unloaded  # s2 still uses it


def test_reap_idle(monkeypatch):
    _reset()
    unloaded = []
    monkeypatch.setattr(ml, "_unload_model", lambda m: unloaded.append(m) or True)
    ml.track_model_use("old-model")
    ml._usage["old-model"] = time.time() - 3600
    ml.track_model_use("fresh-model")
    n = ml.reap_idle(idle_timeout_s=900)
    assert n == 1
    assert "old-model" in unloaded
    assert "fresh-model" not in unloaded


def test_unload_all(monkeypatch):
    _reset()
    monkeypatch.setattr(ml, "_unload_model", lambda m: True)
    ml.track_model_use("a")
    ml.track_model_use("b", session_id="s1")
    n = ml.unload_all()
    assert n == 2
    assert not ml._usage
    assert not ml._sessions
