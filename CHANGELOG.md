# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

## [Unreleased]

### Added

- `systemone.model_lifecycle`: model VRAM lifecycle management. Tracks
  model usage on every route decision and reaps idle models via `lms unload`
  (the only reliable LM Studio unload interface — no stable HTTP endpoint
  exists). Fixes the 128GB VRAM blowout where hung/ended sessions left 15+
  models resident. Includes `track_model_use()`, `release_session_models()`
  (call on session end), `reap_idle()`, `start_reaper()` background thread
  (default 15min idle timeout), and `unload_all()` emergency flush.
  Env overrides: `SYSTEMONE_IDLE_TIMEOUT_S`, `SYSTEMONE_REAPER_INTERVAL_S`,
  `SYSTEMONE_NO_REAPER`, `SYSTEMONE_LMS_BIN`.

## [0.2.0] - 2026-10-02

Ecosystem pull: cascades, ensembles, conformal sets, route controls, bulk
ops, monitoring, and a red-team battery — plus the Cloudflare Clef engine.

### Added

- `CascadeBackend` (FrugalGPT-style): N-stage budgeted cascade over any
  engines, whole-call escalation under a confidence bar, per-call stage /
  cost / budget accounting in `_meta`, fail-open on stage errors.
- `EnsembleBackend`: multi-engine fusion with `average`, `extremized`,
  `vote`, `rrf`, and `borda` strategies; honest mean probabilities under
  rank fusion, mean probability/distribution for noul/score.
- `SelfConsistent`: majority vote over sampled (shuffled) judgments with
  an `agreement` disagreement signal.
- Split-conformal prediction sets (MAPIE/APS method, numpy only):
  `fit_conformal_threshold` / `conformal_set`, and the
  `ConformalChoice` wrapper adding `prediction_set` + `coverage` to
  choice answers.
- RouteLLM-style offline threshold picking: `route_threshold_for_target`
  (cheapest α holding target × strong quality) and
  `quality_cost_frontier`.
- Route controls (OpenRouter-style, all additive): `sort`
  (utility/quality/cost/latency), ordered `fallbacks`, epsilon-greedy
  `explore` with `explored` flag.
- `POST /v1/systemone/batch`: judge up to 32 TypeSafe bodies per call
  with per-item `{status, ...}` results; `SystemOneClient.batch()`.
- Exact-match decision cache (`SYSTEMONE_CACHE_TTL` / `SYSTEMONE_CACHE_MAX`);
  hits return `"cached": true` without touching the engine.
- Hash-chained audit log (`SYSTEMONE_AUDIT_CHAIN=1`) with
  `verify_audit_chain()` replay verification.
- PII scrubbing (`SYSTEMONE_SCRUB_PII=1`): typed redaction of emails,
  phones, SSNs, cards, API keys, IPv4 before judging.
- Eval honesty batch in `metrics.py`: `ndcg_at_k`, `reciprocal_rank`,
  `macro_f1`, `failure_auroc`, `brier_decomposition`,
  `reliability_curve`, `mcnemar`, `paired_bootstrap_ci`.
- Label-free monitoring in `metrics.py`: `psi` drift index and
  `estimated_accuracy` (NannyML CBPE-style).
- Red-team battery (`systemone/battery/redteam.jsonl`): 14 adversarial
  flip attempts (override, roleplay, urgency, smuggling, PII); 14/14
  honest tiers against the live ONNX judge. ECE gate now needs n ≥ 30
  (`MIN_ECE_N`) so small sets report without gating.
- `ClefBackend` (`SYSTEMONE_ENGINE=clef`): Cloudflare clef/clef-flash
  weights run locally through the release's own `joint_schema_model`,
  multimodal, `systemone[clef]` extra.
- `CHANGELOG.md`, release workflow (tag → wheel/sdist → Sigstore
  attestation → SBOM → GitHub Release → PyPI trusted publishing), and
  `Dockerfile` (slim CPU + CUDA variants).

### Changed

- `rank_models` takes `sort` and reports `latency_ms_p50` per model.
- Route responses gain `fallbacks` and `explored` (scoring path only).
- OpenAPI spec documents `/v1/systemone/batch`.

## [0.1.0] - 2026-09-24

Initial release: slim numpy-only base with local GLiClass, SGLang,
JEVK5, ONNX, JEV, and Kev engines; TypeSafe/SGLang/JEV dialects;
route/rank-plans/decide/permute endpoints; See > Decide > Act loop;
MCP/ACP servers; JevBench adapter; calibration battery.
