"""Tests for T1.3: prompt_id provenance, resume, and confirmed-slate consume."""

import json

import pandas as pd
import pytest

from binary_classifier.annotate.run_annotation import (
    CANARY_AUDIT_FILENAME,
    resolve_production_selection,
    resolve_production_specs,
    run_annotation,
    run_annotation_matrix,
)
from binary_classifier.annotate.schema import BinaryLabel, LabelRecord, SourceType
from binary_classifier.config import BakeoffCandidate
from binary_classifier.qc.agreement import run_quality_check


class _StubAnnotator:
    """A no-network annotator that echoes its (spec, prompt_id) provenance."""

    def __init__(self, spec: BakeoffCandidate, prompt_id: str, prompt_text: str) -> None:
        self.spec = spec
        self.prompt_id = prompt_id

    def annotate(self, text: str, ein2: str = "") -> LabelRecord:
        return LabelRecord(
            EIN2=ein2,
            source_id=f"{self.spec.id}__{self.prompt_id}",
            source_type=SourceType.LLM_PROMPT,
            model_id=self.spec.id,
            prompt_id=self.prompt_id,
            temperature=0.0,
            binary_label=BinaryLabel.RELIGIOUS,
        )


def _stub_factory(spec, prompt_id, prompt_text):
    return _StubAnnotator(spec, prompt_id, prompt_text)


class _KeywordStubAnnotator:
    """A no-network annotator that makes validation labels deterministic.

    The T2 regression needs the stage-04 gate to pass without external LLM
    calls. Labeling ``church`` text as religious and everything else as
    nonreligious gives the fixture a tiny, fully controlled validation overlap.
    """

    def __init__(
        self,
        spec: BakeoffCandidate,
        prompt_id: str,
        prompt_text: str,
    ) -> None:
        """Keep model/prompt provenance for the gate-overlap fixture."""
        self.spec = spec
        self.prompt_id = prompt_id

    def annotate(self, text: str, ein2: str = "") -> LabelRecord:
        """Return a deterministic label from the mission text.

        This preserves the production resume key shape ``(EIN2, source_id)``
        while avoiding any API/GPU dependency in the union-manifest test.
        """
        label = (
            BinaryLabel.RELIGIOUS
            if "church" in text.lower()
            else BinaryLabel.NONRELIGIOUS
        )
        return LabelRecord(
            EIN2=ein2,
            source_id=f"{self.spec.id}__{self.prompt_id}",
            source_type=SourceType.LLM_PROMPT,
            model_id=self.spec.id,
            prompt_id=self.prompt_id,
            temperature=0.0,
            confidence=0.9,
            binary_label=label,
        )


def _keyword_stub_factory(_cfg, spec, prompt_id, prompt_text):
    """Mirror ``make_annotator`` while returning a deterministic test double."""
    return _KeywordStubAnnotator(spec, prompt_id, prompt_text)


def _write_prompts(tmp_path) -> list:
    tmp_path.mkdir(parents=True, exist_ok=True)
    paths = []
    for stem in ("v1", "v2", "v3"):
        p = tmp_path / f"{stem}.txt"
        p.write_text(f"prompt {stem}")
        paths.append(p)
    return paths


def _write_annotation_inputs(
    registry,
    cfg,
    silver_ein2s: list[str],
    gold_rows: list[dict[str, str]],
    monitor_ein2s: list[str] | None = None,
    texts: dict[str, str] | None = None,
) -> None:
    """Write minimal stage-03 manifests and mission text for canary tests.

    The canary tests exercise ``run_annotation`` end to end without stage 01, so
    they need the same silver/gold manifest shape and upstream parquet join that
    production stage 03 expects.
    """
    pd.DataFrame({"EIN2": silver_ein2s}).to_csv(
        registry.silver_manifest,
        index=False,
    )
    pd.DataFrame(gold_rows).to_csv(registry.gold_manifest, index=False)
    if monitor_ein2s is not None:
        pd.DataFrame({"EIN2": monitor_ein2s}).to_csv(
            registry.monitor_manifest,
            index=False,
        )

    texts = texts or {}
    mission_ein2s = sorted(set(silver_ein2s) | {row["EIN2"] for row in gold_rows})
    registry.missions_parquet.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "EIN2": mission_ein2s,
            cfg.field: [texts.get(ein2, "community support") for ein2 in mission_ein2s],
        }
    ).to_parquet(registry.missions_parquet, index=False)


class _SpyLimiter:
    """Context-manager probe for provider limiter wiring."""

    def __init__(self) -> None:
        self.entries = 0

    def __enter__(self):
        self.entries += 1
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        return None


def test_distinct_source_id_per_prompt(tmp_path) -> None:
    """One row under v1/v2/v3 yields three distinct source_ids."""
    prompts = _write_prompts(tmp_path)
    df = pd.DataFrame({"EIN2": ["00-1"], "text": ["a mission"]})
    specs = [BakeoffCandidate(id="m1", provider="vllm")]
    store = run_annotation_matrix(
        df=df,
        specs=specs,
        prompt_paths=prompts,
        store_path=tmp_path / "store.csv",
        annotator_factory=_stub_factory,
        checkpoint_every=1,
    )
    frame = store.to_frame()
    assert len(frame) == 3
    assert set(frame["source_id"]) == {"m1__v1", "m1__v2", "m1__v3"}
    assert set(frame["prompt_id"]) == {"v1", "v2", "v3"}


def test_resume_skips_done_no_duplicates(tmp_path) -> None:
    """A resume re-run adds no duplicate (EIN2, source_id) rows."""
    prompts = _write_prompts(tmp_path)
    df = pd.DataFrame({"EIN2": ["00-1", "00-2"], "text": ["m a", "m b"]})
    specs = [BakeoffCandidate(id="m1", provider="vllm")]
    store_path = tmp_path / "store.csv"

    run_annotation_matrix(
        df=df,
        specs=specs,
        prompt_paths=prompts,
        store_path=store_path,
        annotator_factory=_stub_factory,
        checkpoint_every=10,
    )
    # Second run with a fresh store object over the same file.
    store2 = run_annotation_matrix(
        df=df,
        specs=specs,
        prompt_paths=prompts,
        store_path=store_path,
        annotator_factory=_stub_factory,
        checkpoint_every=10,
        resume=True,
    )
    frame = store2.to_frame()
    # 2 rows × 3 prompts = 6, no duplicates after resume.
    assert len(frame) == 6
    assert not frame.duplicated(subset=["EIN2", "source_id"]).any()


def test_resume_normalizes_ein2_dtype_drift(tmp_path) -> None:
    """A CSV-inferred numeric EIN2 still matches a string EIN2 on resume."""
    prompts = _write_prompts(tmp_path)
    specs = [BakeoffCandidate(id="m1", provider="vllm")]
    store_path = tmp_path / "store.csv"

    run_annotation_matrix(
        df=pd.DataFrame({"EIN2": [123], "text": ["m a"]}),
        specs=specs,
        prompt_paths=prompts,
        store_path=store_path,
        annotator_factory=_stub_factory,
        checkpoint_every=10,
    )
    store = run_annotation_matrix(
        df=pd.DataFrame({"EIN2": [" 123 "], "text": ["m a"]}),
        specs=specs,
        prompt_paths=prompts,
        store_path=store_path,
        annotator_factory=_stub_factory,
        checkpoint_every=10,
        resume=True,
    )

    frame = store.to_frame()
    assert len(frame) == 3
    assert set(frame["EIN2"]) == {"123"}
    assert not frame.duplicated(subset=["EIN2", "source_id"]).any()


def test_no_resume_replaces_existing_store(tmp_path) -> None:
    """A start-from-scratch re-run does not append to stale labels."""
    prompts = _write_prompts(tmp_path)
    specs = [BakeoffCandidate(id="m1", provider="vllm")]
    store_path = tmp_path / "store.csv"

    run_annotation_matrix(
        df=pd.DataFrame({"EIN2": ["00-old", "00-keep"], "text": ["a", "b"]}),
        specs=specs,
        prompt_paths=prompts,
        store_path=store_path,
        annotator_factory=_stub_factory,
        checkpoint_every=10,
    )
    store = run_annotation_matrix(
        df=pd.DataFrame({"EIN2": ["00-new"], "text": ["c"]}),
        specs=specs,
        prompt_paths=prompts,
        store_path=store_path,
        annotator_factory=_stub_factory,
        checkpoint_every=10,
        resume=False,
    )

    frame = store.to_frame()
    assert len(frame) == 3
    assert set(frame["EIN2"]) == {"00-new"}


def test_run_annotation_matrix_wraps_calls_with_provider_limiters(tmp_path) -> None:
    """Stage 03 applies each provider limiter on the worker annotate path."""
    prompts = [_write_prompts(tmp_path)[0]]
    specs = [
        BakeoffCandidate(id="m-openai", provider="openai"),
        BakeoffCandidate(id="m-vllm", provider="vllm"),
    ]
    openai_spy = _SpyLimiter()
    vllm_spy = _SpyLimiter()

    store = run_annotation_matrix(
        df=pd.DataFrame(
            {
                "EIN2": ["00-1", "00-2"],
                "text": ["mission a", "mission b"],
            }
        ),
        specs=specs,
        prompt_paths=prompts,
        store_path=tmp_path / "store.csv",
        annotator_factory=_stub_factory,
        checkpoint_every=10,
        provider_limiters={"openai": openai_spy, "vllm": vllm_spy},
    )

    assert len(store.to_frame()) == 4
    assert openai_spy.entries == 2
    assert vllm_spy.entries == 2


def test_run_annotation_includes_gold_validation_for_stage04_gate(
    tiny_config,
    tiny_registry,
    tmp_path,
    monkeypatch,
) -> None:
    """Validation EIN2s outside silver are annotated for the stage-04 gate.

    Silver and gold are independent draws, so a realistic validation split can
    have no pre-existing silver overlap. Stage 03 must annotate silver plus gold
    so stage 04 has LLM labels to compare against the human-coded validation
    rows instead of failing with an empty inner join.
    """
    prompts = _write_prompts(tmp_path)
    specs = [BakeoffCandidate(id="m1", provider="vllm")]
    silver_ein2s = {"00-S1"}
    validation_ein2s = {"00-V0", "00-V1"}
    assert silver_ein2s.isdisjoint(validation_ein2s)

    pd.DataFrame({"EIN2": sorted(silver_ein2s)}).to_csv(
        tiny_registry.silver_manifest,
        index=False,
    )
    pd.DataFrame(
        {
            "EIN2": ["00-V1", "00-V0"],
            "split": ["validation", "validation"],
        }
    ).to_csv(tiny_registry.gold_manifest, index=False)

    tiny_registry.missions_parquet.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "EIN2": ["00-S1", "00-V1", "00-V0"],
            tiny_config.field: [
                "community support",
                "church worship services",
                "food pantry",
            ],
        }
    ).to_parquet(tiny_registry.missions_parquet, index=False)

    pd.DataFrame(
        [
            {"EIN2": "00-V1", "split": "validation", "human_label": 1},
            {"EIN2": "00-V0", "split": "validation", "human_label": 0},
        ]
    ).to_csv(tiny_registry.gold_coding_template, index=False)

    monkeypatch.setattr(
        "binary_classifier.annotate.run_annotation.make_annotator",
        _keyword_stub_factory,
    )
    store = run_annotation(
        cfg=tiny_config,
        registry=tiny_registry,
        prompt_paths=prompts,
        specs=specs,
        checkpoint_every=100,
    )
    frame = store.to_frame()
    assert validation_ein2s.issubset(set(frame["EIN2"]))

    result = run_quality_check(tiny_config, tiny_registry)
    assert result["n_valid"] == len(validation_ein2s)


def test_run_annotation_limit_disables_openai_batch(
    tiny_config,
    tiny_registry,
    tmp_path,
    monkeypatch,
) -> None:
    """Smoke-limited Stage 03 runs stay live even when production batches."""
    tiny_config.annotation.openai_batch = True
    prompt = _write_prompts(tmp_path)[0]
    _write_annotation_inputs(
        registry=tiny_registry,
        cfg=tiny_config,
        silver_ein2s=["00-S1"],
        gold_rows=[{"EIN2": "00-V1", "split": "validation"}],
        texts={"00-S1": "community support", "00-V1": "church worship"},
    )

    def fail_batch_group(**kwargs):
        raise AssertionError("batch path should not run for limited annotation")

    monkeypatch.setattr(
        "binary_classifier.annotate.run_annotation._run_openai_batch_group",
        fail_batch_group,
    )
    monkeypatch.setattr(
        "binary_classifier.annotate.run_annotation.make_annotator",
        _keyword_stub_factory,
    )

    store = run_annotation(
        cfg=tiny_config,
        registry=tiny_registry,
        limit=1,
        prompt_paths=[prompt],
        specs=[BakeoffCandidate(id="gpt-4o-mini-2024-07-18", provider="openai")],
    )

    assert len(store.to_frame()) == 1


def test_run_annotation_canary_uses_monitor_manifest_only(
    tiny_config,
    tiny_registry,
    tmp_path,
    monkeypatch,
) -> None:
    """Canary mode annotates exactly the held-out monitor EIN2s.

    The monitor slice is separate from prompt-dev/validation/test and is used
    only for drift monitoring, so non-monitor silver/gold rows must not appear
    in the canary label store.
    """
    prompts = _write_prompts(tmp_path)
    specs = [BakeoffCandidate(id="m1", provider="vllm")]
    monitor_ein2s = ["00-M1", "00-M0"]
    _write_annotation_inputs(
        registry=tiny_registry,
        cfg=tiny_config,
        silver_ein2s=["00-S1"],
        gold_rows=[
            {"EIN2": "00-M1", "split": "monitor"},
            {"EIN2": "00-M0", "split": "monitor"},
            {"EIN2": "00-V1", "split": "validation"},
        ],
        monitor_ein2s=monitor_ein2s,
        texts={
            "00-M1": "church worship services",
            "00-M0": "food pantry",
            "00-S1": "community support",
            "00-V1": "church school",
        },
    )

    monkeypatch.setattr(
        "binary_classifier.annotate.run_annotation.make_annotator",
        _keyword_stub_factory,
    )
    store = run_annotation(
        cfg=tiny_config,
        registry=tiny_registry,
        prompt_paths=prompts,
        specs=specs,
        checkpoint_every=100,
        resume=False,
        canary_only=True,
    )

    frame = store.to_frame()
    assert set(frame["EIN2"]) == set(monitor_ein2s)
    assert len(frame) == len(monitor_ein2s) * len(prompts) * len(specs)

    audit_path = tiny_registry.interim_dir / CANARY_AUDIT_FILENAME
    audit = json.loads(audit_path.read_text().splitlines()[0])
    assert audit["monitor_manifest_n"] == len(monitor_ein2s)
    assert audit["canary_pool_n"] == len(monitor_ein2s)
    assert audit["canary_set_version"].startswith("sha256:")
    assert audit["kappa_alpha_change_test"]["status"] == "baseline"
    assert audit["model_fingerprints"][0]["pinned_snapshot_id"] == "m1"
    assert audit["model_fingerprints"][0]["seed"] == tiny_config.SEED
    assert (
        audit["model_fingerprints"][0]["temperature"]
        == tiny_config.annotation.temperature
    )


def test_run_annotation_canary_disables_openai_batch(
    tiny_config,
    tiny_registry,
    tmp_path,
    monkeypatch,
) -> None:
    """Canary drift checks stay live even when production OpenAI batches."""
    tiny_config.annotation.openai_batch = True
    prompt = _write_prompts(tmp_path)[0]
    _write_annotation_inputs(
        registry=tiny_registry,
        cfg=tiny_config,
        silver_ein2s=["00-S1"],
        gold_rows=[{"EIN2": "00-M1", "split": "monitor"}],
        monitor_ein2s=["00-M1"],
        texts={"00-M1": "church worship", "00-S1": "community support"},
    )

    def fail_batch_group(**kwargs):
        raise AssertionError("batch path should not run for canary annotation")

    monkeypatch.setattr(
        "binary_classifier.annotate.run_annotation._run_openai_batch_group",
        fail_batch_group,
    )
    monkeypatch.setattr(
        "binary_classifier.annotate.run_annotation.make_annotator",
        _keyword_stub_factory,
    )

    store = run_annotation(
        cfg=tiny_config,
        registry=tiny_registry,
        prompt_paths=[prompt],
        specs=[BakeoffCandidate(id="gpt-4o-mini-2024-07-18", provider="openai")],
        canary_only=True,
    )

    assert set(store.to_frame()["EIN2"]) == {"00-M1"}


def test_run_annotation_canary_missing_monitor_manifest_raises(
    tiny_config,
    tiny_registry,
    tmp_path,
) -> None:
    """Canary mode fails clearly when stage 01 has not written monitor rows."""
    prompts = _write_prompts(tmp_path)
    specs = [BakeoffCandidate(id="m1", provider="vllm")]
    _write_annotation_inputs(
        registry=tiny_registry,
        cfg=tiny_config,
        silver_ein2s=["00-S1"],
        gold_rows=[{"EIN2": "00-V1", "split": "validation"}],
    )

    with pytest.raises(
        FileNotFoundError,
        match="run stage 01 first; monitor manifest lives under the cloud-symlinked interim tree",
    ):
        run_annotation(
            cfg=tiny_config,
            registry=tiny_registry,
            prompt_paths=prompts,
            specs=specs,
            canary_only=True,
        )


def test_run_annotation_canary_empty_pool_overlap_raises(
    tiny_config,
    tiny_registry,
    tmp_path,
) -> None:
    """Canary mode hard-fails instead of silently doing no work."""
    prompts = _write_prompts(tmp_path)
    specs = [BakeoffCandidate(id="m1", provider="vllm")]
    _write_annotation_inputs(
        registry=tiny_registry,
        cfg=tiny_config,
        silver_ein2s=["00-S1"],
        gold_rows=[{"EIN2": "00-V1", "split": "validation"}],
        monitor_ein2s=["00-MISSING"],
    )

    with pytest.raises(ValueError, match="do not overlap the annotation pool"):
        run_annotation(
            cfg=tiny_config,
            registry=tiny_registry,
            prompt_paths=prompts,
            specs=specs,
            canary_only=True,
        )


# ── Confirmed production slate (gate G2 backstop) ────────────────────────────


def _write_slate(
    registry,
    confirmed: bool,
    models: list[dict],
    selected: list[dict] | None = None,
) -> None:
    registry.production_slate.parent.mkdir(parents=True, exist_ok=True)
    registry.production_slate.write_text(
        json.dumps({"confirmed": confirmed, "models": models, "selected": selected or []})
    )


def test_resolve_production_specs_returns_listed_models(tiny_registry) -> None:
    """The production set is exactly what production_slate.json lists."""
    _write_slate(
        tiny_registry,
        confirmed=True,
        models=[
            {"id": "gpt-5-mini", "provider": "openai", "reasoning_effort": "minimal"},
            {"id": "google/gemma-3-27b-it", "provider": "vllm"},
        ],
    )
    specs = resolve_production_specs(tiny_registry)
    assert [s.id for s in specs] == ["gpt-5-mini", "google/gemma-3-27b-it"]
    assert specs[0].reasoning_effort == "minimal"


def test_resolve_production_specs_missing_raises(tiny_registry) -> None:
    with pytest.raises(FileNotFoundError):
        resolve_production_specs(tiny_registry)


def test_resolve_production_specs_unconfirmed_raises(tiny_registry) -> None:
    _write_slate(tiny_registry, confirmed=False, models=[{"id": "m", "provider": "vllm"}])
    with pytest.raises(ValueError, match="not confirmed"):
        resolve_production_specs(tiny_registry)


def test_resolve_production_specs_empty_raises(tiny_registry) -> None:
    _write_slate(tiny_registry, confirmed=True, models=[])
    with pytest.raises(ValueError, match="no models"):
        resolve_production_specs(tiny_registry)


def test_run_annotation_uses_selected_model_prompt_pairs(
    tiny_config,
    tiny_registry,
    monkeypatch,
) -> None:
    """Selected bake-off pairs constrain stage-03 production annotation."""
    _write_prompts(tiny_registry.prompts_dir)
    _write_annotation_inputs(
        registry=tiny_registry,
        cfg=tiny_config,
        silver_ein2s=["00-S1"],
        gold_rows=[{"EIN2": "00-V1", "split": "validation"}],
    )
    _write_slate(
        tiny_registry,
        confirmed=True,
        models=[
            {"id": "m1", "provider": "vllm"},
            {"id": "m2", "provider": "vllm"},
        ],
        selected=[
            {"model_id": "m1", "prompt_id": "v2"},
            {"model_id": "m2", "prompt_id": "v1"},
        ],
    )
    monkeypatch.setattr(
        "binary_classifier.annotate.run_annotation.make_annotator",
        _keyword_stub_factory,
    )

    store = run_annotation(
        cfg=tiny_config,
        registry=tiny_registry,
        checkpoint_every=100,
    )

    frame = store.to_frame()
    assert set(frame["source_id"]) == {"m1__v2", "m2__v1"}
    assert len(frame) == 4


def test_run_annotation_rejects_invalid_selected_pairs(tiny_registry) -> None:
    """A malformed selected slate must not fall back to all default prompts."""
    _write_slate(
        tiny_registry,
        confirmed=True,
        models=[{"id": "m1", "provider": "vllm"}],
        selected=[{"model_id": "typo", "prompt_id": "v1"}],
    )

    with pytest.raises(ValueError, match="invalid model/prompt"):
        resolve_production_selection(tiny_registry)
