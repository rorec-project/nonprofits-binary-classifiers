"""Training sweep enumeration, execution, and model-selection reports (stage 06).

Enumerates the documentation-curve, arm-matrix, and final-seed runs that
constitute the stage-06 experiment grid.  The selection report aggregates
across seeds and recommends a cell (model, targets, arm) using validation
PR-AUC with a tie-to-simpler rule (He et al. 2021/2023, DeBERTa-v3,
arXiv:2006.03654; Warner et al. 2024, ModernBERT, arXiv:2412.13663).
Baselines include a MiniLM embedding + logistic-regression arm (Reimers &
Gurevych 2019, https://doi.org/10.18653/v1/D19-1410; Wang et al. 2020, MiniLM)
and a TF-IDF + logistic-regression arm.  Noisy-label robustness is explored
through soft targets, class-weighted CE, and the pruned arm (Zhu et al. 2022,
https://doi.org/10.18653/v1/2022.insights-1.8; Wang et al. 2023).

References:
    - He et al. (2021), "DeBERTa: Decoding-enhanced BERT with Disentangled
      Attention", ICLR. arXiv:2006.03654
    - Warner et al. (2024), "ModernBERT: The Long-Pretraining、Short-Context
      Revolution", arXiv:2412.13663
    - Reimers & Gurevych (2019), "Sentence-BERT: Sentence Embeddings using
      Siamese BERT-Networks", EMNLP-IJCNLP.
      https://doi.org/10.18653/v1/D19-1410
    - Zhu et al. (2022), "Is BERT Robust to Label Noise? A Study on Learning
      with Noisy Labels in Text Classification", Insights.
      https://doi.org/10.18653/v1/2022.insights-1.8

"""

from __future__ import annotations

import concurrent.futures
import hashlib
import json
import logging
import re
import threading
from collections.abc import Iterable, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter
from typing import TYPE_CHECKING, Any, Literal, cast

import numpy as np
import pandas as pd

from binary_classifier.train import encoder as encoder_mod
from binary_classifier.train.arms import ArmName, run_arm
from binary_classifier.train.baselines import minilm_logreg, tfidf_logreg
from binary_classifier.train.data import subset_fraction

if TYPE_CHECKING:
    from collections.abc import Iterator

    from binary_classifier.config import BinaryClassifierConfig, EncoderArm
    from binary_classifier.paths import PathRegistry

logger = logging.getLogger(__name__)

finetune = encoder_mod.finetune

RunKind = Literal["baseline", "encoder"]
RunPhase = Literal["baseline", "curve", "sweep", "final"]


@dataclass(frozen=True)
class RunSpec:
    """One planned training-stage run.

    Args:
        kind: ``"baseline"`` or ``"encoder"``.
        phase: Sweep phase that owns the run.
        run_id: Stable run identifier used for resume checks.
        seed: Random seed recorded in the run row.
        train_fraction: Fraction of the training frame consumed.
        model: Baseline name or encoder id.
        targets: Target semantics recorded in result rows.
        arm: Training arm recorded in result rows.
        encoder: Encoder configuration for encoder fine-tunes.

    """

    kind: RunKind
    phase: RunPhase
    run_id: str
    seed: int
    train_fraction: float
    model: str
    targets: str
    arm: str
    encoder: EncoderArm | None = None

    @property
    def cell_key(self) -> tuple[str, str, str]:
        """Return the model-selection cell key."""
        return (self.model, self.targets, self.arm)


# ── Run-matrix enumeration ───────────────────────────────────────────────────
#
# The PR-2 grid documents data-size sensitivity (documentation curve), arm
# comparisons (soft vs hard vs class-weighted vs pruned), and encoder
# head-to-heads, then final-seed refits the recommended cell.  Baselines
# (TF-IDF + logreg, MiniLM + logreg) anchor the lower bound of the sweep.


def build_run_matrix(
    cfg: BinaryClassifierConfig,
    *,
    baselines_only: bool = False,
    sweep: bool = True,
    final: bool = False,
    encoder: str | None = None,
    subset: float | None = None,
    recommendation: Mapping[str, Any] | None = None,
) -> list[RunSpec]:
    """Enumerate the stage-06 run matrix.

    Args:
        cfg: Validated task configuration.
        baselines_only: If ``True``, enumerate only configured baselines.
        sweep: Include documentation-curve and arm-matrix encoder runs.
        final: Include final-seed refits for the recommended cell.
        encoder: Optional encoder id filter for local runs.
        subset: Optional fraction override applied to encoder training runs.
        recommendation: Selection-report recommendation used for final runs.

    Returns:
        Ordered run specs matching the PR-2 matrix.

    """
    specs: list[RunSpec] = []
    specs.extend(_baseline_specs(cfg))
    if baselines_only:
        return _filter_specs(specs, encoder)
    if not sweep and not final:
        return _filter_specs(specs, encoder)

    encoders = _filter_encoders(cfg.training.encoders, encoder)
    primary = _primary_encoder(encoders)
    comparison = [arm for arm in encoders if arm.id != primary.id]
    base_fraction = _validated_fraction(subset) if subset is not None else None

    if sweep:
        curve_seed = int(cfg.SEED)
        for enc in encoders:
            for fraction in cfg.training.curve_fractions:
                run_fraction = base_fraction or _validated_fraction(fraction)
                specs.append(
                    _encoder_spec(
                        phase="curve",
                        encoder=enc,
                        targets="soft",
                        arm="default",
                        train_fraction=run_fraction,
                        seed=curve_seed,
                    ),
                )

        matrix_fraction = base_fraction or 1.0
        primary_cells = [
            ("soft", "default"),
            *(_arm_cell(arm) for arm in cfg.training.arms),
        ]
        for targets, arm in primary_cells:
            for seed in cfg.training.sweep_seeds:
                specs.append(
                    _encoder_spec(
                        phase="sweep",
                        encoder=primary,
                        targets=targets,
                        arm=arm,
                        train_fraction=matrix_fraction,
                        seed=int(seed),
                    ),
                )
        for enc in comparison:
            for seed in cfg.training.sweep_seeds:
                specs.append(
                    _encoder_spec(
                        phase="sweep",
                        encoder=enc,
                        targets="soft",
                        arm="default",
                        train_fraction=matrix_fraction,
                        seed=int(seed),
                    ),
                )

    if final:
        if recommendation is None:
            logger.warning("No recommendation available; skipping final-seed specs")
        else:
            final_encoder = _encoder_by_id(encoders, str(recommendation["encoder_id"]))
            final_fraction = base_fraction or 1.0
            for seed in cfg.training.final_seeds:
                specs.append(
                    _encoder_spec(
                        phase="final",
                        encoder=final_encoder,
                        targets=str(recommendation["targets"]),
                        arm=str(recommendation["arm"]),
                        train_fraction=final_fraction,
                        seed=int(seed),
                    ),
                )
    return _filter_specs(specs, encoder)


# ── Run execution and resume ─────────────────────────────────────────────────
#
# Baselines run in parallel (thread pool, max 2 workers); encoder fine-tunes
# run sequentially to avoid GPU contention.  Each run writes a
# ``metrics.json`` sentinel so resume is per-spec, not per-matrix.  The
# orchestrator calls this after the run matrix has been enumerated.


def execute_run_matrix(
    cfg: BinaryClassifierConfig,
    registry: PathRegistry,
    specs: Sequence[RunSpec],
    train_df: pd.DataFrame,
    dev_df: pd.DataFrame,
    validation_df: pd.DataFrame,
) -> list[dict[str, Any]]:
    """Execute run specs, appending each newly completed row to JSONL.

    Args:
        cfg: Validated task configuration.
        registry: Path registry that owns run artifacts.
        specs: Ordered run specifications.
        train_df: Silver training partition.
        dev_df: Silver development partition.
        validation_df: Human-validation rows, or a synthetic substitute for smoke.

    Returns:
        Rows loaded from skipped metrics plus rows completed in this invocation.

    """
    registry.ensure_dirs()
    rows: list[dict[str, Any]] = []

    baseline_specs = [s for s in specs if s.kind == "baseline"]
    encoder_specs = [s for s in specs if s.kind == "encoder"]

    if baseline_specs:
        append_lock = threading.Lock()

        def _run_baseline_spec(spec: RunSpec) -> dict[str, Any]:
            metrics_path = run_metrics_path(registry, spec)
            if metrics_path.exists():
                logger.info("Skipping completed run %s", spec.run_id)
                return _read_json(metrics_path)

            logger.info(
                "Running stage-06 %s/%s: %s",
                spec.phase,
                spec.kind,
                spec.run_id,
            )
            row = _run_one_spec(
                cfg,
                registry,
                spec,
                train_df,
                dev_df,
                validation_df,
            )
            _write_metrics(metrics_path, row)
            with append_lock:
                append_result_row(registry.learning_curve_results, row)
            return row

        with concurrent.futures.ThreadPoolExecutor(
            max_workers=2,
        ) as executor:
            futures = [
                executor.submit(_run_baseline_spec, spec) for spec in baseline_specs
            ]
            for future in concurrent.futures.as_completed(futures):
                rows.append(future.result())

    for spec in encoder_specs:
        metrics_path = run_metrics_path(registry, spec)
        if metrics_path.exists():
            logger.info("Skipping completed run %s", spec.run_id)
            rows.append(_read_json(metrics_path))
            continue

        logger.info(
            "Running stage-06 %s/%s: %s",
            spec.phase,
            spec.kind,
            spec.run_id,
        )
        row = _run_one_spec(cfg, registry, spec, train_df, dev_df, validation_df)
        _write_metrics(metrics_path, row)
        append_result_row(registry.learning_curve_results, row)
        rows.append(row)

    return rows


def run_metrics_path(registry: PathRegistry, spec: RunSpec) -> Path:
    """Return the resume sentinel path for a run spec.

    Args:
        registry: Path registry that owns ``runs_dir``.
        spec: Planned run.

    Returns:
        Path to the spec's ``metrics.json`` sentinel.

    """
    return registry.runs_dir / _storage_phase(spec) / spec.run_id / "metrics.json"


def append_result_row(path: Path, row: Mapping[str, Any]) -> None:
    """Append one completed run row to ``results.jsonl``.

    Args:
        path: JSONL result path.
        row: Result row conforming to the stage-06 schema.

    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(_jsonable(dict(row)), sort_keys=True) + "\n")


# ── Model selection report ───────────────────────────────────────────────────
#
# Aggregates per-cell metrics across seeds, computes mean ± SD and 95% CI,
# and applies the tie-to-simpler rule.  The recommended cell is written to
# ``selection_report.json`` and drives the final-seed refit.


def write_selection_report(
    rows: Sequence[Mapping[str, Any]],
    path: Path,
    *,
    cfg: BinaryClassifierConfig,
    registry: PathRegistry,
    eligible_cells: Iterable[tuple[str, str, str]] | None = None,
) -> dict[str, Any]:
    """Write the seed-aggregated selection report.

    Args:
        rows: Completed run rows.
        path: Destination ``selection_report.json``.
        cfg: Validated task configuration.
        registry: Path registry, used to resolve checkpoint paths for the printed
            selected-model skeleton.
        eligible_cells: Optional iterable of ``(model, targets, arm)`` cells
            eligible for recommendation. Defaults to all non-baseline rows.

    Returns:
        The report payload written to disk.

    """
    eligible = set(eligible_cells) if eligible_cells is not None else None
    cell_rows: dict[tuple[str, str, str], list[Mapping[str, Any]]] = {}
    seen: dict[tuple[str, str, str, int], Mapping[str, Any]] = {}
    for row in rows:
        model = str(row.get("model", ""))
        if model.startswith("baseline:"):
            continue
        key = (model, str(row.get("targets")), str(row.get("arm")))
        if eligible is not None and key not in eligible:
            continue
        seed = int(row["seed"])
        dedup_key = (*key, seed)
        if dedup_key in seen:
            continue
        seen[dedup_key] = row
        cell_rows.setdefault(key, []).append(row)

    if not cell_rows:
        report = _empty_selection_report(cfg)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, indent=2, sort_keys=True))
        return report

    summaries = [
        _summarize_cell(key, group) for key, group in sorted(cell_rows.items())
    ]
    recommendation, verdicts = _recommend_cell(summaries)
    skeleton = selected_model_skeleton(recommendation, cfg=cfg, registry=registry)
    report = {
        "schema_version": 1,
        "generated_at": datetime.now(UTC).isoformat(),
        "selection_metric": "validation.pr_auc",
        "tie_rule": (
            "An arm wins only if its seed-mean PR-AUC advantage exceeds the "
            "larger across-seed SD; otherwise ties go to the simpler arm."
        ),
        "summaries": summaries,
        "tie_rule_verdicts": verdicts,
        "recommendation": recommendation,
        "selected_model_skeleton": skeleton,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(report), indent=2, sort_keys=True))
    logger.info("Wrote model-selection report to %s", path)
    return report


def selected_model_skeleton(
    recommendation: Mapping[str, Any],
    *,
    cfg: BinaryClassifierConfig,
    registry: PathRegistry,
) -> dict[str, Any]:
    """Build the human-copied ``selected_model.json`` skeleton.

    Args:
        recommendation: Recommended selection cell.
        cfg: Validated task configuration.
        registry: Path registry with checkpoint directory locations.

    Returns:
        Dict with the §5.4 selected-model fields. If no ``model.safetensors`` is
        present yet, checkpoint fields contain explicit TODO placeholders.

    """
    encoder_id = str(recommendation["encoder_id"])
    arm = str(recommendation["arm"])
    seed = int(recommendation.get("representative_seed", cfg.training.final_seeds[0]))
    model_path = _find_model_safetensors(registry, encoder_id, arm, seed)
    if model_path is None:
        checkpoint_relpath = "TODO_RUN_FINAL_AND_FILL_CHECKPOINT_RELPATH"
        checkpoint_sha256 = "TODO_RUN_FINAL_AND_FILL_SHA256"
    else:
        checkpoint_relpath = str(model_path.relative_to(registry.models_dir))
        checkpoint_sha256 = sha256_file(model_path)
    return {
        "checkpoint_relpath": checkpoint_relpath,
        "checkpoint_sha256": checkpoint_sha256,
        "tokenizer_id": encoder_id,
        "encoder_id": encoder_id,
        "targets": str(recommendation["targets"]),
        "recipe_snapshot": {
            "arm": arm,
            "learning_rate": cfg.training.learning_rate,
            "batch_size": cfg.training.batch_size,
            "epochs": cfg.training.epochs,
            "weight_decay": cfg.training.weight_decay,
            "warmup_fraction": cfg.training.warmup_fraction,
            "early_stopping_patience": cfg.training.early_stopping_patience,
        },
        "selected_at": datetime.now(UTC).isoformat(),
    }


def sha256_file(path: Path) -> str:
    """Return the SHA-256 hex digest of a file.

    Args:
        path: File to hash.

    Returns:
        Hex digest string.

    """
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def print_selected_model_skeleton(report: Mapping[str, Any]) -> None:
    """Print the human-copied selected-model JSON skeleton.

    Args:
        report: Selection report containing ``selected_model_skeleton``.

    """
    skeleton = report.get("selected_model_skeleton")
    if skeleton is None:
        return
    print("\nCopy this JSON to selected_model.json after human review:")
    print(json.dumps(skeleton, indent=2, sort_keys=True))


def _baseline_specs(cfg: BinaryClassifierConfig) -> list[RunSpec]:
    specs = []
    for name in cfg.training.baselines:
        run_id = f"baseline-{name}-seed-{int(cfg.SEED)}"
        specs.append(
            RunSpec(
                kind="baseline",
                phase="baseline",
                run_id=run_id,
                seed=int(cfg.SEED),
                train_fraction=1.0,
                model=str(name),
                targets="hard",
                arm="default",
            ),
        )
    return specs


def _filter_encoders(
    encoders: Sequence[EncoderArm],
    encoder_id: str | None,
) -> list[EncoderArm]:
    if not encoders:
        raise ValueError("training.encoders must contain at least one encoder.")
    if encoder_id is None:
        return list(encoders)
    filtered = [arm for arm in encoders if arm.id == encoder_id]
    if not filtered:
        raise ValueError(f"No configured encoder has id {encoder_id!r}.")
    return filtered


def _filter_specs(specs: Sequence[RunSpec], encoder_id: str | None) -> list[RunSpec]:
    if encoder_id is None:
        return list(specs)
    return [
        spec for spec in specs if spec.kind == "baseline" or spec.model == encoder_id
    ]


def _primary_encoder(encoders: Sequence[EncoderArm]) -> EncoderArm:
    for arm in encoders:
        if arm.arm == "primary":
            return arm
    return encoders[0]


def _encoder_by_id(encoders: Sequence[EncoderArm], encoder_id: str) -> EncoderArm:
    for arm in encoders:
        if arm.id == encoder_id:
            return arm
    raise ValueError(f"Recommended encoder {encoder_id!r} is not configured.")


def _arm_cell(arm: str) -> tuple[str, str]:
    if arm == "hard":
        return "hard", "hard"
    return "soft", arm


def _encoder_spec(
    *,
    phase: RunPhase,
    encoder: EncoderArm,
    targets: str,
    arm: str,
    train_fraction: float,
    seed: int,
) -> RunSpec:
    run_id = "-".join(
        _run_id_parts(phase, encoder.id, targets, arm, train_fraction, seed),
    )
    return RunSpec(
        kind="encoder",
        phase=phase,
        run_id=run_id,
        seed=int(seed),
        train_fraction=float(train_fraction),
        model=encoder.id,
        targets=targets,
        arm=arm,
        encoder=encoder,
    )


def _run_one_spec(
    cfg: BinaryClassifierConfig,
    registry: PathRegistry,
    spec: RunSpec,
    train_df: pd.DataFrame,
    dev_df: pd.DataFrame,
    validation_df: pd.DataFrame,
) -> dict[str, Any]:
    if spec.kind == "baseline":
        row = _run_baseline(cfg, registry, spec, train_df, dev_df, validation_df)
    else:
        row = _run_encoder(cfg, registry, spec, train_df, dev_df, validation_df)
    row["run_id"] = spec.run_id
    row["model"] = "baseline:" + spec.model if spec.kind == "baseline" else spec.model
    row["targets"] = spec.targets
    row["arm"] = spec.arm
    row["train_fraction"] = float(spec.train_fraction)
    row["seed"] = int(spec.seed)
    return _jsonable(row)


def _run_baseline(
    cfg: BinaryClassifierConfig,
    registry: PathRegistry,
    spec: RunSpec,
    train_df: pd.DataFrame,
    dev_df: pd.DataFrame,
    validation_df: pd.DataFrame,
) -> dict[str, Any]:
    run_train = subset_fraction(train_df, spec.train_fraction, spec.seed)
    eval_dfs = {"dev": dev_df, "validation": validation_df}
    start = perf_counter()
    if spec.model == "tfidf_logreg":
        return tfidf_logreg(cfg, run_train, eval_dfs, seed=spec.seed)
    if spec.model == "minilm_logreg":
        return minilm_logreg(
            cfg,
            run_train,
            eval_dfs,
            seed=spec.seed,
            registry=registry,
            model_name=cfg.training.minilm_id,
            device=cfg.training.device,
            batch_size=cfg.training.batch_size,
        )
    raise ValueError(
        f"Unknown baseline {spec.model!r} after {perf_counter() - start:.3f}s",
    )


def _run_encoder(
    cfg: BinaryClassifierConfig,
    registry: PathRegistry,
    spec: RunSpec,
    train_df: pd.DataFrame,
    dev_df: pd.DataFrame,
    validation_df: pd.DataFrame,
) -> dict[str, Any]:
    if spec.encoder is None:
        raise ValueError("Encoder run spec missing encoder config.")
    run_train = subset_fraction(train_df, spec.train_fraction, spec.seed)
    oof_probs = (
        _oof_probs_for_pruned(run_train, registry, cfg)
        if spec.arm == "pruned"
        else None
    )
    if spec.arm in {"hard", "pruned", "class_weighted"}:
        run_train, loss_spec = run_arm(
            cast(ArmName, spec.arm),
            run_train,
            oof_probs=oof_probs,
        )
        targets = loss_spec.targets
        arm = loss_spec.arm
    else:
        targets = spec.targets
        arm = spec.arm

    with _validation_override(validation_df):
        return finetune(
            cfg,
            registry,
            spec.encoder,
            run_train,
            dev_df,
            targets=targets,
            arm=arm,
            train_fraction=spec.train_fraction,
            seed=spec.seed,
            run_root=registry.runs_dir / _storage_phase(spec),
        )


def _run_ids(rows: Iterable[Mapping[str, Any]]) -> set[str]:
    return {str(row.get("run_id")) for row in rows}


def _storage_phase(spec: RunSpec) -> str:
    """Return the run directory group used for resume sentinels."""
    return "sweep" if spec.phase == "final" else spec.phase


def _run_id_parts(
    phase: RunPhase,
    encoder_id: str,
    targets: str,
    arm: str,
    train_fraction: float,
    seed: int,
) -> list[str]:
    """Return run-id parts, keeping final ids identical to sweep ids."""
    parts = [
        _model_slug(encoder_id),
        targets,
        arm,
        f"f{str(float(train_fraction)).replace('.', 'p')}",
        f"s{int(seed)}",
    ]
    if phase == "curve":
        return ["curve", *parts]
    return parts


def _oof_probs_for_pruned(
    train_df: pd.DataFrame,
    registry: PathRegistry,
    cfg: BinaryClassifierConfig,
) -> pd.DataFrame:
    if registry.oof_pred_probs.exists():
        logger.info(
            "Loading true OOF probabilities from %s for pruned arm",
            registry.oof_pred_probs,
        )
        oof_df = pd.read_parquet(registry.oof_pred_probs)
        merged = (
            train_df[["EIN2"]]
            .astype(str)
            .merge(
                oof_df[["EIN2", "p1"]].astype(str),
                on="EIN2",
                how="left",
            )
        )
        p1 = pd.to_numeric(merged["p1"], errors="coerce")
        if p1.isna().any():
            missing = int(p1.isna().sum())
            logger.warning(
                "OOF probabilities missing for %d training rows; "
                "falling back to vote-share proxy",
                missing,
            )
        else:
            return pd.DataFrame(
                {
                    "EIN2": train_df["EIN2"].astype(str).tolist(),
                    "p0": 1.0 - p1.to_numpy(dtype=float),
                    "p1": p1.to_numpy(dtype=float),
                },
            )
    logger.warning(
        "No OOF probabilities at %s; using vote-share proxy for pruned arm",
        registry.oof_pred_probs,
    )
    return pd.DataFrame(
        {
            "EIN2": train_df["EIN2"].astype(str).tolist(),
            "p0": 1.0 - pd.to_numeric(train_df["p_pos"], errors="coerce"),
            "p1": pd.to_numeric(train_df["p_pos"], errors="coerce"),
        },
    )


@contextmanager
def _validation_override(validation_df: pd.DataFrame) -> Iterator[None]:
    """Temporarily route encoder validation loading to an in-memory frame."""
    old_loader: Any = encoder_mod.load_human_split

    def load_validation(cfg: Any, registry: Any, split: str) -> pd.DataFrame:
        del cfg, registry
        if split != "validation":
            raise ValueError("Stage 06 may only load the validation split.")
        return validation_df.copy()

    setattr(encoder_mod, "load_human_split", load_validation)
    try:
        yield
    finally:
        setattr(encoder_mod, "load_human_split", old_loader)


def _summarize_cell(
    key: tuple[str, str, str],
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    model, targets, arm = key
    pr_auc = [_metric(row, "pr_auc") for row in rows]
    f1 = [_metric(row, "f1") for row in rows]
    seeds = [int(row["seed"]) for row in rows]
    return {
        "encoder_id": model,
        "targets": targets,
        "arm": arm,
        "n_seeds": len(rows),
        "seeds": seeds,
        "validation_pr_auc": _mean_sd_ci(pr_auc),
        "validation_minority_f1": _mean_sd_ci(f1),
        "representative_seed": seeds[-1],
    }


def _metric(row: Mapping[str, Any], name: str) -> float:
    validation = row.get("validation")
    if not isinstance(validation, Mapping):
        raise ValueError(f"Run row {row.get('run_id')} missing validation metrics.")
    value = validation.get(name)
    if value is None:
        raise ValueError(f"Run row {row.get('run_id')} missing validation.{name}.")
    return float(value)


def _mean_sd_ci(values: Sequence[float]) -> dict[str, float]:
    arr = np.asarray(values, dtype=float)
    mean = float(np.mean(arr))
    sd = float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0
    half_width = 1.96 * sd / float(np.sqrt(len(arr))) if len(arr) > 1 else 0.0
    return {
        "mean": mean,
        "sd": sd,
        "ci_lower": mean - half_width,
        "ci_upper": mean + half_width,
    }


def _recommend_cell(
    summaries: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    ordered = sorted(
        summaries,
        key=lambda item: (
            -float(_nested(item, "validation_pr_auc", "mean")),
            _complexity_rank(item),
            str(item["encoder_id"]),
            str(item["targets"]),
            str(item["arm"]),
        ),
    )
    incumbent = ordered[0]
    verdicts: list[dict[str, Any]] = []
    for challenger in ordered[1:]:
        incumbent, verdict = _tie_rule_pair(incumbent, challenger)
        verdicts.append(verdict)
    recommendation = {
        "encoder_id": str(incumbent["encoder_id"]),
        "targets": str(incumbent["targets"]),
        "arm": str(incumbent["arm"]),
        "validation_pr_auc_mean": float(
            _nested(incumbent, "validation_pr_auc", "mean"),
        ),
        "validation_minority_f1_mean": float(
            _nested(incumbent, "validation_minority_f1", "mean"),
        ),
        "representative_seed": int(incumbent["representative_seed"]),
    }
    return recommendation, verdicts


def _tie_rule_pair(
    incumbent: Mapping[str, Any],
    challenger: Mapping[str, Any],
) -> tuple[Mapping[str, Any], dict[str, Any]]:
    inc_mean = float(_nested(incumbent, "validation_pr_auc", "mean"))
    ch_mean = float(_nested(challenger, "validation_pr_auc", "mean"))
    inc_sd = float(_nested(incumbent, "validation_pr_auc", "sd"))
    ch_sd = float(_nested(challenger, "validation_pr_auc", "sd"))
    margin = ch_mean - inc_mean
    incumbent_margin = inc_mean - ch_mean
    threshold = max(inc_sd, ch_sd)
    clear_win = margin > threshold
    clear_incumbent = incumbent_margin > threshold
    simpler = _simpler_cell(incumbent, challenger)
    if clear_win:
        winner = challenger
        reason = "clear_pr_auc_win"
    elif clear_incumbent:
        winner = incumbent
        reason = "clear_pr_auc_win"
    else:
        winner = simpler
        reason = "tie_to_simpler"
    return winner, {
        "incumbent": _cell_label(incumbent),
        "challenger": _cell_label(challenger),
        "margin": float(margin),
        "sd_threshold": float(threshold),
        "clear_win": bool(clear_win),
        "winner": _cell_label(winner),
        "reason": reason,
    }


def _simpler_cell(
    left: Mapping[str, Any],
    right: Mapping[str, Any],
) -> Mapping[str, Any]:
    left_rank = _complexity_rank(left)
    right_rank = _complexity_rank(right)
    if right_rank < left_rank:
        return right
    return left


def _complexity_rank(item: Mapping[str, Any]) -> int:
    arm_rank = {"default": 0, "hard": 1, "pruned": 2, "class_weighted": 3}
    target_rank = {"soft": 0, "hard": 1}
    encoder_id = str(item["encoder_id"])
    encoder_rank = 0 if "deberta" in encoder_id.lower() else 1
    return (
        encoder_rank * 100
        + target_rank.get(str(item["targets"]), 9) * 10
        + arm_rank.get(str(item["arm"]), 9)
    )


def _cell_label(item: Mapping[str, Any]) -> dict[str, str]:
    return {
        "encoder_id": str(item["encoder_id"]),
        "targets": str(item["targets"]),
        "arm": str(item["arm"]),
    }


def _nested(item: Mapping[str, Any], key: str, subkey: str) -> Any:
    nested = item[key]
    if not isinstance(nested, Mapping):
        raise ValueError(f"Expected mapping at {key!r}.")
    return nested[subkey]


def _find_model_safetensors(
    registry: PathRegistry,
    encoder_id: str,
    arm: str,
    seed: int,
) -> Path | None:
    root = registry.checkpoints_dir / _model_slug(encoder_id) / arm / f"s{seed}"
    if not root.exists():
        return None
    candidates = sorted(root.rglob("model.safetensors"))
    return candidates[-1] if candidates else None


def _empty_selection_report(cfg: BinaryClassifierConfig) -> dict[str, Any]:
    del cfg
    return {
        "schema_version": 1,
        "generated_at": datetime.now(UTC).isoformat(),
        "selection_metric": "validation.pr_auc",
        "tie_rule": "No eligible encoder rows were available.",
        "summaries": [],
        "tie_rule_verdicts": [],
        "recommendation": None,
        "selected_model_skeleton": None,
    }


def _write_metrics(path: Path, row: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(dict(row)), indent=2, sort_keys=True))


def _read_json(path: Path) -> dict[str, Any]:
    raw = json.loads(path.read_text())
    if not isinstance(raw, dict):
        raise ValueError(f"Expected JSON object at {path}.")
    return raw


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def _validated_fraction(value: float) -> float:
    fraction = float(value)
    if not 0 < fraction <= 1:
        raise ValueError("Training subset fractions must be in (0, 1].")
    return fraction


def _model_slug(model_name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "__", model_name).strip("_")
