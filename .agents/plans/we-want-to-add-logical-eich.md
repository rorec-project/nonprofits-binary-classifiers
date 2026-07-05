# Sprint: Stage-10 visualization enrichment & hardening

## Context

Stage 10 (`scripts/10_visualize.py`) renders ~15 publication-styled figures through one
`_save_plot` choke point that saves each as PDF+SVG+PNG with `transparent=True`. Style is
centralized in `viz/style.py` (Okabe–Ito palette, paper rcParams). Text-diagnostic figures
are corpus-count based over `mission_text` conditioned on `silver_label` (1 = religious,
0 = non-religious), joined by `EIN2` — the deployed DeBERTa model exposes no interpretable
coefficients.

Four needs drive this sprint:

1. **N-gram diagnostics are thin.** The one text figure `ngram_log_odds`
   (`viz/ngrams.py`) uses *naive* additive log-odds (rare words dominate), covers only 1–2
   grams, and hardcodes off-palette `tab:` colors.
2. **No wordclouds** for class-0 vs class-1 across uni/bi/tri-grams.
3. **Transparency uncertainty** — user suspects some figures render white, not transparent.
4. **Overflow/clipping** in existing figures — e.g. `subgroup_performance` has markers not
   visible at the right margin. Needs a generalized fix, not a one-off.

**Sprint goal:** ship a richer, publication-quality set of text-diagnostic figures and
eliminate edge-clipping across all stage-10 figures, with transparency verified.

## Decisions (confirmed with user)

- **Log-odds:** *add* Monroe et al. weighted log-odds as new variants; keep the existing
  naive `ngram_log_odds.*` (may be cited in `paper/paper.md`). Fix its palette regardless.
- **Wordclouds:** produce *both* raw-frequency and class-distinctiveness variants.
- **Stopwords:** no blanket removal; remove only generic tokens with no class signal.

## Method notes (research-grounded)

- **Weighted log-odds — Monroe, Colaresi & Quinn (2008), "Fightin' Words."** Log-odds-ratio
  with an **informative Dirichlet prior** (background = pooled corpus counts, scaled) divided
  by its posterior SE → a **z-score**. Standard fix for the naive method's rare-word
  dominance. ~30 lines of numpy, no new dependency. The signed z-score doubles as the per-term
  per-class distinctiveness weight reused by the wordclouds.
- **Wordcloud transparency:** `WordCloud(mode="RGBA", background_color=None, ...)` has a known
  white-background bug (word_cloud #569). **Verify rendered alpha; fall back to
  `background_color="rgba(255,255,255,0)"` if opaque.** Output is raster (embedded via
  `imshow`) — these figures stay raster inside PDF/SVG, unlike the vector line/bar figures.
- **Font match:** `font_path=matplotlib.font_manager.findfont("DejaVu Sans")` so clouds match
  the paper sans-serif.
- **Stopwords:** curated English list (avoid sklearn's flawed `ENGLISH_STOP_WORDS`) + small
  nonprofit-boilerplate set (`organization`, `provide`, `services`, `inc`, `foundation`,
  `mission`…). Apply to **frequency wordclouds only**; keep all tokens for log-odds bars and
  distinctiveness clouds. Whitelist nothing carrying religious/secular signal.
- **Edge clipping:** hard axis limits (`set_xlim(0,1)`) + default `clip_on=True` clip markers
  whose centers sit on the boundary. Generalized fix = small symmetric data margins (e.g.
  `set_xlim(-0.02, 1.02)`) and/or `clip_on=False` for edge markers; keep legends off the data.

---

## Sprint tickets

### S1 — Figure-hygiene sweep: eliminate edge clipping *(do first; low-risk, high-value)*
- **`viz/curves.py::subgroup_performance`**: replace `ax.set_xlim(0.0, 1.0)` with padded
  limits (`-0.02, 1.02`) and set `clip_on=False` on the scatter calls; move legend so it
  never covers edge dots (e.g. `bbox_to_anchor` outside, or `loc="best"` after margins).
- **Generalize:** audit every drawer in `viz/` for hard `set_xlim`/`set_ylim` with data at the
  boundary and for markers/text at edges. Add a shared helper in `viz/style.py`
  (e.g. `pad_axes(ax, x=0.02, y=0.02)` or a `clip_on=False` convention) and apply it wherever
  data can hit a limit. Verify multi-panel drawers (`score_distribution_by_tier_label`,
  `threshold_sweep_plot`, `frozen_test_confusion_matrices`) too.
- **Acceptance:** re-render all figures; no marker/label is cut by a spine or the figure edge.
- **Files:** `viz/curves.py`, `viz/prevalence_plots.py`, `viz/bakeoff.py`, `viz/style.py`.

### S2 — Transparency audit *(verify-first, minimal code)*
- `grep -rn "set_facecolor\|facecolor=" src/binary_classifier/viz/` — confirm no drawer
  overrides the global `transparent=True`; fix any that do.
- PIL alpha check on an existing PNG: mode contains `"A"` and `min(alpha) < 255`.
- If files are genuinely transparent (expected), the "white" is a viewer/pandoc compositing
  artifact — **document that**; do not change already-correct save settings.
- **Acceptance:** written finding (code fix or documented caveat) + all PNGs confirmed
  transparent.

### S3 — N-gram log-odds: weighted method + palette fix
- **`viz/ngrams.py`:** extract reusable core
  `compute_ngram_scores(texts, labels, *, ngram_range, method, min_df, stopwords=None)` →
  `(terms, signed_scores, pos_counts, neg_counts)`, `method ∈ {"naive","weighted"}`.
  - `"naive"` = current formula (keeps existing figure intent stable).
  - `"weighted"` = Monroe et al. z-scored weighted log-odds.
- Keep `ngram_log_odds(df, ax, *, top_k=30)` (naive) but swap `tab:blue`/`tab:orange`/`black`
  → `OKABE_ITO_BLUE`/`OKABE_ITO_VERMILLION`/`OKABE_ITO_BLACK`.
- Add `ngram_weighted_log_odds(df, ax, *, ngram_range, top_k=30)` — diverging bars of top-k
  |z-score|, Okabe–Ito palette, order named in title/labels.
- Reuse `_detect_column`, `_validate_binary_labels`.
- **Acceptance:** naive figure unchanged in content (new colors); weighted variant renders for
  each n-gram order.

### S4 — Wordclouds: new module + both weightings
- **`viz/wordclouds.py` (new):**
  `class_wordclouds(df, ax, *, ngram_range, weighting, max_words=150, stopwords=None)`,
  `weighting ∈ {"frequency","distinctive"}`.
  - Two-panel (religious | non-religious) via the `ax.remove()` + `fig.subplots(1,2)` pattern
    (mirror `curves.py:146`), since `_save_plot` hands a single `ax`.
  - Per-class color ramps keyed to Okabe–Ito (religious→blue, non-religious→vermillion) via a
    `color_func`.
  - `"frequency"` → per-class counts (stopwords applied); `"distinctive"` → positive-side
    weighted z-scores from `compute_ngram_scores` (no stopword removal).
  - Transparent RGBA per method note; assert alpha after render.
- **`pyproject.toml`:** add `wordcloud` (numpy 2.x / py3.13 compatible); `uv sync`.
- **Acceptance:** both cloud variants render per class per n-gram order with transparent bg.

### S5 — Wire into stage 10 + exports
- **`scripts/10_visualize.py`:** add `_maybe_render_*` steps (reuse `_silver_with_text`,
  `_save_plot`) and register in the `run_visualization` tuple (~lines 87–103):
  - `ngram_weighted_log_odds_{unigram,bigram,trigram}` — ranges (1,1)/(2,2)/(3,3).
  - `wordcloud_frequency_{unigram,bigram,trigram}` — 2-panel.
  - `wordcloud_distinctive_{unigram,bigram,trigram}` — 2-panel.
- **`viz/__init__.py`:** export new public drawers in `__all__`.
- **Acceptance:** stage-10 run renders all new figures; missing inputs degrade gracefully.

### S6 — Verification & lint
- Mirror existing `tests/` patterns for the new/changed drawers (at least import + smoke
  render on a tiny synthetic frame with both classes).
- `ruff check` clean; existing viz tests pass.

## Files touched
- `viz/ngrams.py` (refactor + weighted + palette), `viz/wordclouds.py` (new),
  `viz/curves.py` / `viz/prevalence_plots.py` / `viz/bakeoff.py` (hygiene),
  `viz/style.py` (padding helper), `viz/__init__.py` (exports),
  `scripts/10_visualize.py` (render steps), `pyproject.toml` / `uv.lock`.

## End-to-end verification
1. `uv sync`; `python -c "import wordcloud"` works.
2. `python scripts/10_visualize.py --config config/religious_missions.yaml` runs clean; log
   shows all new figures rendered, none silently skipped (confirm `silver_labels.csv` +
   `missions_cross_section.parquet` present so the join yields both classes).
3. New files under `data/processed/figures/`:
   `ngram_weighted_log_odds_{unigram,bigram,trigram}.*`,
   `wordcloud_{frequency,distinctive}_{unigram,bigram,trigram}.*`.
4. **Clipping:** visually confirm `subgroup_performance` and all figures show every
   marker/label inside the spines.
5. **Transparency:** every new PNG has `min(alpha) < 255`.
6. **Sanity:** distinctiveness clouds/bars show religious (ministry, gospel, church, faith…)
   vs secular (research, community, arts…); frequency clouds differ from distinctiveness.
7. `ngram_log_odds.*` unchanged content, now Okabe–Ito colors. `ruff check` + tests pass.
