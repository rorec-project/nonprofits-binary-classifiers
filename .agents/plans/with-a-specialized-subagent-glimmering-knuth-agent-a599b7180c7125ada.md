# Literature-Currency Audit — Modeling / Fine-Tuning / Calibration / Evaluation

**Date:** 2026-06-10 · **Mode:** read-only · **Scope:** future stages 05–08 (training, eval, calibration, inference)
**Repo:** `/home/dubidub/Documents/Projects/nonprofits-binary-classifiers`

This audit independently checks the planned modeling/fine-tuning/calibration/evaluation methods
against 2024–2026 external literature (web sources cited inline), not just the internal dated
`.agents/docs/*` handoffs. **Bottom line: the plan is overwhelmingly ALIGNED with 2026 practice.**
There is one concrete config DRIFT (fp16→bf16 on Blackwell), two GAP-minor optional additions
(soft/confidence-weighted labels; decision-curve analysis), and two genuinely UNRESOLVED
choices to put to the human (default encoder; selective-prediction vs rule-cascade framing).

---

## (a) Planned design (read, with file:line)

**The regime — these constraints decide every verdict below:**
- ~20,000 frozen weak/silver labels (`SampleSizesConfig.silver = 20_000`, config.py:166) — **not** a
  few-shot regime; over-provisioned so a learning-curve sweep finds optimal N.
- Rare positive class: corpus religious base rate ≈13% (`we-work-on-the-floofy-wreath.md:34`).
- Short text: median 22 words per `LONGEST_MISSION` (`we-work-on-the-floofy-wreath.md:32`).
- Downstream estimand is **population prevalence** via calibrated scores (QuaPy CC/ACC/PACC, Saerens
  EM/SLD, ReadMe, PPI) — `with-a-specialized-subagent-glimmering-knuth.md:69-73`. This *requires a
  calibrated scalar probability per record*, which constrains the calibration choice.
- Full-corpus inference at ~560k records (`we-work-on-the-floofy-wreath.md:32`) — cost/throughput +
  reproducibility/archivability matter.

**Training stubs** — `TrainingConfig`, config.py:219–235: `learning_rate=5e-5` (:227),
`batch_size=16` (:228), `epochs=10` (:229), `weight_decay=0.01` (:230),
`metric_for_best_model="pr_auc"` (:231), `greater_is_better=True` (:232),
`early_stopping_patience=4` (:233), `save_total_limit=2` (:234), `fp16=True` (:235). The docstring
(config.py:220–225) explicitly states these are **placeholders**; real hyperparameters come from the
learning-curve sweep + best-model selection on the human validation set.

**Model grid (stage 05 stub)** — `we-work-on-the-floofy-wreath.md:110`: TF-IDF+LR and **MiniLM+LR**
baselines; encoders **DeBERTa-v3-base / RoBERTa-base / DistilBERT**; **small open-weight LLM (LoRA)**
as a comparison arm. ModernBERT-base listed as an *optional* encoder
(`20260606-tech-short-text-model-alternatives.md:195`). PR-AUC is the declared selection metric (a
deliberate deviation from the roadmap's best-val-F1). Learning-curve sweep
{0.5k,1k,2k,4k,8k,16k} → plateau = optimal N. Weighted + unweighted CE compared.

**Aggregation / denoising** — majority vote is default; **crowd-kit (Dawid–Skene)** and
**cleanlab 2.9 (confident learning / CROWDLAB)** are kept as drop-in comparison arms
(`we-work-on-the-floofy-wreath.md:19,103`), currently quarantined behind `NotImplementedError`
(`with-a-specialized-subagent-glimmering-knuth.md:163-171`). Internal stance: adopt a noisy-label
method *only if it beats LLM-only on the human held-out set*
(`20260606-tech-llm-weak-supervision-noisy-labels.md`).

**Evaluation (stage 06 stub)** — `we-work-on-the-floofy-wreath.md:111`: PR-AUC + precision/recall/F1
+ MCC + balanced-acc + calibration (**Brier / ECE / reliability**) + subgroup error by NTEE group /
length, all with **bootstrap CIs** (the ~200-row frozen test is small-N); acceptance criteria set
before unlocking test.

**Calibration** — `20260606-tech-calibration-quantification-prevalence.md` + `…knuth.md:69-73`:
held-out Platt/logistic or temperature scaling first; isotonic only if the calibration set is large
enough; netcal / scikit-learn `CalibratedClassifierCV`; subgroup/multicalibration under shift;
prevalence = mean of calibrated probabilities.

**Rule-layer + classifier hybrid at inference** — `apply_rule_label`, quality.py:602–633: strong-
tradition lexicon hit → `1`; very short (<6 words) + no religious lexicon → `0`; else `None` →
routed to the classifier. Bare-label churches (~9% of positives, in the LOW tier) are handled by
this high-precision rule layer rather than dropped (`we-work-on-the-floofy-wreath.md:23,112`). The
classifier's validated scope is HIGH+MEDIUM; LOW goes to the rule layer.

---

## (b) Per-method verdict table (2025–2026 sources)

| Method (as planned) | Verdict | Finding & 2025–2026 source |
|---|---|---|
| **Base model = fine-tuned encoder (DeBERTa-v3 / RoBERTa / DistilBERT) on ~20k weak labels** | **ALIGNED** | At ~20k labels this is *not* the few-shot regime; supervised encoder fine-tuning remains the 2026 default for pattern-driven binary text classification. A 2025 controlled study trained ModernBERT and a DeBERTaV3-based model on **identical** data and found **DeBERTaV3 wins on accuracy and sample efficiency**; ModernBERT wins on speed but is "particularly sensitive to learning rate choices" / had convergence failures. DeBERTa-v3-base as the default encoder is current. arXiv:2504.08716 (https://arxiv.org/html/2504.08716v1); HF ModernBERT blog (https://huggingface.co/blog/modernbert) |
| **SetFit / contrastive few-shot as an alternative** | **N/A — correctly excluded** | SetFit's design regime is ~8–64 examples per class (HF SetFit, https://huggingface.co/blog/setfit; arXiv:2209.11055). At 20k labels it is dominated by plain fine-tuning. The plan does not propose it; that is the right call. Do **not** add it. |
| **Embedding + linear head (MiniLM + LogReg)** | **ALIGNED** | Plan already includes `all-MiniLM-L6-v2` + LogReg as a cheap, reproducible, auditable baseline (`20260606-tech-short-text-model-alternatives.md:185`). Still endorsed 2026 as a baseline (not the production model at 560k scale). |
| **LoRA/PEFT on a small decoder** | **ALIGNED (as comparison arm)** | Plan has a LoRA `SEQ_CLS` arm explicitly as a *comparison*, run only after encoder baselines (`…short-text-model-alternatives.md:108-132`). 2026 evidence keeps decoder-LoRA a comparison arm, not the production classifier at scale (cost/reproducibility). |
| **Distilling the LLM annotator into a small student** | **ALIGNED (this *is* the plan)** | "LLM labels → fine-tune a small supervised classifier" *is* knowledge distillation from an LLM annotator; students on LLM labels match human-label students (Pangakis & Wolken 2024, already cited; PGKD EMNLP-2024 https://aclanthology.org/2024.emnlp-main.215/; arXiv:2406.17633). The plan is squarely inside this paradigm. |
| **Training on noisy labels: majority-vote + weighted/unweighted CE** | **GAP-minor / optional** | Majority-vote + class-weighted CE is a reasonable floor. The plan's cleanlab/confident-learning arm is the 2026-current *data-cleaning* layer (cleanlab docs https://docs.cleanlab.ai/stable/cleanlab/classification.html; confident learning https://l7.curtisnorthcutt.com/confident-learning). One value left on the table: **use the per-example silver agreement as a soft label / loss weight** (confidence-weighted training). Of the named noise-robust techniques: **label smoothing** is the cheap default-add (one flag, mild regularization against over-confident wrong labels); **co-teaching / noise-robust losses (GCE, symmetric CE)** are heavier machinery likely overkill given clean-ish LLM-majority labels plus the cleanlab cleaning arm already planned. All fall under the internal "adopt only if it beats LLM-only on human held-out" gate, which is sound — keep it. |
| **Aggregation: crowd-kit (D-S) + cleanlab arms** | **ALIGNED** | Confident learning (cleanlab 2.9) is the SOTA noisy-label *audit/cleaning* tool in 2026; Dawid–Skene for per-source reliability. Both already first-class comparison arms; both Py3.13-ready. |
| **Learning-curve sweep to choose training N** | **ALIGNED** | Empirical data-size curve (train at 50…all labels, report mean±sd, find plateau) is still the standard 2026 recommendation (`…short-text-model-alternatives.md:198`). No displacing standard found; **active learning** is the adjacent alternative *if label budget were the binding constraint* — but here the silver pool is ≈free, so a plateau sweep is the right tool. |
| **PR-AUC selection metric (over best-val-F1)** | **ALIGNED** | For a rare-positive class, PR-AUC + MCC are the recommended primaries; ROC-AUC alone is criticized as insufficient for high imbalance (MDPI Technologies 2026, https://www.mdpi.com/2227-7080/14/1/54). The plan's PR-AUC-primary deviation is *more* current than the roadmap's F1. |
| **Calibration: Platt / temperature first; isotonic if set large enough** | **ALIGNED** | Because the estimand is *population prevalence = mean of calibrated p*, the plan needs a **calibrated scalar probability**, which Platt/temperature/isotonic provide and conformal does **not**. Held-out temperature/Platt first, isotonic only with enough data, subgroup/multicalibration under shift — still the 2026 recommendation (Silva Filho 2023 survey; netcal). Conformal prediction gives prediction-*set* coverage, not scalar calibration — it is an *add-on for abstention*, not a replacement (arXiv:2512.17048; https://valeman.substack.com/p/the-fallacy-of-predict_proba). **Focal-calibration** is a *training-time* lever (focal loss can lower ECE under imbalance) that **complements, not replaces** the held-out post-hoc step — it is not a substitute for Platt/temperature (Mukhoti 2020, https://papers.neurips.cc/paper/2020/hash/aeb7b30ef1d024a76f21a1d40e30c302-Abstract.html). Venn–Abers is a current alternative worth a diagnostic look if a proper-scoring guarantee is wanted. |
| **Evaluation bundle: P/R/F1 + MCC + balanced-acc + PR-AUC + Brier/ECE/reliability + bootstrap CIs + per-stratum** | **ALIGNED (+1 GAP-minor)** | This is the current imbalanced-eval bundle (MCC + PR-AUC over ROC-AUC; calibration reported alongside discrimination). Subgroup/slice error by NTEE group/length already present. **GAP-minor: decision-curve analysis (net benefit vs threshold)** is the 2024–2026 addition for a *deployed* classifier where a threshold/workload decision is made — directly informs the inference cut-point. arXiv:2509.24608; arXiv:2504.04528 (consequentialist critique of binary eval). |
| **Inference hybrid: high-precision rule cascade + classifier** | **ALIGNED (framing UNRESOLVED)** | The rule layer covers bare-label churches the classifier is **not trained on** (LOW tier excluded from train/serve scope) — a defensible coverage device, not a crutch. The modern alternative *framing* is **selective prediction / abstention** within one calibrated model. These are reconcilable: rule layer = deterministic high-precision shortcut on inputs outside the trained distribution; selective prediction = abstain on low-confidence in-distribution cases. Worth deciding explicitly (see UNRESOLVED). |
| **`fp16=True` training dtype (config.py:235)** | **DRIFTED (concrete)** | On the target **NVIDIA Blackwell B200** node, **bf16 is the recommended 16-bit training dtype** over fp16 (same FP32 dynamic range, better numerical stability, native on Ampere+); fp16 is the legacy choice prone to gradient overflow. Set `bf16=True` (or expose an `fp16`/`bf16` switch and default bf16 on B200). FP8/MXFP8 is a *further* B200-only speedup to consider later, not a default. Megatron/NeMo mixed-precision docs (https://docs.nvidia.com/nemo/megatron-bridge/latest/training/mixed-precision.html); https://acecloud.ai/blog/fp8-vs-bf16-mixed-precision-tensor-cores/ |
| **`learning_rate=5e-5` stub (config.py:227)** | **note only (placeholder)** | Not a committed value (set by the sweep). Flagged only because 5e-5 is high for DeBERTa-v3 — the internal tech doc and HF docs use **2e-5** as the DeBERTa default and add 5e-6/8e-6 if unstable (`…short-text-model-alternatives.md:46,196`). The sweep grid `{1e-5,2e-5,3e-5,5e-5}` already covers this. No action beyond keeping the sweep. |

---

## (c) Ranked concrete recommendations

1. **[Config DRIFT — do this] Default to bf16, not fp16, for training on B200.** Change the intent of
   `TrainingConfig.fp16` (config.py:235): either rename/extend to a precision switch defaulting to
   **bf16** on Blackwell, or document that the trainer must set `bf16=True` on the B200 node. Low
   effort, removes a real numerical-stability/perf footgun. (Optionally note FP8/MXFP8 as a future
   B200-only throughput arm.)

2. **[Base-model default — confirm] Keep DeBERTa-v3-base as the default encoder; treat ModernBERT-base
   as a speed/throughput arm, not the default.** The only controlled head-to-head (arXiv:2504.08716)
   has DeBERTa-v3 ahead on accuracy *and* sample efficiency, with ModernBERT more LR-brittle —
   important at 20k labels with a rare positive. Short 22-word missions also negate ModernBERT's main
   selling point (long-context). This matches the current grid; no change needed, but see UNRESOLVED #1.

3. **[Eval GAP-minor — add] Add decision-curve analysis (net benefit vs threshold) to the stage-06
   bundle.** With a deployed threshold + 560k-record workload, net-benefit/clinical-impact-style curves
   are the 2024–2026 addition that turns the metric bundle into a *deployment* decision tool and pairs
   naturally with the calibration the plan already does. Everything else in the bundle is current.

4. **[Noisy-label GAP-minor — optional arm] Add a confidence-weighted / soft-label training arm using
   the per-example silver agreement.** The model×prompt agreement already computed for the silver label
   is a free confidence signal; use it as a soft label or per-example loss weight (alongside the
   existing cleanlab cleaning arm). Keep the plan's gate: adopt only if it beats LLM-majority on the
   human held-out set.

5. **[No-ops — explicitly do NOT add] SetFit/contrastive few-shot** (wrong regime at 20k);
   **conformal prediction as a calibration replacement** (gives set coverage, not the scalar calibrated
   probability prevalence needs — at most an abstention add-on); **decoder-LoRA or LLM-distillation as
   the *primary* classifier** (cost/reproducibility at 560k — already correctly comparison arms). The
   plan's restraint here is correct; recording it prevents future churn.

---

## UNRESOLVED — to ask the human

1. **Encoder default vs throughput at 560k inference.** External evidence says DeBERTa-v3 = accuracy,
   ModernBERT = ~2–4× speed but more brittle. At 560k records, is inference throughput a hard
   constraint that would justify ModernBERT-base as the *production* encoder despite the accuracy/
   sample-efficiency edge of DeBERTa-v3? (Mixed sources → human call, not a unilateral drift.)

2. **Rule-cascade vs selective prediction framing for the inference path.** Keep the deterministic
   high-precision rule layer for out-of-trained-scope bare-label inputs *and/or* add per-record
   abstention (selective prediction) on low-confidence in-distribution cases? They can coexist;
   confirm the intended division of labor and whether abstained records get human review or a default.

3. **Soft/confidence-weighted labels (rec #4):** is it worth the added arm now, or deferred until the
   majority-vote baseline is measured on the human test set?

4. **Calibration data budget:** with a ~200-row frozen test and small positive count, isotonic and
   subgroup/multicalibration are unstable. Confirm whether the gold/calibration split is large enough
   for per-stratum calibration, or whether to restrict to global temperature/Platt + report subgroup
   reliability as diagnostic only. (The internal calibration doc already warns on this.)

---

## Sources

- ModernBERT vs DeBERTaV3 (controlled, identical-data study) — https://arxiv.org/html/2504.08716v1
- ModernBERT model/blog — https://huggingface.co/blog/modernbert ; arXiv:2412.13663
- SetFit (few-shot regime) — https://huggingface.co/blog/setfit ; https://arxiv.org/abs/2209.11055
- Knowledge distillation from LLM labels — https://aclanthology.org/2024.emnlp-main.215/ ; https://arxiv.org/pdf/2406.17633
- Confident learning / cleanlab — https://docs.cleanlab.ai/stable/cleanlab/classification.html ; https://l7.curtisnorthcutt.com/confident-learning
- Conformal vs Platt/isotonic; scalar-calibration caveat — https://arxiv.org/html/2512.17048v1 ; https://valeman.substack.com/p/the-fallacy-of-predict_proba
- Imbalanced eval (MCC/PR-AUC over ROC-AUC) — https://www.mdpi.com/2227-7080/14/1/54
- Decision-curve analysis / net benefit — https://arxiv.org/html/2509.24608 ; https://arxiv.org/pdf/2504.04528
- bf16 vs fp16 / FP8 on Blackwell B200 — https://docs.nvidia.com/nemo/megatron-bridge/latest/training/mixed-precision.html ; https://acecloud.ai/blog/fp8-vs-bf16-mixed-precision-tensor-cores/
</content>
</invoke>
