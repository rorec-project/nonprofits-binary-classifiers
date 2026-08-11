# Non-Technical Overview of the Religious-Mission Classifier

This project estimates how many nonprofit organizations appear to have a **religious mission** based on their mission text.

## What the classifier does

- It reads a nonprofit's mission text.
- It predicts whether that text looks **religious** or **non-religious**.
- It produces both a probability score and one or more yes/no labels.

The classifier is strongest as a **screening and measurement tool**. It is not a legal determination, and it should not be treated as proof of an organization's identity or status.

## What the current results say

- On the archived frozen test, the model is very good at **finding positives**: recall is `0.987012987012987` and F1 is `0.8941176470588236`.
- The best estimate of the share of organizations with religious mission text is about **14.4%** (95% CI 12.7–16.1%), computed over the full ~560,000-organization corpus.
- That `14.4%` is slightly below the earlier `14.45%` headline because the LOW-quality-text slice is now handled more carefully.

## What “prevalence” means here

“Prevalence” means the estimated share of **all organizations in the population**, not just the share inside a hand-labeled sample.

That matters because the hand-labeled test set was intentionally enriched with harder and more positive-looking cases. Its precision and F1 are useful for judging model behavior, but they are **not** the same thing as the population share.

The classifier's raw output by NTEE group (industry category) is *not* the same thing as prevalence either, and the two can disagree substantially in the same group. See [Why the Religious Share for Arts Organizations Is 5.33% in One Figure and 0.59% in Another](memos/classified-share-vs-corrected-prevalence.md) for a walk-through of why, using real numbers.

## Why there are three yes/no labels now

The released dataset includes three binary labels because different users need different operating points:

| Label | Plain meaning |
|---|---|
| `pred_label` | Recall-first label used for the prevalence estimate |
| `pred_label_maxf1` | A more balanced label that maximizes F1 |
| `pred_label_baserate` | A stricter label meant to target about 90% precision at the expected population base rate |

The current base-rate label uses threshold `0.09368807964553742`, and the target is attainable on current anchor data.

## How to read the figures

- **Reliability diagram**: whether predicted probabilities are well calibrated.
- **Score distribution**: where organizations cluster by score, quality tier, and label.
- **Prevalence decomposition**: how the final prevalence estimate combines higher-quality rows and LOW-quality rows.
- **Rule-validation intervals**: how certain we are about the deterministic LOW-tier rule layer.
- **Subgroup performance**: whether performance looks different across NTEE groups or text-length bands.
- **Mean score by NTEE major group**: the classifier's average calibrated and raw score for each of the 26 NTEE groups, covering only the classifier-scored subset of organizations (rows a rule router did not already decide).
- **Classified share by NTEE major group**: the raw share of each NTEE group's organizations the classifier labeled religious, at three operating thresholds — raw classifier output, not a prevalence estimate.
- **Count classified religious by NTEE major group**: the raw number of organizations the classifier labeled religious in each NTEE group, shown against each group's size.
- **Classified share against corrected prevalence estimate**: the raw classified share and the corrected prevalence estimate plotted together per NTEE group, so the gap between them — over-calling in low-prevalence groups, under-calling in high-prevalence ones — is visible directly. See the memo linked above for why.

Two frozen-test visual sections are still intentionally blank in the current real report:

- the full PR/ROC curves
- the confusion matrices at all three release thresholds

Those are finalized only after the controlled post-sprint UCloud re-evaluation.

## Important caveats

- Identical mission text is treated as the same mission and gets the same prediction when the dataset is expanded back to raw organizations.
- Very short or boilerplate LOW-quality text is handled partly by deterministic rules, not only by the classifier.
- The current local documentation refresh does **not** reopen the real frozen test.
- The canonical post-sprint run will regenerate the final released dataset and fill in the pending frozen-test figure sections.

## Where to go next

- Technical refresh report: [audits/20260702-local-evaluation-refresh.md](audits/20260702-local-evaluation-refresh.md)
- Released dataset dictionary: [predictions-full-data-dictionary.md](predictions-full-data-dictionary.md)
