# Why the Religious Share for Arts Organizations Is 5.33% in One Figure and 0.59% in Another

This memo is for anyone reading the NTEE figures who wonders why two of them disagree. They are not a mistake. They answer different questions, and the gap between them is itself the finding worth reporting for some NTEE groups.

## Walking through group A (Arts, Culture & Humanities)

1. There are 56,171 arts organizations in the frame this project measures — the **501C3-charity mission frame**. The classifier reads each organization's mission text and labels 2,995 of them religious. Divide 2,995 by 56,171 and you get 5.33%. That number is the **classified share**: the share of organizations the classifier happened to label religious. It counts classifier decisions, not organizations that are actually religious.
2. To find out how often the classifier is right, a person read a representative sample of 47 arts organizations by hand — the **Anchor** for this group. Comparing the classifier's labels to the human reading on those 47 rows shows how often the classifier agrees with a person, and how often it does not.
3. Arts is a group where almost nothing is religious to begin with. When true positives are rare, even a small false-positive rate produces many false positives relative to the few true ones. So most of those 2,995 labels are probably wrong — the classifier is inventing religious organizations in a group that barely has any.
4. The pipeline measures that error rate on the 47 human-read Anchor rows and removes it mathematically (a PPI + Rogan–Gladen correction). The result for group A is a **corrected estimate** of 0.59%, with a range of 0.09% to 1.09%. That is the pipeline's best estimate of the true share of religious arts organizations — about nine times smaller than the raw classified share.
5. Why a range, and why so wide relative to the point estimate: the correction rests on only 47 human-read rows. Fewer rows means less certainty, so the range is wider. If a group has fewer than `ntee_min_n` (10 by default) Anchor rows, the correction cannot be trusted at all, and the group is **suppressed** — no estimate is reported for it.
6. The correction does not always shrink the number. In group P (Human Services), the classified share is 15.06% but the corrected estimate is 19.45%. In a group where a large share of organizations really are religious, the classifier's mistake runs the other way: it misses more religious organizations than it invents, so the raw share undercounts and the correction pushes the number up.

## The rule in one sentence

The classifier's raw output is biased toward the group's own base rate: it over-calls where religion is rare (Arts, Environment, Animal-Related, Recreation) and under-calls where religion is common (Human Services, Religion-Related). The **corrected estimate** removes that bias using the human-coded Anchor; the **classified share** does not.

## Which number to use

Only the **corrected estimate**, from `prevalence_by_ntee.csv`, is **prevalence**. The **classified share** and **mean score**, from `ntee_descriptives.csv`, describe what the classifier did, not how many organizations are actually religious. Figures 1–3 report classifier output and never use the word "prevalence." Figure 4 puts classified share and corrected estimate side by side, on purpose, so the gap is visible rather than something a reader has to discover by comparing two separately captioned files.
