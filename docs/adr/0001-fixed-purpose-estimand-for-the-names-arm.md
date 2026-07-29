# 1. Fixed purpose estimand for the names arm

Date: 2026-07-28

## Status

Accepted

## Context

The **names arm** applies the mission-trained classifier to organization names by
**cross-field transfer**. Before any labelling or evaluation could be designed, one
question had to be settled: what does a positive label *mean* for a name?

The missions construct (`src/binary_classifier/annotate/prompts/v1.txt`) defines
religious as an *observable religious or spiritual purpose, tradition, or motivation
as a core driver of the organization's work*, explicitly labelled as observable —
"we label what the text says, not what the organization 'really is' in a latent
sense." It carries a `saint_name_only` domain code whose instruction is to abstain.

Names do not carry that evidence. They carry religious **identity or affiliation**:
denominational tokens, saint names, "Trinity", "Grace". The distinction is not ours —
it is the Sider & Unruh typology, and the June 2026 stub
(`.agents/stubs/names_with_missions_vs_missions-idea.md`) reached it independently
and listed the estimand as its first open decision.

So the mission construct and the evidence a name supplies are systematically
misaligned, and three targets were genuinely available.

## Decision

**The names arm uses the same construct as the missions pipeline, unchanged.**
Positive means observable religious *purpose* as a core driver of the work.

Consequences that follow directly, and that all downstream labelling must honour:

- "St Mary's Hospital" is a true **negative** — a saint name is not a purpose.
- "Trinity Health" is a true **negative** — faith-founded identity without purpose.
- "First Baptist Church" is a true **positive** — a named tradition.
- All rung-4 hand coding is performed under this rubric, not a name-native one.

## Alternatives considered

**A new auspice/identity construct.** Positive means religious auspice or
affiliation — what a name can actually carry. More natural for the input, and would
score better. Rejected because it measures a *different target* than the missions
arm: the two could not be pooled, compared, or reported as one prevalence figure
without equivocation, and the project's standing constraint is that the missions
results stay first-class and additive-only.

**Either-signal-positive (purpose OR auspice).** Maximises measured prevalence and
is arguably the sociologically interesting target. Rejected because it silently
redefines what the existing composite prevalence figure refers to — a reinterpretation
of shipped numbers, which the additive-only constraint forbids.

**Defer and decide from pilot behaviour.** Rejected as fitting the construct to the
model's quirks rather than to the research question.

## Consequences

**Accepted cost — a low ceiling by construction.** Names carrying identity but not
purpose are negatives, so recall is capped well below what a name-native construct
would report. Some of what looks like transfer failure will be the model correctly
agreeing with the construct. Evaluation must not read that as a defect.

**Benefit — the arms stay commensurable.** Mission-based and name-based outputs
measure the same target, so the paired overlap comparison is meaningful and any
future pooling is defensible.

**It sharpens the hypothesis worth testing.** Under this estimand the interesting
claim is that the encoder beats a lexicon on **precision** — by rejecting the
saint-named seculars a keyword rule fires on — rather than on recall. That is the
falsifiable prediction the names arm exists to test.

**It is expensive to reverse.** Rung-4 gold coded under this rubric cannot be
reinterpreted under an auspice construct without recoding; the boundary cases are
precisely where the two disagree.
