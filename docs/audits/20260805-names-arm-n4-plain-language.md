# What N4 Tells Us About Religious Names

This memo explains the N4 results in ordinary language. It is based on `data/processed/names/name_validation.json` and answers three questions:

1. Can the fine-tuned mission model recognize religious organization names?
2. Do religious missions tend to go with religious names?
3. Is the names model ready to use as a final classifier?

## Short Answer

The fine-tuned model can recognize useful signal in organization names. It is especially good at recognizing explicit religious institutions, such as churches, synagogues, mosques, and Buddhist or Hindu centers.
That does not mean that a religious mission always has a religious name. Many organizations have religious missions but neutral or secular-sounding names. Names and missions are related, but they are not interchangeable sources of evidence.

The current evidence supports using the model as a **provisional screening or ranking tool**. It does not yet support treating its score as a calibrated probability, choosing a final production threshold, or using it to estimate the population prevalence of religious organizations.

## What Was Tested

N4 used two kinds of comparison.

### 1. Human-coded BMF-only names

We drew 400 organizations from the BMF-only population and had them coded by humans using the project's purpose-based definition:

> A positive means that the name gives evidence of observable religious or
> spiritual purpose, tradition, or motivation as a core driver of the work.

This was a deliberately difficult sample. It included 335 religious labels and 65 non-religious labels. It was not a simple random sample with a natural balance of religious and non-religious organizations.

This is the part of N4 that uses the **only-BMF names**. These organizations do not have a matching panel mission, so they cannot be used for a names-versus-missions comparison. Instead, they answer the direct question: **when humans judge the name itself, does the model get that name judgment right?**

### 2. Organizations with both names and missions

We also compared names with the existing mission-model results for the same organizations. After removing organizations that had been used elsewhere in model development and organizations without usable mission scores, 471,053 organizations remained.

This is a large comparison, but the mission result is still a model result, not an independent human judgment of the name. It is therefore useful for detecting broad agreement or disagreement, but it cannot prove that the names model is correct.

This second comparison does **not** use the BMF-only organizations. It uses panel organizations that have both a usable name and a usable mission result.

## Can the Model Recognize Religious Names?

Yes, the model is doing something meaningful.

Using the current temporary score cutoff of `0.5`, and using the suffix-stripped name, the 400 human-coded cases produced this simple picture:

| Human label   | Model says religious | Model says non-religious | Total |
| ------------- | -------------------: | -----------------------: | ----: |
| Religious     |                  286 |                       49 |   335 |
| Non-religious |                   13 |                       52 |    65 |
| Total         |                  299 |                      101 |   400 |

In plain language:

- The model found about **85% of the human-coded religious names**: 286 out of 335.
- It missed 49 religious names.
- It incorrectly flagged 13 of the 65 non-religious names.
- Of the 299 names it called religious, 286 were religious in this sample. That is about **96% precision within this sample**.

The suffix-retaining version was slightly weaker: it found 278 of the 335 religious names and incorrectly flagged 12 of the 65 non-religious names. This is why the suffix-stripped version is the better provisional input.

The 400 cases were intentionally enriched for difficult and religious-looking names, so these percentages should not be read as the expected percentages for every organization in the country. They show that the model has real signal and can be useful, not that the final population error rate is already known.

## What Kind of Names Does It Understand?

The diagnostic examples show a clear pattern:

- `First Baptist Church`: very high score.
- `Beth Shalom Synagogue`: very high score.
- `Al Noor Mosque`: very high score.
- `Lakshmi Hindu Center`: very high score.
- `Lotus Buddhist Center`: very high score.
- `First Community Soccer Club`: very low score.
- `Lakshmi Civic Center`: very low score.
- `Lotus Community Center`: very low score.

This is good evidence that the model detects explicit religious institutions and tradition words.

It also shows the main risk:

- `Saint Luke Academy`: very high score, even though a saint's name alone is not enough under the project's purpose definition.
- `Beth Shalom Center`: very high score even when the religious institution word `synagogue` is removed.
- `Al Noor Center`: only a middling score, despite the religious association of the name.

So the model sometimes recognizes religious identity or heritage rather than religious purpose. That is not a technical accident; it is a real limitation of trying to infer an organization's work from its name alone.

## Are Religious Missions and Religious Names Correlated?

Yes, but the relationship is **moderate, not strong**, and the two signals are not interchangeable.

Here are the actual numbers for the `471,053` organizations with both usable names and mission results. The BMF-only human-gold results above are the evidence about whether the names model can predict names. The numbers below answer a separate question: whether name-based scores tend to agree with mission-based scores when both fields exist.

### Continuous scores

The Pearson correlation between the name score and the mission score was:

| Name input       | Correlation with mission score |
| ---------------- | -----------------------------: |
| Suffix-stripped  |                        `0.570` |
| Suffix-retaining |                        `0.565` |

A correlation of `1.0` would mean that organizations are ranked identically by names and missions. A correlation of `0` would mean no linear relationship. The observed value, about `0.57`, means that the two scores generally move in the same direction, but with substantial disagreement.

### Binary comparison

For this comparison, N4 gave the name model exactly `15,536` positive calls, or `3.30%` of the organizations. This was done to match the number of positive calls from the lexicon; it was not a final production threshold.

The mission model called `61,032` organizations positive, or `12.95%`. Comparing the suffix-stripped name calls with those mission calls gives:

| Result                          | Organizations | Plain meaning                                          |
| ------------------------------- | ------------: | ------------------------------------------------------ |
| Both name and mission positive  |      `11,845` | Name model found 11,845 of the mission-model positives |
| Name positive, mission negative |       `3,691` | Name model's additional positive calls                 |
| Name negative, mission positive |      `49,187` | Mission positives missed by the name model             |
| Both name and mission negative  |     `406,330` | Both methods called them negative                      |

That means:

- The name model captured **19.4%** of the organizations that the mission model called religious: `11,845 / 61,032`.
- Of the organizations the name model called religious, **76.2%** were also called religious by the mission model: `11,845 / 15,536`.
- The two binary labels agreed for **88.8%** of organizations: `418,175 / 471,053`.
- Almost all of that overall agreement came from shared negatives: **99.1%** of the mission-negative organizations were also name-negative.

The suffix-retaining version is nearly identical: it captures `19.6%` of the mission positives, has `76.9%` agreement among its positive calls, and agrees with the mission labels for `88.8%` of all organizations.

The important result is therefore not simply “89% agreement.” The fuller statement
is:

> Names and missions have a moderate score correlation of about `0.57`. At the
> tested conservative name operating point, names identify about one in five of the
> organizations that missions identify as religious, while correctly remaining
> negative for nearly all mission-negative organizations.

This is exactly what we would expect if names are good at identifying explicit religious organizations but miss many organizations whose religious purpose is not visible in their name.

The appropriate human interpretation is:

> A religious-looking name is useful evidence that an organization may have a
> religious mission, but a non-religious-looking name does not demonstrate that its
> mission is non-religious.

## Did the Fine-Tuned Model Beat the Simple Lexicon?

No, not in the N4 comparison.

The comparison gave the model and the lexicon the same number of positive calls. At that matched level:

- Lexicon precision: about **78%**.
- Fine-tuned model precision: about **76%** for suffix-stripped names and **77%** for suffix-retaining names.

The difference is small, but it goes in the wrong direction for the preregistered hypothesis that the fine-tuned model would improve precision by rejecting misleading religious words.

This does not mean the fine-tuned model is useless. It means that, on this specific paired comparison, it did not demonstrate an advantage over the simpler rule.

## What Does the External-Flag Test Mean?

The model also followed a known population difference in IRS religious-auspice flags:

- BMF-only organizations had an external religious flag rate of about **20%**.
- Panel organizations had a rate of about **9%**.
- The model's positive rate was about **22%** for BMF-only organizations and **11%** for panel organizations.

This is a useful sanity check. The model reacts in the expected direction when the population composition changes. However, the external flag measures religious auspice or affiliation, while the project target is observable religious purpose. The test therefore cannot establish that the model is measuring the intended purpose construct.

## What We Can Say Now

We can responsibly say:

- Organization names contain enough information for useful religious-name screening.
- The fine-tuned model recognizes explicit religious institutions and traditions.
- Suffix-stripped names are slightly better than suffix-retaining names in the human-coded sample.
- Religious missions and religious-looking names are positively related, but the relationship is incomplete and only moderate.
- The model is not clearly better than the strong-tradition lexicon in the current paired comparison.

## What We Cannot Say Yet

We cannot yet say:

- The model's raw score is a real probability for a name.
- `0.5`, `0.0577`, `0.0937`, or `0.6083` is the correct production cutoff for names.
- The model's precision and recall in the 400-name sample are the population-wide precision and recall.
- A non-religious-looking name means the organization's mission is non-religious.
- The names arm can produce a defensible population prevalence estimate.

## Practical Recommendation

For now, use the suffix-stripped model as a **ranking and review aid**:

- High scores can prioritize organizations for review or likely-positive screening.
- Middle scores should be treated as uncertain, not as calibrated probabilities.
- Low scores should not be treated as proof of a non-religious mission.

Before making final binary classifications, the next useful analysis is a human-gold threshold table: evaluate several cutoffs on the 400 coded names and report how many religious names are found and how many non-religious names are mistakenly flagged at each cutoff. That would turn the current diagnostic evidence into an explicit, understandable operating choice.
