# Draft and Writing Contract

Identical in `NonProfitData`, `nonprofits-binary-classifiers`, and `nonprofits-topic-modelling` (the three sibling repos under `~/Documents/Projects`). Keep the three copies identical. Term definitions live in each repo's `CONTEXT.md`.

## The manuscript and the symlink

- **The manuscript** is the shared Overleaf document (current title *Written in the Taxes*; may change), co-written with collaborators.
- It is reached from every repo as `paper/draft`, a symlink to the **Overleaf project** — a Dropbox directory whose absolute path is machine-specific. Resolve the manuscript through the symlink; the target path is never hardcoded.
- The manuscript is a live collaborator document synced via Dropbox/Overleaf, with no git rollback: every write through `paper/draft` is an immediate edit to the shared paper. Keep `paper/` contents out of git.

## Write policy

- This repo is **read-only** against the Overleaf project. The project self-contains its figures and tables; pipeline-generated **draft artifacts** (figures, tables, stats) are copied in *manually* by a human. Pipelines never write into `paper/draft`.

## Versioning and slides

- The manuscript is versioned by filename (`main01.tex` … `mainNN.tex`), not by git. Always fork or edit the latest `mainNN.tex`; leave older versions untouched.
- Slides live in a `slides/` folder inside the Overleaf project.

## Voice

- The canonical author voice profile is `../voice-profile/profile.md` (sibling of this repo, present on every machine). It overrides the `voice-profile` / `humanize` / `proofread` skills' default path (`docs/research/voice-profile/…`).

## Division of labor

- `NonProfitData` → panel data, missions corpus, statistics
- `nonprofits-binary-classifiers` → religious-organization classification
- `nonprofits-topic-modelling` → topic models and case studies
