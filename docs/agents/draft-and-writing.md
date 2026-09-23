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

## Tables

A table lives in `tables/` in the Overleaf project as a **fragment**. The same fragment is used by the manuscript and by the slides, and each of those supplies everything else around it.

- **The fragment** is one bare `tabular`, from `\begin{tabular}` to `\end{tabular}`, with `booktabs` rules. It has no `table` float, `\caption`, `\label`, note, font size or `\arraystretch`.
- **Its width is fixed.** Numeric columns are `l`/`c`/`r`, and a text column is `p{<length>}` with `\raggedright\arraybackslash`. Never use `tabularx`, `X` columns or `\textwidth`/`\linewidth` inside a fragment: that width depends on the page the fragment lands on, and a fragment like that cannot be scaled onto a slide.
- **Size a text table to the manuscript.** Choose the `p{}` widths so that the table fills the manuscript's `\textwidth` at the manuscript's table font size. The slides then scale it down.
- **In the manuscript**, the surrounding markup is a `table` float with `\caption`, `\label` and the font size, then `\begin{adjustbox}{max width=\textwidth}` around the `\input`, then `tablenotes`. The `adjustbox` only shrinks a table that is too wide and never enlarges one.
- **In the slides**, the surrounding markup is `\slidetable[width=…, caption={…}, note={…}]{<path>}`, which scales the fragment to `width=` or to the height ceiling, whichever binds. Adjust the size with `width=` on each slide, and never edit the fragment to make it fit a slide.
- An exporter that writes a fragment for the manuscript must follow this shape. The caption and note belong in the exporter's spec, not in the `.tex` file.

## Voice

- The canonical author voice profile is `../voice-profile/docs/research/voice-profile/profile.md` (sibling of this repo, present on every machine). It overrides the `voice-profile` / `humanize` / `proofread` skills' default path (`docs/research/voice-profile/…`).

## Division of labor

- `NonProfitData` → panel data, missions corpus, statistics
- `nonprofits-binary-classifiers` → religious-organization classification
- `nonprofits-topic-modelling` → topic models and case studies
