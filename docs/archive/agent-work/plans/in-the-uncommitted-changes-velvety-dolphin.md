---
created: 2026-07-05
---

# Sprint: Publication-grade vector wordclouds — renderer-perfect **and** selectable

## Context

Stage-10 (`scripts/10_visualize.py`) emits 12 wordclouds (frequency × distinctive,
uni/bi/tri-gram, religious/non-religious) as PNG + SVG + PDF. The PNG (PIL
`WordCloud.to_image()`) looks excellent, but the SVG **drifts** from it: words are spaced
differently and can overlap.

**Root cause (confirmed).** `amueller/wordcloud` packs words using PIL/FreeType metrics, then
`cloud.to_svg()` exports `<text>` nodes that a *different* engine (rsvg/browser) re-lays-out
with different glyph advances/kerning; `optimize_embedded_font=True` also strips hinting. A
text-node SVG is therefore renderer-dependent by construction and cannot match the raster.
The pipeline also shells out to `rsvg-convert` for the PDF.

**Sprint goal:** ship all 12 wordclouds as (1) the unchanged beautiful PNG, (2) a PDF that is
renderer-perfect *and* selectable, and (3) an SVG that is renderer-perfect *everywhere*
(vector paths) yet still selectable/searchable in browsers (invisible `<text>` layer).

**Scope & anti-over-engineering guardrails.** Touch **only the wordcloud path** — no other
stage-10 figure, drawer, or the shared `_save_plot` choke point. **No new dependency**; the
`rsvg-convert` subprocess is *removed* (net less infra). Only the **SVG** carries the
paths+invisible-text mechanism; the **PDF** stays a single plain matplotlib render, and the
**PNG** is the unchanged one-liner. Prefer the smaller validated variant if any piece sprawls.

## Decisions (confirmed with user)

- **Fidelity:** "Bulletproof paths + invisible text" for SVG — chosen over the simpler
  d3-equivalent text-node fix, knowing the trade-off (extra code vs. surviving a font-less
  renderer).
- **Format split:** PNG = `to_image()` (unchanged); PDF = embedded-font text; SVG = paths +
  invisible text.
- **Drop `rsvg-convert`;** stay pure-Python (matplotlib + wordcloud + fontTools, all already
  deps; `xml.etree` from stdlib).

## Method notes (prototype-verified this session, in `/tmp`)

- **SVG `@font-face` embedding does not help.** Embedding DejaVu *Serif* under the name
  "DejaVu Sans" still rendered **sans-serif** in both `rsvg-convert` and Inkscape — they
  ignore embedded WOFF and use system fonts. Font-embedding only helps browsers → only
  vector **paths** are guaranteed identical on every renderer.
- **PDF differs from SVG: it always embeds its fonts.** matplotlib PDF with `pdf.fonttype=42`
  embeds a DejaVuSans subset → selectable *and* renderer-perfect (verified: `pdffonts` shows
  `emb yes sub yes`, `pdftotext` extracts words, render matches PIL). No paths needed for PDF.
- **Dual-layer SVG works** ("searchable-scanned-PDF" trick): visible `<path>` art + an
  invisible (`fill-opacity:0`) `<text>` layer. Verified: merged SVG rasterizes (rsvg) to a
  clean, path-only image identical to PIL while carrying 100 real `<text>` nodes.
- **Layout coordinate math** mirrors `WordCloud.to_svg` exactly: baseline from PIL
  `font.font.getsize` / `getmetrics`; `Image.ROTATE_90` → `rotation=90`; use `cloud.font_path`;
  parse the stored `rgb(r,g,b)` color string. Pixel-sized figure (`figsize=(W/dpi, H/dpi)`,
  axes `[0,0,1,1]`, `set_xlim(0,W)`, `set_ylim(H,0)`, `axis('off')`), draw with
  `ax.text(..., ha='left', va='baseline', rotation_mode='anchor')`.
- **Margin bump** (`~2 → 8`) absorbs the small PIL-vs-matplotlib advance differences so the
  visible layer never overlaps. One visible side effect: slightly airier clouds.

---

## Sprint tickets

### S1 — Shared layout renderer *(do first; foundation for S2/S3)*
- **`viz/wordclouds.py`:** in `_make_wordcloud` add `margin=8` (tunable); keep
  `prefer_horizontal`, `random_state`, RGBA transparency, `color_func`.
- Add `_layout_glyphs(cloud)` yielding per word
  `(word, baseline_x, baseline_y, rotation_deg, rgba)` — baseline math copied from
  `WordCloud.to_svg` (use `cloud.font_path`; parse `rgb(r,g,b)`).
- Add `_render_layout_axes(cloud, dpi=100)` building the pixel-sized figure/axes and drawing
  every glyph via `ax.text(...)` as above.
- **Acceptance:** helper renders a `WordCloud` (from `build_class_wordcloud`) to a matplotlib
  figure whose raster visually matches `cloud.to_image()`.
- **Files:** `viz/wordclouds.py`.

### S2 — PDF writer: embedded-font, selectable, renderer-perfect
- **`viz/wordclouds.py`:** add `write_wordcloud_pdf(cloud, path)` — set `pdf.fonttype=42`,
  call `_render_layout_axes`, `savefig(path, transparent=True)`.
- **Acceptance:** `pdffonts` shows `DejaVuSans … emb yes sub yes`; `pdftotext` extracts the
  words; `pdftoppm` render matches the PNG.
- **Files:** `viz/wordclouds.py`.

### S3 — SVG writer: visible paths + invisible selectable text
- **`viz/wordclouds.py`:** replace `write_wordcloud_svg` (currently `cloud.to_svg`, drifts).
  New impl renders `_render_layout_axes` **twice** — `svg.fonttype='path'` (visible art) then
  `svg.fonttype='none'` (text) — parses both with `xml.etree.ElementTree`, appends the
  `<text>`-bearing `<g>` groups from the text SVG into the path SVG wrapped in
  `<g style="fill-opacity:0">`, and keeps the `viewBox` injection.
  - *Smaller alternative if preferred:* single pass with `PathPatch` (visible, from `TextPath`)
    + `alpha=0` `Text` (invisible) under `svg.fonttype='none'`.
- Drop the now-unused `_VIEWBOX_RE`-only path as needed; keep transparency assertions.
- **Acceptance:** `grep -c "<path"` > 0 and `grep -c "<text"` > 0 with real word strings;
  `rsvg-convert` and `inkscape` both rasterize to a clean, path-only image identical to the
  PNG (invisible layer does not appear, no overlaps).
- **Files:** `viz/wordclouds.py`.

### S4 — Wire into stage 10 + drop rsvg-convert
- **`scripts/10_visualize.py`:** rewrite `_save_wordcloud_plot` →
  `_save_wordcloud_outputs(registry, name, cloud)` (no `imshow`/axes): call
  `cloud.to_image().save(png)`, `write_wordcloud_pdf`, `write_wordcloud_svg`.
- `_maybe_render_wordcloud` calls `build_class_wordcloud(...)` (already returns the cloud with
  `layout_`) instead of `class_wordcloud(ax, ...)`, then `_save_wordcloud_outputs`.
- **Remove** the `subprocess` / `rsvg-convert` block and its raster fallback.
- **`viz/__init__.py`:** export `write_wordcloud_pdf`; keep `write_wordcloud_svg`; update the
  `scripts/10` import list. (`class_wordcloud`/`class_wordclouds` imshow drawers may stay for
  interactive use but are off the stage-10 path.)
- **`pyproject.toml`:** no new dependency; optionally drop any librsvg note.
- **Acceptance:** a stage-10 run writes all 12 wordclouds; no `rsvg-convert` invocation
  remains; missing inputs still degrade gracefully.
- **Files:** `scripts/10_visualize.py`, `viz/__init__.py`, `pyproject.toml`.

### S5 — Tests & lint
- **`tests/test_viz.py`:** build a small two-class synthetic frame, get a cloud via
  `build_class_wordcloud`, assert the SVG contains both `<path` and invisible `<text` nodes;
  assert the PDF has extractable text (skip if `pdftotext` unavailable).
- `ruff check` clean; `pytest tests/test_viz.py` passes.
- **Files:** `tests/test_viz.py`.

## Files touched
- `viz/wordclouds.py` (margin + layout renderer + PDF/SVG writers),
  `scripts/10_visualize.py` (save path + drop rsvg), `viz/__init__.py` (exports),
  `pyproject.toml` (no new dep), `tests/test_viz.py` (assertions).

## End-to-end verification
1. `uv run python scripts/10_visualize.py --config config/religious_missions.yaml` runs clean;
   log shows all 12 wordclouds in `data/processed/figures/`, none skipped; no `rsvg-convert`.
2. **SVG renderer-perfect:** `rsvg-convert wordcloud_*.svg -o out.png` and
   `inkscape wordcloud_*.svg --export-filename=out2.png` both match the `.png` (no
   drift/overlap; visible layer is paths).
3. **SVG selectable:** `grep -c "<text" wordcloud_*.svg` > 0 with real words; invisible layer
   absent from the rasterized render.
4. **PDF selectable + embedded:** `pdftotext wordcloud_*.pdf -` extracts words; `pdffonts`
   shows `DejaVuSans … emb yes`; `pdftoppm` render matches the PNG.
5. **Visual sanity:** clouds slightly airier (margin) but clean; distinctive clouds still show
   religious (gospel, ministry, church…) vs secular (research, community…).
6. `ruff check` clean; `pytest tests/test_viz.py` passes.
