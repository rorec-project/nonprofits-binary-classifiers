# Research handoffs

AI-generated research reports that inform the project's design decisions. Each
report investigates a question against primary sources and ends with a handoff
block of concrete, citable findings.

New research (e.g. from the `research` skill) belongs here — this is the
canonical location for such notes.

## Layout

- `*.md` — the research reports themselves.
- `notebooks/` — the per-report scratchpads. Each report's frontmatter links to
  its scratchpad via a `scratchpad:` field.

## Naming

Reports are prefixed `YYYYMMDD-<track>-<topic>`, where `<track>` is one of:

- `literature` — what the published literature says.
- `replication` — how comparable projects implement it.
- `tech` — concrete tooling, APIs, and implementation options.

Each track has a `*-synthesis-map.md` that indexes its reports and their status.
