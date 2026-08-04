# Comments and Documentation

## Default Style

This repository prefers heavy, scan-friendly comments so a reader can follow a script by sections before reading the implementation details.

## Required Pattern

- Add a script header when a file contains a pipeline or multi-step analysis.
- Add section headers for code blocks and sections with a description under the section delimiters to explain what the code section does.
- Place comments above code, never inline.
- Explain business logic, economic reasoning, data provenance, and why a transformation exists.
- When in doubt, prefer a brief section comment over no comment.

## What to Comment

- merge assumptions and key choices
- filters, exclusions, and recodes
- methodology decisions
- non-obvious calculations
- the role of each major section in the larger pipeline

## What Not to Do

- Do not narrate trivial syntax line by line.
- Do not use filler comments that only restate the next line.
- Do not hide rationale inside inline comments.

## Language Notes

- Python: add docstrings for reusable functions and classes when the surrounding file uses them.
