# Git Workflow

## Conventional Commits

All commits must follow **Conventional Commits** for clear history and automated changelogs.

### Format

```
<type>(<scope>): <subject>
<blank line>
<body>
<blank line>
<footer>
```

### Types

| Type       | Use For                    |
| ---------- | -------------------------- |
| `feat`     | New features               |
| `fix`      | Bug fixes                  |
| `refactor` | Code refactoring           |
| `docs`     | Documentation              |
| `test`     | Adding/updating tests      |
| `chore`    | Build, dependencies, CI/CD |
| `perf`     | Performance improvements   |
| `style`    | Code formatting            |

### Subject Rules

- Imperative mood: "add" not "added"
- No capitalization at start
- No period at end
- Max 50 characters

### No automatic attribution trailers

Do **not** add `Co-Authored-By`, `Signed-off-by`, or any other automatic attribution
trailer to commit messages. The human author controls all footers. AI agents must
omit attribution trailers entirely.

### Examples

```
feat(genSOvillages): add village-level treatment aggregation

fix(reproducibility): set seed before sample() in clustering

docs: update git workflow examples

refactor(conflicts): extract geocoding logic to separate function

Splits conflict data processing into three stages:
1. Raw data loading
2. Geocoding and validation
3. Spatial aggregation

Closes #45
```

## Branching Strategy

The integration branch is **`master`** (not `main`). Work happens on typed feature branches, and **nesting is allowed** (a sub-branch off another feature branch). PRs go **hierarchically into `master`** and are usually reviewed and merged **manually**.

- **`master`** — integration branch; stable, reviewed code
- Feature branches: `feature/short-description`
- Fix branches: `fix/short-description`
- Refactor branches: `refactor/short-description` (nesting allowed)

### Workflow

```bash
# Branch off master (or off a parent feature branch when nesting)
git checkout -b feature/learning-curve-sweep master

# Commit with conventional commits
git commit -m "feat(training): add learning-curve sweep over silver N"

# Push to remote
git push origin feature/learning-curve-sweep

# Open a PR into master (or into the parent branch); merge is handled manually
```

### Branch Naming

- Use hyphens, lowercase
- Descriptive: `refactor/harmonize-pipeline`
- Avoid: `feature/test`, `feature/fix1`

## Multi-File Commits

Group logically related changes:

**Good:**

```bash
git commit -m "refactor(annotate): extract source_id builder

src/binary_classifier/annotate/annotators/base.py — add _build_source_id()
src/binary_classifier/annotate/bakeoff_prompts.py — use it
src/binary_classifier/annotate/run_annotation.py — use it"
```

**Avoid:**

- Mixing unrelated changes
- Commits touching more than 5 files unless unified

## Pull Request Process

1. **Create PR** with descriptive title
2. **PR description:** What changed and why, link related issues
3. **Code review:** Address feedback
4. **Merge:** Use "Squash and merge" for feature branches

## Undoing Changes

```bash
# Undo unstaged changes
git checkout -- scripts/run_pipeline.py

# Undo staged changes
git reset HEAD scripts/run_pipeline.py

# Undo last commit (keep changes)
git reset --soft HEAD~1

# Undo last commit (discard changes)
git reset --hard HEAD~1
```

## Tips

1. **Small and focused** — One logical change per commit
2. **Atomic** — Each commit should work independently
3. **Descriptive** — Read commit in 6 months, understand why
4. **No debug code** — Don't commit `print()`/`breakpoint()`; use `logging` instead
