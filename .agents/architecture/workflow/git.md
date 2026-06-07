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

| Type | Use For |
|------|---------|
| `feat` | New features |
| `fix` | Bug fixes |
| `refactor` | Code refactoring |
| `docs` | Documentation |
| `test` | Adding/updating tests |
| `chore` | Build, dependencies, CI/CD |
| `perf` | Performance improvements |
| `style` | Code formatting |

### Subject Rules

- Imperative mood: "add" not "added"
- No capitalization at start
- No period at end
- Max 50 characters

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

**Main-only (trunk-based) workflow:**

- **main** — Production-ready, stable code
- Feature branches off main: `feature/short-description`
- Fix branches: `fix/short-description`
- Refactor branches: `refactor/short-description`

### Workflow

```bash
# Create feature branch from main
git checkout -b feature/laki-grievances-integration main

# Commit with conventional commits
git commit -m "feat(conflicts): integrate HISCOD geocoding"

# Push to remote
git push origin feature/laki-grievances-integration

# Create pull request against main
# After review, merge into main
```

### Branch Naming

- Use hyphens, lowercase
- Descriptive: `feature/laki-grievances-integration`
- Avoid: `feature/test`, `feature/fix1`

## Multi-File Commits

Group logically related changes:

**Good:**

```bash
git commit -m "refactor(spatial): extract aggregation function

src/functions/spatialFunctions.R — New aggregation_by_grid()
src/scripts/laki/genSOcells.R — Use new function
src/scripts/laki/genSOpolys.R — Use new function"
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
git checkout -- src/scripts/analysis.R

# Undo staged changes
git reset HEAD src/scripts/analysis.R

# Undo last commit (keep changes)
git reset --soft HEAD~1

# Undo last commit (discard changes)
git reset --hard HEAD~1
```

## Tips

1. **Small and focused** — One logical change per commit
2. **Atomic** — Each commit should work independently
3. **Descriptive** — Read commit in 6 months, understand why
4. **No debug code** — Don't commit `print()` or `browser()`
