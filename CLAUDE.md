# Claude Code Instructions

Follow `AGENT.md` as the full repository policy. This file is the short Claude-specific checklist.

## Default Git Workflow

1. Start with `git status --short --branch`.
2. Read relevant files before editing.
3. Keep changes scoped to the user request.
4. Stage only explicit paths that belong to the task.
5. Run `git diff --check` and relevant project checks before committing.
6. Review `git diff --staged` before committing.
7. Check `git remote -v` before pushing.

## Safety Rules

* Protect user work. Never discard or overwrite uncommitted changes unless explicitly instructed.
* Do not use `git add -A` or `git add .` when unrelated files are present.
* Do not commit `.DS_Store`, secrets, caches, virtual environments, or downloaded model weights such as `*.pt`.
* Do not force-push, use `--force-with-lease`, hard reset, clean, or delete branches without explicit approval.
* Before requesting approval for `git push --force` or `git push --force-with-lease`, explain why rewriting remote history is necessary and what branch/commits may be replaced.
* If remote history diverges, fetch and inspect before merging or rebasing.
* Do not add Claude, Codex, or any other AI CLI/tool as a GitHub collaborator, assignee, reviewer, commit co-author, or attribution trailer.

## Commit Style

Write commit messages from the staged diff. Keep them short, specific, and action-oriented. Prefer imperative mood, name the behavior or documentation changed, and do not mention Claude, Codex, or other AI tools.

Good examples:

```text
Document agent git practices
Fix markdown lint formatting
Add inference service scaffold
```

Bad examples:

```text
Update files
Changes from Claude
Codex edits
WIP
```

When reporting completion, include changed files, commit hash if applicable, pushed branch if applicable, and validation results.
