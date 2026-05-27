# AI Agent Git Practices

This repository may be edited by AI coding agents. Follow these git practices to protect user work, keep history readable, and avoid committing local artifacts.

## Core Rules

* Inspect `git status --short --branch` before changing files, staging, committing, pulling, rebasing, merging, or pushing.
* Treat existing uncommitted changes as user work unless you made them in the current task.
* Never discard, overwrite, reset, checkout, rebase away, or delete user changes unless the user explicitly asks for that exact operation.
* Keep edits scoped to the requested task. Do not include unrelated cleanup, formatting, or generated output in the same commit.
* Stage intentional files explicitly by path. Do not use `git add -A` unless the full worktree has been reviewed and confirmed in scope.
* Review `git diff --staged` before committing.
* Use clear, focused commit messages that describe what changed and why it matters, not the tool that made it.
* Never force-push, including `--force-with-lease`, unless the user explicitly approves the exact command after you explain why it is necessary and what remote history it may replace.
* Never add Claude, Codex, or any other AI CLI/tool as a GitHub collaborator, assignee, reviewer, or commit co-author.

## Working Tree Safety

Before making changes:

1. Run `git status --short --branch`.
2. Identify modified, staged, deleted, and untracked files.
3. Decide which files belong to the current task.
4. Leave unrelated files untouched.

If unrelated changes exist in files you need to edit, read them carefully and work with them. Do not revert them. If the changes make the task unsafe or ambiguous, stop and ask the user how to proceed.

## Staging And Commits

Use explicit staging:

```bash
git add path/to/file.py path/to/README.md
```

Avoid broad staging commands unless the entire diff is intentionally part of the task:

```bash
git add -A
git add .
```

Before committing:

* Run `git diff --check`.
* Run relevant tests or checks when available.
* Inspect `git diff --staged`.
* Confirm no secrets, local artifacts, caches, model weights, or unrelated files are staged.

Commit messages must be written by the agent, reviewed against the staged diff, and kept short, specific, and action-oriented. Use the imperative mood when possible, name the behavior or documentation changed, and do not mention AI tools.

Good commit messages:

```text
Add inference API adapter
Fix markdown lint formatting
Document agent git practices
```

Avoid vague or tool-centered messages:

```text
Update files
Changes from Claude
Codex edits
WIP
```

Do not add AI tool attribution trailers such as `Co-authored-by: Claude`, `Co-authored-by: Codex`, or similar lines for other AI CLIs unless the user explicitly requests them.

## Pulling, Fetching, And Pushing

Before pushing:

1. Run `git status --short --branch`.
2. Fetch or pull if the remote branch may have changed.
3. Resolve divergence by preserving both local and remote work unless the user explicitly requests another strategy.
4. Push the current branch to the intended remote.

Do not assume `origin` points to the correct repository. Check `git remote -v` before pushing when the remote target matters.

If a push is rejected because the remote has new commits, fetch and inspect the difference before merging, rebasing, or retrying. Do not force-push or use `--force-with-lease` to bypass the rejection unless the user explicitly approves after you explain why rewriting the remote branch is necessary.

## Files That Should Not Be Committed

Do not commit:

* `.DS_Store` or other OS metadata.
* `.env`, credentials, tokens, private keys, or local secret files.
* Virtual environments and dependency caches.
* Test, lint, coverage, and build caches.
* Downloaded model weights such as `*.pt`, `*.onnx`, and `*.engine`.
* Large generated assets unless the user explicitly asks to version them.

If a generated file is needed to run the project, document how to create or download it instead of committing it.

## Destructive Commands

Treat these commands as destructive unless the user explicitly asks for them:

```bash
git reset --hard
git checkout -- path
git clean -fd
git push --force
git push --force-with-lease
git branch -D
```

Prefer non-destructive inspection first:

```bash
git diff
git diff --staged
git log --oneline --decorate --graph --max-count=20
git status --short --branch
```

## Communication

When reporting git work to the user, include:

* Files changed.
* Commit hash and message, if committed.
* Remote and branch, if pushed.
* Checks run and their results.
* Anything intentionally left uncommitted.
