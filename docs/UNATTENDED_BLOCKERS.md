# Unattended Workflow Blockers

## 2026-09-26 — GitHub credentials unavailable

- Branch: `codex/unattended-2days`
- Completed local commit: `8147890 test: cover ML1M data invariants`
- Validation: `15 passed in 6.91s`; `git diff --check` passed.
- Failed operation: `git push origin codex/unattended-2days`
- Exact error: `fatal: could not read Username for 'https://github.com': No such device or address`
- Impact: the validated commit exists locally but cannot be pushed to the
  required remote branch. Work stopped according to the unattended workflow's
  credential stop condition.
- Required resolution: provide working GitHub credentials for the configured
  HTTPS remote, or change the remote to an already authenticated transport.
