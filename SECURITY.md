# Security Policy

## Supported Versions

| Version | Supported |
|---------|-----------|
| 0.2.x   | ✅ Current release — patches applied |
| < 0.2   | ❌ Not supported — upgrade to 0.2.x |

apple-basefm follows [SemVer](https://semver.org/). During the pre-release
period (v0.x), breaking changes may occur at minor version bumps; security
fixes are back-ported to the most recent minor release only.

## Reporting a Vulnerability

**Do not open a public GitHub issue for security vulnerabilities.**

Report vulnerabilities via GitHub's private security advisory system:

> **[Report a vulnerability →](https://github.com/zombat/Apple-BaseFM/security/advisories/new)**

Alternatively, send a PGP-encrypted email (key on file in the repository) to
the maintainer. Include:

- A description of the vulnerability and its potential impact
- Steps to reproduce (proof-of-concept code is welcome)
- Any suggested mitigations you have already identified

We aim to acknowledge reports within **48 hours** and provide an initial
assessment within **7 days**. Critical and high-severity issues will be
patched and released within **14 days** of confirmation.

## Disclosure Policy

We follow [coordinated disclosure](https://en.wikipedia.org/wiki/Coordinated_vulnerability_disclosure):

1. Reporter submits details privately.
2. Maintainer confirms and begins a fix.
3. Fix is prepared in a private branch.
4. A new release is cut and published to PyPI.
5. A GitHub Security Advisory is published after the fix has been available
   for at least 7 days, giving users time to upgrade.

We request that reporters do not publicly disclose details until a fix has
been released, or until 90 days have elapsed from the initial report
(whichever comes first).

## Scope

In scope:
- The `apple_basefm` package distributed via PyPI (`pip install apple-basefm`)
- The `scripts/` tooling distributed with the source distribution
- GitHub Actions workflows in `.github/workflows/`

Out of scope:
- The Apple Foundation Models SDK itself (`apple-fm-sdk`) — report those to Apple
- The `mlx` / `mlx-lm` libraries — report those to the Apple MLX team
- The `dspy` library — report those to the Stanford NLP DSPy team
- Issues affecting only development tooling in `.github/skills/` (not in the wheel)

## Security Design Notes

- **No hard runtime dependencies.** `pip install apple-basefm` installs nothing
  additional, minimising transitive supply-chain exposure.
- **Subprocess calls** use list form with `shell=False` throughout the package.
- **All subprocess calls in the package** use bound argument lists; no user
  input is interpolated into shell commands.
- **LRU-bounded tool cache** (256 entries) prevents unbounded memory growth from
  adversarial tool registration in DSPy pipelines.
- **Input validation** is performed at all public API boundaries per `GOVERNANCE.md`.
- **Static analysis** is enforced pre-release via `scripts/guard.py --strict`
  (exit 1 on any violation), `ruff`, and `mypy`.
