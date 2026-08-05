# Releasing

Maintainers release `semantic-factors` from GitHub using PyPI trusted publishing. The release
workflow never uses a long-lived PyPI password or API token.

## Prepare the release

1. Update `src/semantic_factors/version.py` and date the matching `CHANGELOG.md` entry.
2. Run the local checks documented in `CONTRIBUTING.md`.
3. Confirm the `main` branch CI and security checks are green.
4. Confirm the PyPI `semantic-factors` project trusts this repository's `pypi` environment and
   `.github/workflows/publish.yml` workflow.
5. Create an annotated tag matching the package version, for example `v0.1.0`, and push it.
6. Create and publish a GitHub release from that tag.

Publishing the GitHub release builds the distribution again, validates it with Twine, and
publishes it to PyPI through OpenID Connect. A manual workflow run builds and validates artifacts
but deliberately cannot publish them.

After publishing, install the package from PyPI into a clean environment and rerun the CLI smoke
checks from `CONTRIBUTING.md`.
