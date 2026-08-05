# Security policy

Please do not report suspected vulnerabilities in a public issue. Use this repository's
[private vulnerability reporting](https://github.com/NosibleAI/semantic-factors/security/advisories/new)
with enough detail to reproduce the issue, or contact NOSIBLE privately through
[nosible.com](https://www.nosible.com). Do not include API keys, customer data, World responses,
or other sensitive material in an issue, pull request, CI log, or artifact.

Security reports will be acknowledged as soon as practical. We will investigate, coordinate a
fix and disclosure where needed, and credit reporters who want attribution.

The package should never log credentials or event payloads. If a credential may have been exposed,
rotate it immediately and then investigate the exposure separately from this repository.
