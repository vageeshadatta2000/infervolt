# Security policy

infervolt launches inference servers and runs load against them, and can call LLM APIs with your keys.

- Never commit `.env`, `~/.thunder`, or any file containing tokens. A gitleaks pre-commit hook is configured.
- Report vulnerabilities privately via GitHub Security Advisories on this repository. We aim to respond within 7 days.
- Supported versions: the latest minor release.
