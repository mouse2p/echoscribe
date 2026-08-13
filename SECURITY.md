# Security policy

## Supported versions

| Version | Supported          |
| ------- | ------------------ |
| 0.1.x   | :white_check_mark: |

## Reporting a vulnerability

Please report privately through GitHub Security Advisories on this repository
(the **Security** tab -> **Report a vulnerability**). Do not open a public
issue. Include a reproduction that runs offline; expected response time is
best effort within a week.

## Threat model notes

echoscribe is an offline research workbench: it runs no network service,
handles no credentials and downloads no models. The realistic attack surface
is parsing of serialized artifacts -- JSONL manifests and sessions, JSON
score/bench files, and torch bundle files. Bundles are loaded with
`torch.load(..., weights_only=True)` by design, and configurations are
required to be JSON-serializable on save; a change that weakens either
property should be treated as a security regression and reported as one.
