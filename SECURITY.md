# Security policy

Report vulnerabilities privately through GitHub's security-advisory feature.

The MVP deliberately contains no live exchange client. Never grant withdrawal permission to
any research or trading key. If authenticated read-only sources are added:

- use a dedicated low-privilege account and environment-injected secrets;
- restrict source IPs where supported;
- rotate leaked or logged credentials immediately;
- prevent secrets from entering experiment parameters and artifacts;
- pin and scan dependencies and container images.

Live execution, leverage, custody, and withdrawal functionality are not accepted in this
repository without a separate threat model and maintainer decision.

