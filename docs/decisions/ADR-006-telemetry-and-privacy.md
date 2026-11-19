# ADR-006: Local-First Privacy and Telemetry Architecture

## Status
Accepted

## Context
Many modern developer runtimes and consumer applications transmit telemetry, hardware identifiers, or usage statistics to external cloud endpoints. QERLIO's core mission is private, on-device local execution.

## Decision
1. **Network Disabled by Default:** The inference core operates in strict offline mode. No sockets or network connections are initialized during model loading, inference, or benchmarking.
2. **Zero Telemetry by Default:** Crash reporting, usage analytics, and hardware fingerprints are strictly disabled by default.
3. **Explicit Opt-in Only:** Telemetry or remote crash reporting can only be activated by an explicit, documented opt-in setting with transparent schema disclosure.
4. **Log Sanitization:** Default logs record operational milestones (e.g., model loaded, token count, timing metrics) but **never** record prompt text, generated tokens, or user conversation history.

## Consequences
- **Positive:** Uncompromising privacy guarantees; verifiable offline compliance for sensitive enterprise deployments.
- **Negative:** Development team receives zero automated feedback on real-world crashes unless users deliberately submit anonymized benchmark/diagnostic bundles.
