# Security Policy

## Reporting Security Issues

The QERLIO team takes the security of local AI runtime execution seriously. If you discover a vulnerability, parser flaw, memory safety issue, or security concern, please report it responsibly.

### Disclosure Process
- Contact the maintainers via security@zaqeris.ai or submit a private advisory on GitHub.
- Do not disclose security issues publicly until a patch or mitigation has been reviewed and released.
- Provide reproducible steps, sample malformed inputs (if applicable), and system details.

### Security Guarantees in QERLIO
1. **Offline & Zero Network:** The inference engine does not initiate outbound network requests during model loading or generation.
2. **Untrusted Model Policy:** Model files are treated as untrusted binary inputs. The GGUF parser validates headers and tensor table bounds to mitigate buffer overruns.
3. **Local Loopback API:** The local server binds to `127.0.0.1` by default and enforces origin and token validation.
