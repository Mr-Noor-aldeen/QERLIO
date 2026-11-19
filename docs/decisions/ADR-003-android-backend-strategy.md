# ADR-003: Android Acceleration and NPU Strategy

## Status
Accepted

## Context
Mobile edge deployment on Android is the primary tier for QERLIO. Historically, Android developers relied on NNAPI (Neural Networks API) for hardware acceleration. However, Google officially deprecated NNAPI starting in Android 15. Furthermore, mobile NPUs (Qualcomm NPU via QNN, MediaTek via NeuroPilot, Samsung Exynos) have fragmented vendor-specific SDKs and operator limitations.

## Decision
1. **NNAPI Rejection:** QERLIO will **not** rely on NNAPI as its primary mobile acceleration strategy.
2. **CPU First-Class Baseline:** ARM64-v8a and ARM64-v9a CPU execution with optimized NEON and dot-product instructions is the mandatory, guaranteed fallback on all Android devices.
3. **GPU Pathway:** Vulkan is adopted as the primary portable mobile GPU acceleration pathway.
4. **Vendor NPU Adapters:** Specific NPU SDKs (e.g., Qualcomm QNN, LiteRT/ExecuTorch NPU delegates) will be introduced as modular, opt-in backend adapters with capability detection. An NPU backend is never advertised as supported unless proven on real silicon.
5. **Memory & Thermal Awareness:** Android execution will subscribe to `ComponentCallbacks2.onTrimMemory()` and Android Thermal APIs (`PowerManager.addThermalStatusListener`) to throttle or pause workloads before system termination.

## Consequences
- **Positive:** Immune to NNAPI deprecation; stable fallback prevents crashes; respects device thermals and battery life.
- **Negative:** Accessing peak NPU capabilities requires individual vendor SDK integrations rather than a single universal Android API.
