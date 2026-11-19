#include "adaptive_scheduler.hpp"
#include <algorithm>

namespace qerlio {

SchedulerPolicy AdaptiveScheduler::compute_policy(
    uint32_t physical_cores,
    qerlio_power_profile_t power_profile,
    ThermalState thermal_state
) {
    SchedulerPolicy policy;
    uint32_t safe_cores = std::max(1u, physical_cores);

    if (thermal_state == ThermalState::Critical) {
        policy.allow_generation = false;
        policy.throttle_active = true;
        policy.active_threads = 1;
        policy.decision_reason = "CRITICAL: System thermal limit exceeded. Generation suspended.";
        return policy;
    }

    switch (power_profile) {
        case QERLIO_POWER_SAVER:
            policy.active_threads = std::max(1u, safe_cores / 2);
            policy.batch_size = 256;
            policy.decision_reason = "Power Saver profile: threads reduced to 50% of physical cores.";
            break;

        case QERLIO_POWER_PERFORMANCE:
            policy.active_threads = safe_cores;
            policy.batch_size = 512;
            policy.decision_reason = "Performance profile: utilizing 100% of physical cores.";
            break;

        case QERLIO_POWER_BALANCED:
        default:
            policy.active_threads = std::clamp(safe_cores, 2u, 6u);
            policy.batch_size = 512;
            policy.decision_reason = "Balanced profile: bounded threads to maintain responsive UI.";
            break;
    }

    if (thermal_state == ThermalState::Warm) {
        policy.active_threads = std::max(1u, (policy.active_threads * 3) / 4);
        policy.throttle_active = true;
        policy.decision_reason += " (Throttled: Warm thermal state detected)";
    } else if (thermal_state == ThermalState::Hot) {
        policy.active_threads = std::max(1u, policy.active_threads / 2);
        policy.batch_size = 128;
        policy.throttle_active = true;
        policy.decision_reason += " (Aggressive Throttling: Hot thermal state detected)";
    }

    return policy;
}

}
