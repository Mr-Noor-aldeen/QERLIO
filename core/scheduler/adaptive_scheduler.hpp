#ifndef QERLIO_ADAPTIVE_SCHEDULER_HPP
#define QERLIO_ADAPTIVE_SCHEDULER_HPP

#include <cstdint>
#include <string>
#include "qerlio/qerlio.h"

namespace qerlio {

enum class ThermalState {
    Normal = 0,
    Warm = 1,
    Hot = 2,
    Critical = 3
};

struct SchedulerPolicy {
    uint32_t active_threads{4};
    uint32_t batch_size{512};
    uint32_t micro_batch_size{32};
    bool allow_generation{true};
    bool throttle_active{false};
    std::string decision_reason;
};

class AdaptiveScheduler {
public:
    static SchedulerPolicy compute_policy(
        uint32_t physical_cores,
        qerlio_power_profile_t power_profile,
        ThermalState thermal_state
    );
};

} // namespace qerlio

#endif // QERLIO_ADAPTIVE_SCHEDULER_HPP
