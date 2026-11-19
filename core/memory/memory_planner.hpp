#ifndef QERLIO_MEMORY_PLANNER_HPP
#define QERLIO_MEMORY_PLANNER_HPP

#include <cstddef>
#include <cstdint>
#include <string>

namespace qerlio {

struct ModelArchitectureDimensions {
    uint32_t layers = 32;
    uint32_t kv_heads = 8;        // Grouped-Query Attention (GQA) heads
    uint32_t head_dim = 128;
    uint32_t bytes_per_element = 2; // Default F16 = 2 bytes, Q8 = 1 byte
};

struct MemoryBudgetResult {
    size_t weight_bytes = 0;
    size_t kv_cache_bytes = 0;
    size_t workspace_bytes = 0;
    size_t system_safety_margin_bytes = 0;
    size_t total_estimated_bytes = 0;
    bool fits_in_available_ram = false;
    uint32_t recommended_context_length = 2048;
    std::string diagnostic_message;
};

class MemoryPlanner {
public:
    static size_t calculate_kv_cache_bytes(
        const ModelArchitectureDimensions& dims,
        uint32_t context_length
    );

    static MemoryBudgetResult plan_memory(
        size_t model_file_size,
        const ModelArchitectureDimensions& dims,
        uint32_t requested_context,
        size_t available_system_ram_bytes
    );
};

} // namespace qerlio

#endif // QERLIO_MEMORY_PLANNER_HPP
