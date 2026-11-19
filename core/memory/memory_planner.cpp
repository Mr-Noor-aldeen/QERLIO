#include "memory_planner.hpp"
#include <algorithm>

namespace qerlio {

constexpr size_t DEFAULT_WORKSPACE_BYTES = 128 * 1024 * 1024;
constexpr size_t MIN_SAFETY_MARGIN_BYTES = 512 * 1024 * 1024;

size_t MemoryPlanner::calculate_kv_cache_bytes(
    const ModelArchitectureDimensions& dims,
    uint32_t context_length
) {
    uint64_t total = 2ULL * dims.layers * dims.kv_heads * dims.head_dim * context_length * dims.bytes_per_element;
    return static_cast<size_t>(total);
}

MemoryBudgetResult MemoryPlanner::plan_memory(
    size_t model_file_size,
    const ModelArchitectureDimensions& dims,
    uint32_t requested_context,
    size_t available_system_ram_bytes
) {
    MemoryBudgetResult result;
    result.weight_bytes = model_file_size;
    result.workspace_bytes = DEFAULT_WORKSPACE_BYTES;
    result.system_safety_margin_bytes = MIN_SAFETY_MARGIN_BYTES;

    size_t kv_bytes = calculate_kv_cache_bytes(dims, requested_context);
    size_t total = result.weight_bytes + kv_bytes + result.workspace_bytes + result.system_safety_margin_bytes;

    if (total <= available_system_ram_bytes) {
        result.kv_cache_bytes = kv_bytes;
        result.total_estimated_bytes = total;
        result.fits_in_available_ram = true;
        result.recommended_context_length = requested_context;
        result.diagnostic_message = "Requested context fits within safe memory budget.";
        return result;
    }

    const uint32_t fallback_contexts[] = {2048, 1024, 512};
    for (uint32_t fallback_ctx : fallback_contexts) {
        if (fallback_ctx >= requested_context) {
            continue;
        }
        kv_bytes = calculate_kv_cache_bytes(dims, fallback_ctx);
        total = result.weight_bytes + kv_bytes + result.workspace_bytes + result.system_safety_margin_bytes;
        if (total <= available_system_ram_bytes) {
            result.kv_cache_bytes = kv_bytes;
            result.total_estimated_bytes = total;
            result.fits_in_available_ram = true;
            result.recommended_context_length = fallback_ctx;
            result.diagnostic_message = "Context auto-adjusted to avoid OOM.";
            return result;
        }
    }

    result.kv_cache_bytes = kv_bytes;
    result.total_estimated_bytes = total;
    result.fits_in_available_ram = false;
    result.recommended_context_length = 0;
    result.diagnostic_message = "Insufficient memory: model weights and workspace exceed available safe budget.";
    return result;
}

}
