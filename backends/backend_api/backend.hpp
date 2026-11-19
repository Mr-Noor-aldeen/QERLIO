#ifndef QERLIO_BACKEND_API_HPP
#define QERLIO_BACKEND_API_HPP

#include <string>
#include <vector>
#include <memory>
#include <cstdint>
#include "qerlio/qerlio.h"

namespace qerlio {

struct BackendCapabilities {
    std::string name;
    std::string version;
    bool supports_cpu{true};
    bool supports_gpu{false};
    bool supports_npu{false};
    bool supports_quantized_kv{false};
    bool supports_speculative_decoding{false};
    uint32_t max_context_tested{4096};
    std::vector<std::string> supported_quantizations;
    std::vector<std::string> supported_architectures;
};

struct BackendSessionContext {
    uint32_t context_length{2048};
    uint32_t threads{4};
    int32_t gpu_layers{0};
    size_t kv_cache_allocated_bytes{0};
};

class IBackend {
public:
    virtual ~IBackend() = default;

    virtual qerlio_status_t initialize(const qerlio_engine_config_t& config) = 0;
    virtual void shutdown() = 0;

    virtual BackendCapabilities get_capabilities() const = 0;

    virtual qerlio_status_t load_model(
        const std::string& model_path,
        const qerlio_model_config_t& config,
        void** out_backend_model_handle
    ) = 0;

    virtual void unload_model(void* backend_model_handle) = 0;

    virtual qerlio_status_t create_session(
        void* backend_model_handle,
        BackendSessionContext& session_ctx,
        void** out_backend_session_handle
    ) = 0;

    virtual void destroy_session(void* backend_session_handle) = 0;

    virtual qerlio_status_t eval_prompt(
        void* backend_session_handle,
        const std::string& prompt,
        uint32_t& out_prompt_tokens
    ) = 0;

    virtual qerlio_status_t sample_next_token(
        void* backend_session_handle,
        const qerlio_generation_config_t& config,
        std::string& out_token_piece,
        bool& out_is_eos
    ) = 0;
};

} // namespace qerlio

#endif // QERLIO_BACKEND_API_HPP
