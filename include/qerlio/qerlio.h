#ifndef QERLIO_H
#define QERLIO_H

#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

#if defined(_WIN32) || defined(__CYGWIN__)
  #if defined(QERLIO_BUILD_SHARED)
    #define QERLIO_API __declspec(dllexport)
  #elif defined(QERLIO_USE_SHARED)
    #define QERLIO_API __declspec(dllimport)
  #else
    #define QERLIO_API
  #endif
#else
  #if defined(__GNUC__) && __GNUC__ >= 4
    #define QERLIO_API __attribute__((visibility("default")))
  #else
    #define QERLIO_API
  #endif
#endif

typedef struct qerlio_engine qerlio_engine_t;
typedef struct qerlio_model qerlio_model_t;
typedef struct qerlio_session qerlio_session_t;
typedef struct qerlio_cancellation_token qerlio_cancellation_token_t;

typedef enum {
    QERLIO_OK = 0,
    QERLIO_ERR_INVALID_ARGUMENT = 1,
    QERLIO_ERR_UNSUPPORTED_PLATFORM = 2,
    QERLIO_ERR_UNSUPPORTED_ARCHITECTURE = 3,
    QERLIO_ERR_UNSUPPORTED_QUANTIZATION = 4,
    QERLIO_ERR_MODEL_CORRUPT = 5,
    QERLIO_ERR_MODEL_HASH_MISMATCH = 6,
    QERLIO_ERR_INSUFFICIENT_MEMORY = 7,
    QERLIO_ERR_BACKEND_INIT_FAILED = 8,
    QERLIO_ERR_OPERATOR_UNSUPPORTED = 9,
    QERLIO_ERR_CANCELLED = 10,
    QERLIO_ERR_TIMEOUT = 11,
    QERLIO_ERR_PERMISSION_DENIED = 12,
    QERLIO_ERR_NETWORK_DISABLED = 13,
    QERLIO_ERR_IO = 14,
    QERLIO_ERR_INTERNAL = 15
} qerlio_status_t;

typedef enum {
    QERLIO_BACKEND_AUTO = 0,
    QERLIO_BACKEND_CPU = 1,
    QERLIO_BACKEND_GPU = 2,
    QERLIO_BACKEND_NPU = 3
} qerlio_backend_preference_t;

typedef enum {
    QERLIO_POWER_BALANCED = 0,
    QERLIO_POWER_SAVER = 1,
    QERLIO_POWER_PERFORMANCE = 2
} qerlio_power_profile_t;

typedef struct {
    uint32_t max_concurrent_sessions;
    bool enable_telemetry;
    const char* log_directory;
} qerlio_engine_config_t;

typedef struct {
    const char* model_path;
    const char* expected_sha256;
    uint32_t context_length;
    int32_t gpu_layers;
    qerlio_backend_preference_t backend_preference;
} qerlio_model_config_t;

typedef struct {
    uint32_t max_output_tokens;
    float temperature;
    float top_p;
    uint32_t top_k;
    float repeat_penalty;
    uint64_t seed;
    qerlio_power_profile_t power_profile;
} qerlio_generation_config_t;

typedef struct {
    uint64_t time_to_first_token_ms;
    double prompt_tokens_per_sec;
    double generation_tokens_per_sec;
    uint32_t prompt_tokens_evaluated;
    uint32_t tokens_generated;
    size_t peak_memory_bytes;
    size_t kv_cache_bytes;
    const char* active_backend_name;
} qerlio_generation_metrics_t;

typedef bool (*qerlio_token_callback_t)(
    const char* token_chunk,
    void* user_data
);

QERLIO_API qerlio_status_t qerlio_engine_create(
    const qerlio_engine_config_t* config,
    qerlio_engine_t** out_engine
);

QERLIO_API void qerlio_engine_destroy(
    qerlio_engine_t* engine
);

QERLIO_API const char* qerlio_get_version(void);

QERLIO_API qerlio_status_t qerlio_get_last_error(
    qerlio_engine_t* engine,
    char* buffer,
    size_t buffer_size
);

QERLIO_API qerlio_status_t qerlio_model_load(
    qerlio_engine_t* engine,
    const qerlio_model_config_t* config,
    qerlio_model_t** out_model
);

QERLIO_API void qerlio_model_destroy(
    qerlio_model_t* model
);

QERLIO_API qerlio_status_t qerlio_session_create(
    qerlio_model_t* model,
    qerlio_session_t** out_session
);

QERLIO_API void qerlio_session_destroy(
    qerlio_session_t* session
);

QERLIO_API qerlio_status_t qerlio_generate_stream(
    qerlio_session_t* session,
    const char* prompt,
    const qerlio_generation_config_t* config,
    qerlio_token_callback_t callback,
    void* user_data,
    qerlio_generation_metrics_t* out_metrics
);

QERLIO_API qerlio_status_t qerlio_session_cancel(
    qerlio_session_t* session
);

#ifdef __cplusplus
}
#endif

#endif
