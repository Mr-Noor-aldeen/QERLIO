#include "qerlio/qerlio.h"
#include "core/engine/engine_internal.hpp"
#include "core/model_registry/model_internal.hpp"
#include "core/session/session_internal.hpp"
#include "core/model_registry/gguf_parser.hpp"

#include <cstring>
#include <chrono>
#include <thread>

static const char* QERLIO_VERSION_STRING = "0.3.0";

const char* qerlio_get_version(void) {
    return QERLIO_VERSION_STRING;
}

qerlio_status_t qerlio_engine_create(
    const qerlio_engine_config_t* config,
    qerlio_engine_t** out_engine
) {
    if (!out_engine) {
        return QERLIO_ERR_INVALID_ARGUMENT;
    }

    try {
        auto* engine = new qerlio_engine();
        if (config) {
            engine->config = *config;
        } else {
            engine->config.max_concurrent_sessions = 4;
            engine->config.enable_telemetry = false;
            engine->config.log_directory = nullptr;
        }
        engine->is_initialized = true;
        *out_engine = engine;
        return QERLIO_OK;
    } catch (...) {
        return QERLIO_ERR_INTERNAL;
    }
}

void qerlio_engine_destroy(qerlio_engine_t* engine) {
    if (engine) {
        delete engine;
    }
}

qerlio_status_t qerlio_get_last_error(
    qerlio_engine_t* engine,
    char* buffer,
    size_t buffer_size
) {
    if (!engine || !buffer || buffer_size == 0) {
        return QERLIO_ERR_INVALID_ARGUMENT;
    }

    std::string err = engine->get_error();
    std::strncpy(buffer, err.c_str(), buffer_size - 1);
    buffer[buffer_size - 1] = '\0';
    return QERLIO_OK;
}

qerlio_status_t qerlio_model_load(
    qerlio_engine_t* engine,
    const qerlio_model_config_t* config,
    qerlio_model_t** out_model
) {
    if (!engine || !config || !config->model_path || !out_model) {
        return QERLIO_ERR_INVALID_ARGUMENT;
    }

    try {
        auto* model = new qerlio_model();
        model->parent_engine = engine;
        model->model_path = config->model_path;
        model->config = *config;

        std::string parse_err;
        qerlio_status_t status = qerlio::GGUFParser::parse_header(
            model->model_path,
            model->header,
            model->metadata,
            parse_err
        );

        if (status != QERLIO_OK) {
            engine->set_error(parse_err);
            delete model;
            return status;
        }

        *out_model = model;
        return QERLIO_OK;
    } catch (...) {
        return QERLIO_ERR_INTERNAL;
    }
}

void qerlio_model_destroy(qerlio_model_t* model) {
    if (model) {
        delete model;
    }
}

qerlio_status_t qerlio_session_create(
    qerlio_model_t* model,
    qerlio_session_t** out_session
) {
    if (!model || !out_session) {
        return QERLIO_ERR_INVALID_ARGUMENT;
    }

    try {
        auto* session = new qerlio_session();
        session->parent_model = model;
        *out_session = session;
        return QERLIO_OK;
    } catch (...) {
        return QERLIO_ERR_INTERNAL;
    }
}

void qerlio_session_destroy(qerlio_session_t* session) {
    if (session) {
        session->cancel();
        delete session;
    }
}

qerlio_status_t qerlio_session_cancel(qerlio_session_t* session) {
    if (!session) {
        return QERLIO_ERR_INVALID_ARGUMENT;
    }
    session->cancel();
    return QERLIO_OK;
}

qerlio_status_t qerlio_generate_stream(
    qerlio_session_t* session,
    const char* prompt,
    const qerlio_generation_config_t* config,
    qerlio_token_callback_t callback,
    void* user_data,
    qerlio_generation_metrics_t* out_metrics
) {
    if (!session || !prompt || !callback) {
        return QERLIO_ERR_INVALID_ARGUMENT;
    }

    session->reset_cancellation();
    session->is_generating.store(true, std::memory_order_release);

    auto start_time = std::chrono::steady_clock::now();
    uint32_t max_tokens = (config && config->max_output_tokens > 0) ? config->max_output_tokens : 128;

    uint32_t generated = 0;
    qerlio_status_t final_status = QERLIO_OK;

    for (uint32_t i = 0; i < max_tokens; ++i) {
        if (session->should_stop()) {
            final_status = QERLIO_ERR_CANCELLED;
            break;
        }

        const char* chunk = " ";
        bool keep_going = callback(chunk, user_data);
        generated++;

        if (!keep_going) {
            final_status = QERLIO_ERR_CANCELLED;
            break;
        }
    }

    auto end_time = std::chrono::steady_clock::now();
    double duration_sec = std::chrono::duration<double>(end_time - start_time).count();
    (void)duration_sec;

    if (out_metrics) {
        out_metrics->tokens_generated = generated;
        out_metrics->prompt_tokens_evaluated = 0;
        out_metrics->prompt_tokens_per_sec = 0.0;
        out_metrics->generation_tokens_per_sec = 0.0;
        out_metrics->time_to_first_token_ms = 0;
        out_metrics->peak_memory_bytes = 0;
        out_metrics->kv_cache_bytes = 0;
        out_metrics->active_backend_name = "qerlio-reference-cpu-stub";
    }

    session->is_generating.store(false, std::memory_order_release);
    return final_status;
}
