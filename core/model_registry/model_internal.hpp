#ifndef QERLIO_MODEL_INTERNAL_HPP
#define QERLIO_MODEL_INTERNAL_HPP

#include <string>
#include <memory>
#include "qerlio/qerlio.h"
#include "core/model_registry/gguf_parser.hpp"
#include "core/engine/engine_internal.hpp"

struct qerlio_model {
    qerlio_engine* parent_engine{nullptr};
    std::string model_path;
    std::string sha256_hash;
    qerlio_model_config_t config{};
    qerlio::GGUFHeader header{};
    qerlio::GGUFMetadata metadata{};
};

#endif // QERLIO_MODEL_INTERNAL_HPP
