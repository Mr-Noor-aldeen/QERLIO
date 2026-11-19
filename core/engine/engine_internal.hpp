#ifndef QERLIO_ENGINE_INTERNAL_HPP
#define QERLIO_ENGINE_INTERNAL_HPP

#include <string>
#include <mutex>
#include <vector>
#include <memory>
#include "qerlio/qerlio.h"

struct qerlio_engine {
    qerlio_engine_config_t config{};
    std::mutex engine_mutex;
    std::string last_error;
    bool is_initialized{false};

    void set_error(const std::string& err) {
        std::lock_guard<std::mutex> lock(engine_mutex);
        last_error = err;
    }

    std::string get_error() {
        std::lock_guard<std::mutex> lock(engine_mutex);
        return last_error;
    }
};

#endif // QERLIO_ENGINE_INTERNAL_HPP
