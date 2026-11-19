#ifndef QERLIO_SESSION_INTERNAL_HPP
#define QERLIO_SESSION_INTERNAL_HPP

#include <atomic>
#include <string>
#include "qerlio/qerlio.h"
#include "core/model_registry/model_internal.hpp"

struct qerlio_session {
    qerlio_model* parent_model{nullptr};
    std::atomic<bool> is_cancelled{false};
    std::atomic<bool> is_generating{false};

    void cancel() {
        is_cancelled.store(true, std::memory_order_release);
    }

    void reset_cancellation() {
        is_cancelled.store(false, std::memory_order_release);
    }

    bool should_stop() const {
        return is_cancelled.load(std::memory_order_acquire);
    }
};

#endif // QERLIO_SESSION_INTERNAL_HPP
