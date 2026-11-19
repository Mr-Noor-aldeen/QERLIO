#ifndef QERLIO_ZAQERIS_COMPRESSION_ADAPTER_HPP
#define QERLIO_ZAQERIS_COMPRESSION_ADAPTER_HPP

#include <string>
#include <vector>
#include <cstdint>
#include <cstddef>
#include "qerlio/qerlio.h"

namespace qerlio {

struct ZaqerisCompressionHeader {
    uint32_t magic = 0x334D515A; // 'ZQM3'
    uint32_t version = 1;
    uint32_t in_features = 0;
    uint32_t out_features = 0;
    uint32_t rvq_stages = 4;
    uint32_t block_size = 8;
    float scale = 1.0f;
};

struct DecompressionStats {
    uint64_t decompress_time_ns = 0;
    size_t uncompressed_bytes = 0;
    size_t compressed_bytes = 0;
    double compression_ratio = 0.0;
};

class ZaqerisM3Adapter {
public:
    static bool is_zaqeris_tensor(const std::string& tensor_name) {
        return tensor_name.find(".m3.") != std::string::npos;
    }

    static qerlio_status_t validate_header(
        const ZaqerisCompressionHeader& header,
        std::string& out_error
    ) {
        if (header.magic != 0x334D515A) {
            out_error = "Invalid ZAQERIS M3 magic header";
            return QERLIO_ERR_UNSUPPORTED_QUANTIZATION;
        }
        if (header.in_features == 0 || header.out_features == 0) {
            out_error = "Zero dimension in ZAQERIS M3 tensor";
            return QERLIO_ERR_MODEL_CORRUPT;
        }
        return QERLIO_OK;
    }
};

} // namespace qerlio

#endif // QERLIO_ZAQERIS_COMPRESSION_ADAPTER_HPP
