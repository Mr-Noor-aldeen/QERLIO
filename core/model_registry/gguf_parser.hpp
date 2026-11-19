#ifndef QERLIO_GGUF_PARSER_HPP
#define QERLIO_GGUF_PARSER_HPP

#include <string>
#include <vector>
#include <cstdint>
#include <cstddef>
#include "qerlio/qerlio.h"

namespace qerlio {

struct GGUFHeader {
    uint32_t magic = 0;             // Must be 0x46554747 ('GGUF')
    uint32_t version = 0;           // GGUF specification version (2 or 3)
    uint64_t tensor_count = 0;
    uint64_t metadata_kv_count = 0;
};

struct GGUFMetadata {
    std::string architecture;
    std::string name;
    uint32_t context_length = 0;
    uint32_t embedding_length = 0;
    uint32_t block_count = 0;
    uint32_t feed_forward_length = 0;
    std::string quantization_type;
    uint64_t total_file_size = 0;
};

class GGUFParser {
public:
    static qerlio_status_t parse_header(
        const std::string& file_path,
        GGUFHeader& out_header,
        GGUFMetadata& out_metadata,
        std::string& out_error
    );

    static qerlio_status_t compute_sha256(
        const std::string& file_path,
        std::string& out_sha256,
        std::string& out_error
    );

    static bool is_magic_valid(uint32_t magic) {
        // 'G' = 0x47, 'G' = 0x47, 'U' = 0x55, 'F' = 0x46
        return magic == 0x46554747;
    }
};

} // namespace qerlio

#endif // QERLIO_GGUF_PARSER_HPP
