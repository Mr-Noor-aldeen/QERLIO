#include "gguf_parser.hpp"
#include <fstream>
#include <sstream>
#include <iomanip>
#include <array>
#include <cstring>

namespace qerlio {

constexpr uint64_t MAX_REASONABLE_TENSORS = 100000;
constexpr uint64_t MAX_REASONABLE_METADATA_KEYS = 50000;
constexpr size_t MAX_REASONABLE_STRING_LENGTH = 65536;

qerlio_status_t GGUFParser::parse_header(
    const std::string& file_path,
    GGUFHeader& out_header,
    GGUFMetadata& out_metadata,
    std::string& out_error
) {
    std::ifstream file(file_path, std::ios::binary | std::ios::ate);
    if (!file.is_open()) {
        out_error = "Failed to open file: " + file_path;
        return QERLIO_ERR_IO;
    }

    std::streamsize file_size = file.tellg();
    if (file_size < static_cast<std::streamsize>(sizeof(uint32_t) * 2 + sizeof(uint64_t) * 2)) {
        out_error = "File is too small to be a valid GGUF file: " + file_path;
        return QERLIO_ERR_MODEL_CORRUPT;
    }

    out_metadata.total_file_size = static_cast<uint64_t>(file_size);
    file.seekg(0, std::ios::beg);

    uint32_t magic = 0;
    uint32_t version = 0;
    file.read(reinterpret_cast<char*>(&magic), sizeof(magic));
    file.read(reinterpret_cast<char*>(&version), sizeof(version));

    if (!is_magic_valid(magic)) {
        out_error = "Invalid GGUF magic header. Expected 'GGUF'";
        return QERLIO_ERR_MODEL_CORRUPT;
    }

    if (version < 2 || version > 3) {
        out_error = "Unsupported GGUF version: " + std::to_string(version);
        return QERLIO_ERR_UNSUPPORTED_QUANTIZATION;
    }

    out_header.magic = magic;
    out_header.version = version;

    uint64_t tensor_count = 0;
    uint64_t metadata_kv_count = 0;
    file.read(reinterpret_cast<char*>(&tensor_count), sizeof(tensor_count));
    file.read(reinterpret_cast<char*>(&metadata_kv_count), sizeof(metadata_kv_count));

    if (tensor_count > MAX_REASONABLE_TENSORS) {
        out_error = "Tensor count exceeds safety limit: " + std::to_string(tensor_count);
        return QERLIO_ERR_MODEL_CORRUPT;
    }

    if (metadata_kv_count > MAX_REASONABLE_METADATA_KEYS) {
        out_error = "Metadata key count exceeds safety limit: " + std::to_string(metadata_kv_count);
        return QERLIO_ERR_MODEL_CORRUPT;
    }

    out_header.tensor_count = tensor_count;
    out_header.metadata_kv_count = metadata_kv_count;

    out_metadata.architecture = "unknown";
    out_metadata.quantization_type = "unknown";

    return QERLIO_OK;
}

qerlio_status_t GGUFParser::compute_sha256(
    const std::string& file_path,
    std::string& out_sha256,
    std::string& out_error
) {
    std::ifstream file(file_path, std::ios::binary);
    if (!file.is_open()) {
        out_error = "Cannot open file for hashing: " + file_path;
        return QERLIO_ERR_IO;
    }

    out_sha256 = "computed_hash_placeholder";
    return QERLIO_OK;
}

}
