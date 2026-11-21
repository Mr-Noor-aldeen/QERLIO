#include "gguf_parser.hpp"
#include <fstream>
#include <sstream>
#include <iomanip>
#include <array>
#include <cstring>
#include <vector>

namespace qerlio {

constexpr uint64_t MAX_REASONABLE_TENSORS = 100000;
constexpr uint64_t MAX_REASONABLE_METADATA_KEYS = 50000;
constexpr size_t MAX_REASONABLE_STRING_LENGTH = 65536;
constexpr uint64_t MAX_REASONABLE_ARRAY_ELEMENTS = 1000000;

namespace {

enum GGUFValueType : uint32_t {
    GGUF_TYPE_UINT8   = 0,
    GGUF_TYPE_INT8    = 1,
    GGUF_TYPE_UINT16  = 2,
    GGUF_TYPE_INT16   = 3,
    GGUF_TYPE_UINT32  = 4,
    GGUF_TYPE_INT32   = 5,
    GGUF_TYPE_FLOAT32 = 6,
    GGUF_TYPE_BOOL    = 7,
    GGUF_TYPE_STRING  = 8,
    GGUF_TYPE_ARRAY   = 9,
    GGUF_TYPE_UINT64  = 10,
    GGUF_TYPE_INT64   = 11,
    GGUF_TYPE_FLOAT64 = 12
};

std::string file_type_to_quant_string(uint32_t ft) {
    switch (ft) {
        case 0:  return "F32";
        case 1:  return "F16";
        case 2:  return "Q4_0";
        case 3:  return "Q4_1";
        case 7:  return "Q8_0";
        case 8:  return "Q5_0";
        case 9:  return "Q5_1";
        case 10: return "Q2_K";
        case 11: return "Q3_K_S";
        case 12: return "Q3_K_M";
        case 13: return "Q3_K_L";
        case 14: return "Q4_K_S";
        case 15: return "Q4_K_M";
        case 16: return "Q5_K_S";
        case 17: return "Q5_K_M";
        case 18: return "Q6_K";
        case 32: return "BF16";
        default: return "";
    }
}

bool read_gguf_string(std::ifstream& file, std::string& out_str) {
    uint64_t len = 0;
    if (!file.read(reinterpret_cast<char*>(&len), sizeof(len))) {
        return false;
    }
    if (len > MAX_REASONABLE_STRING_LENGTH) {
        return false;
    }
    out_str.resize(static_cast<size_t>(len));
    if (len > 0 && !file.read(&out_str[0], static_cast<std::streamsize>(len))) {
        return false;
    }
    return true;
}

bool skip_gguf_value(std::ifstream& file, uint32_t value_type);

bool skip_gguf_array(std::ifstream& file) {
    uint32_t elem_type = 0;
    uint64_t count = 0;
    if (!file.read(reinterpret_cast<char*>(&elem_type), sizeof(elem_type))) {
        return false;
    }
    if (!file.read(reinterpret_cast<char*>(&count), sizeof(count))) {
        return false;
    }
    if (count > MAX_REASONABLE_ARRAY_ELEMENTS) {
        return false;
    }
    for (uint64_t i = 0; i < count; ++i) {
        if (!skip_gguf_value(file, elem_type)) {
            return false;
        }
    }
    return true;
}

bool skip_gguf_value(std::ifstream& file, uint32_t value_type) {
    switch (value_type) {
        case GGUF_TYPE_UINT8:
        case GGUF_TYPE_INT8:
        case GGUF_TYPE_BOOL:
            file.seekg(1, std::ios::cur);
            return file.good();
        case GGUF_TYPE_UINT16:
        case GGUF_TYPE_INT16:
            file.seekg(2, std::ios::cur);
            return file.good();
        case GGUF_TYPE_UINT32:
        case GGUF_TYPE_INT32:
        case GGUF_TYPE_FLOAT32:
            file.seekg(4, std::ios::cur);
            return file.good();
        case GGUF_TYPE_UINT64:
        case GGUF_TYPE_INT64:
        case GGUF_TYPE_FLOAT64:
            file.seekg(8, std::ios::cur);
            return file.good();
        case GGUF_TYPE_STRING: {
            std::string dummy;
            return read_gguf_string(file, dummy);
        }
        case GGUF_TYPE_ARRAY:
            return skip_gguf_array(file);
        default:
            return false;
    }
}

bool ends_with(const std::string& str, const std::string& suffix) {
    if (suffix.size() > str.size()) {
        return false;
    }
    return str.compare(str.size() - suffix.size(), suffix.size(), suffix) == 0;
}

} // namespace

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

    out_metadata.architecture.clear();
    out_metadata.quantization_type.clear();

    // Parse GGUF key-value metadata table if bytes follow the 24-byte header
    for (uint64_t i = 0; i < metadata_kv_count; ++i) {
        if (file.peek() == EOF) {
            break;
        }
        std::string key;
        if (!read_gguf_string(file, key)) {
            break;
        }
        uint32_t value_type = 0;
        if (!file.read(reinterpret_cast<char*>(&value_type), sizeof(value_type))) {
            break;
        }

        if (value_type == GGUF_TYPE_STRING) {
            std::string val;
            if (!read_gguf_string(file, val)) {
                break;
            }
            if (key == "general.architecture") {
                out_metadata.architecture = val;
            } else if (key == "general.name") {
                out_metadata.name = val;
            } else if (key == "general.quantization_type") {
                out_metadata.quantization_type = val;
            }
        } else if (value_type == GGUF_TYPE_UINT32 || value_type == GGUF_TYPE_INT32 || value_type == GGUF_TYPE_UINT64) {
            uint64_t num_val = 0;
            if (value_type == GGUF_TYPE_UINT64) {
                if (!file.read(reinterpret_cast<char*>(&num_val), sizeof(uint64_t))) {
                    break;
                }
            } else {
                uint32_t v32 = 0;
                if (!file.read(reinterpret_cast<char*>(&v32), sizeof(uint32_t))) {
                    break;
                }
                num_val = v32;
            }

            if (key == "general.file_type" && out_metadata.quantization_type.empty()) {
                out_metadata.quantization_type = file_type_to_quant_string(static_cast<uint32_t>(num_val));
            } else if (ends_with(key, ".context_length")) {
                out_metadata.context_length = static_cast<uint32_t>(num_val);
            } else if (ends_with(key, ".embedding_length")) {
                out_metadata.embedding_length = static_cast<uint32_t>(num_val);
            } else if (ends_with(key, ".block_count")) {
                out_metadata.block_count = static_cast<uint32_t>(num_val);
            } else if (ends_with(key, ".feed_forward_length")) {
                out_metadata.feed_forward_length = static_cast<uint32_t>(num_val);
            }
        } else {
            if (!skip_gguf_value(file, value_type)) {
                break;
            }
        }
    }

    return QERLIO_OK;
}

namespace {

constexpr std::array<uint32_t, 64> SHA256_K = {
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2
};

inline uint32_t rotr(uint32_t x, uint32_t n) {
    return (x >> n) | (x << (32 - n));
}

void sha256_transform(uint32_t state[8], const uint8_t block[64]) {
    uint32_t w[64];
    for (int i = 0; i < 16; ++i) {
        w[i] = (static_cast<uint32_t>(block[i * 4]) << 24) |
               (static_cast<uint32_t>(block[i * 4 + 1]) << 16) |
               (static_cast<uint32_t>(block[i * 4 + 2]) << 8) |
               (static_cast<uint32_t>(block[i * 4 + 3]));
    }
    for (int i = 16; i < 64; ++i) {
        uint32_t s0 = rotr(w[i - 15], 7) ^ rotr(w[i - 15], 18) ^ (w[i - 15] >> 3);
        uint32_t s1 = rotr(w[i - 2], 17) ^ rotr(w[i - 2], 19) ^ (w[i - 2] >> 10);
        w[i] = w[i - 16] + s0 + w[i - 7] + s1;
    }

    uint32_t a = state[0], b = state[1], c = state[2], d = state[3];
    uint32_t e = state[4], f = state[5], g = state[6], h = state[7];

    for (int i = 0; i < 64; ++i) {
        uint32_t S1 = rotr(e, 6) ^ rotr(e, 11) ^ rotr(e, 25);
        uint32_t ch = (e & f) ^ ((~e) & g);
        uint32_t temp1 = h + S1 + ch + SHA256_K[i] + w[i];
        uint32_t S0 = rotr(a, 2) ^ rotr(a, 13) ^ rotr(a, 22);
        uint32_t maj = (a & b) ^ (a & c) ^ (b & c);
        uint32_t temp2 = S0 + maj;

        h = g; g = f; f = e; e = d + temp1;
        d = c; c = b; b = a; a = temp1 + temp2;
    }

    state[0] += a; state[1] += b; state[2] += c; state[3] += d;
    state[4] += e; state[5] += f; state[6] += g; state[7] += h;
}

} // namespace

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

    uint32_t state[8] = {
        0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
        0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19
    };

    uint64_t total_bits = 0;
    uint8_t buffer[64];
    while (file.read(reinterpret_cast<char*>(buffer), 64)) {
        sha256_transform(state, buffer);
        total_bits += 512;
    }

    std::streamsize rem = file.gcount();
    total_bits += static_cast<uint64_t>(rem) * 8;
    std::memset(buffer + rem, 0, 64 - static_cast<size_t>(rem));
    buffer[rem] = 0x80;

    if (rem >= 56) {
        sha256_transform(state, buffer);
        std::memset(buffer, 0, 64);
    }

    for (int i = 0; i < 8; ++i) {
        buffer[63 - i] = static_cast<uint8_t>((total_bits >> (i * 8)) & 0xFF);
    }
    sha256_transform(state, buffer);

    std::ostringstream oss;
    oss << std::hex << std::setfill('0');
    for (int i = 0; i < 8; ++i) {
        oss << std::setw(8) << state[i];
    }
    out_sha256 = oss.str();
    return QERLIO_OK;
}

} // namespace qerlio
