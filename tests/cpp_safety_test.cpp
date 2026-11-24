#include "qerlio/qerlio.h"
#include "qerlio/native_gemm.h"
#include "core/compression/universal_quant.hpp"
#include "core/memory/memory_planner.hpp"
#include "core/model_registry/gguf_parser.hpp"
#include "core/scheduler/adaptive_scheduler.hpp"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#define QERLIO_CHECK(cond)                                                        \
    do {                                                                          \
        if (!(cond)) {                                                            \
            std::cerr << "[FAIL] " << __FILE__ << ":" << __LINE__                 \
                      << " failed: " << #cond << "\n";                            \
            std::exit(1);                                                         \
        }                                                                         \
    } while (false)

namespace {

void write_u32(std::vector<uint8_t>& buf, uint32_t v) {
    for (int i = 0; i < 4; ++i) {
        buf.push_back(static_cast<uint8_t>((v >> (i * 8)) & 0xFF));
    }
}

void write_u64(std::vector<uint8_t>& buf, uint64_t v) {
    for (int i = 0; i < 8; ++i) {
        buf.push_back(static_cast<uint8_t>((v >> (i * 8)) & 0xFF));
    }
}

void write_str(std::vector<uint8_t>& buf, const std::string& s) {
    write_u64(buf, static_cast<uint64_t>(s.size()));
    buf.insert(buf.end(), s.begin(), s.end());
}

std::vector<uint8_t> build_synthetic_seed_oss_gguf() {
    std::vector<uint8_t> buf;
    // "GGUF" little-endian
    buf.push_back('G');
    buf.push_back('G');
    buf.push_back('U');
    buf.push_back('F');
    write_u32(buf, 3);  // version = 3
    write_u64(buf, 0);  // tensor_count = 0
    write_u64(buf, 7);  // metadata_kv_count = 7

    auto add_str_kv = [&](const std::string& k, const std::string& v) {
        write_str(buf, k);
        write_u32(buf, 8);  // GGUF_TYPE_STRING
        write_str(buf, v);
    };
    auto add_u32_kv = [&](const std::string& k, uint32_t v) {
        write_str(buf, k);
        write_u32(buf, 4);  // GGUF_TYPE_UINT32
        write_u32(buf, v);
    };

    add_str_kv("general.architecture", "seed_oss");
    add_str_kv("general.name", "Hermes 4_3_36b");
    add_u32_kv("seed_oss.context_length", 524288);
    add_u32_kv("seed_oss.embedding_length", 5120);
    add_u32_kv("seed_oss.block_count", 64);
    add_u32_kv("seed_oss.feed_forward_length", 27648);
    add_u32_kv("general.file_type", 15);  // Q4_K_M

    return buf;
}

bool stream_stop_after_two(const char* token_chunk, void* user_data) {
    if (!token_chunk || !user_data) {
        return false;
    }
    int* count = static_cast<int*>(user_data);
    (*count)++;
    return (*count) < 2;
}

bool stream_collect_all(const char* token_chunk, void* user_data) {
    if (!token_chunk || !user_data) {
        return false;
    }
    int* count = static_cast<int*>(user_data);
    (*count)++;
    return true;
}

void test_abi_and_null_guards(const std::filesystem::path& tmp_dir) {
    QERLIO_CHECK(std::string(qerlio_get_version()) == "0.3.1");

    QERLIO_CHECK(qerlio_engine_create(nullptr, nullptr) == QERLIO_ERR_INVALID_ARGUMENT);
    qerlio_engine_destroy(nullptr);
    qerlio_model_destroy(nullptr);
    qerlio_session_destroy(nullptr);
    QERLIO_CHECK(qerlio_session_cancel(nullptr) == QERLIO_ERR_INVALID_ARGUMENT);

    qerlio_engine_t* engine = nullptr;
    QERLIO_CHECK(qerlio_engine_create(nullptr, &engine) == QERLIO_OK);
    QERLIO_CHECK(engine != nullptr);

    char small_buf[5] = {'x', 'x', 'x', 'x', 'x'};
    QERLIO_CHECK(qerlio_get_last_error(nullptr, small_buf, sizeof(small_buf)) == QERLIO_ERR_INVALID_ARGUMENT);
    QERLIO_CHECK(qerlio_get_last_error(engine, nullptr, sizeof(small_buf)) == QERLIO_ERR_INVALID_ARGUMENT);
    QERLIO_CHECK(qerlio_get_last_error(engine, small_buf, 0) == QERLIO_ERR_INVALID_ARGUMENT);

    // Null config / null model_path / null out_model
    QERLIO_CHECK(qerlio_model_load(nullptr, nullptr, nullptr) == QERLIO_ERR_INVALID_ARGUMENT);
    qerlio_model_t* model = nullptr;
    QERLIO_CHECK(qerlio_model_load(engine, nullptr, &model) == QERLIO_ERR_INVALID_ARGUMENT);
    qerlio_model_config_t bad_cfg{};
    bad_cfg.model_path = nullptr;
    QERLIO_CHECK(qerlio_model_load(engine, &bad_cfg, &model) == QERLIO_ERR_INVALID_ARGUMENT);

    // Missing file
    const auto missing_path = (tmp_dir / "does_not_exist.gguf").string();
    bad_cfg.model_path = missing_path.c_str();
    QERLIO_CHECK(qerlio_model_load(engine, &bad_cfg, &model) != QERLIO_OK);
    QERLIO_CHECK(model == nullptr);

    // Small error buffer truncation & NUL termination check
    QERLIO_CHECK(qerlio_get_last_error(engine, small_buf, sizeof(small_buf)) == QERLIO_OK);
    QERLIO_CHECK(small_buf[4] == '\0');

    // Bad magic file
    const auto bad_magic_path = tmp_dir / "bad_magic.gguf";
    {
        std::ofstream f(bad_magic_path, std::ios::binary);
        const char bytes[32] = {'N', 'O', 'T', 'G', 3, 0, 0, 0};
        f.write(bytes, sizeof(bytes));
    }
    const auto bad_magic_str = bad_magic_path.string();
    bad_cfg.model_path = bad_magic_str.c_str();
    QERLIO_CHECK(qerlio_model_load(engine, &bad_cfg, &model) == QERLIO_ERR_MODEL_CORRUPT);

    // Truncated small file (< 24 bytes)
    const auto tiny_path = tmp_dir / "tiny.gguf";
    {
        std::ofstream f(tiny_path, std::ios::binary);
        const char bytes[6] = {'G', 'G', 'U', 'F', 3, 0};
        f.write(bytes, sizeof(bytes));
    }
    const auto tiny_str = tiny_path.string();
    bad_cfg.model_path = tiny_str.c_str();
    QERLIO_CHECK(qerlio_model_load(engine, &bad_cfg, &model) == QERLIO_ERR_MODEL_CORRUPT);

    // Valid synthetic Seed-OSS GGUF v3 file
    const auto valid_path = tmp_dir / "valid_seed_oss.gguf";
    {
        const auto bytes = build_synthetic_seed_oss_gguf();
        std::ofstream f(valid_path, std::ios::binary);
        f.write(reinterpret_cast<const char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
    }
    const auto valid_str = valid_path.string();
    qerlio_model_config_t valid_cfg{};
    valid_cfg.model_path = valid_str.c_str();
    valid_cfg.context_length = 4096;
    QERLIO_CHECK(qerlio_model_load(engine, &valid_cfg, &model) == QERLIO_OK);
    QERLIO_CHECK(model != nullptr);

    // Session lifecycle + generation + cancellation
    qerlio_session_t* session = nullptr;
    QERLIO_CHECK(qerlio_session_create(nullptr, &session) == QERLIO_ERR_INVALID_ARGUMENT);
    QERLIO_CHECK(qerlio_session_create(model, nullptr) == QERLIO_ERR_INVALID_ARGUMENT);
    QERLIO_CHECK(qerlio_session_create(model, &session) == QERLIO_OK);
    QERLIO_CHECK(session != nullptr);

    qerlio_generation_config_t gen_cfg{};
    gen_cfg.max_output_tokens = 5;
    int token_count = 0;
    qerlio_generation_metrics_t metrics{};

    QERLIO_CHECK(qerlio_generate_stream(nullptr, "hi", &gen_cfg, stream_collect_all, &token_count, &metrics) == QERLIO_ERR_INVALID_ARGUMENT);
    QERLIO_CHECK(qerlio_generate_stream(session, nullptr, &gen_cfg, stream_collect_all, &token_count, &metrics) == QERLIO_ERR_INVALID_ARGUMENT);
    QERLIO_CHECK(qerlio_generate_stream(session, "hi", &gen_cfg, nullptr, &token_count, &metrics) == QERLIO_ERR_INVALID_ARGUMENT);

    QERLIO_CHECK(qerlio_generate_stream(session, "hi", &gen_cfg, stream_collect_all, &token_count, &metrics) == QERLIO_OK);
    QERLIO_CHECK(token_count == 5);
    QERLIO_CHECK(metrics.tokens_generated == 5);

    int cancel_count = 0;
    QERLIO_CHECK(qerlio_generate_stream(session, "hi", &gen_cfg, stream_stop_after_two, &cancel_count, &metrics) == QERLIO_ERR_CANCELLED);
    QERLIO_CHECK(cancel_count == 2);

    QERLIO_CHECK(qerlio_session_cancel(session) == QERLIO_OK);
    qerlio_session_destroy(session);
    qerlio_model_destroy(model);
    qerlio_engine_destroy(engine);

    std::filesystem::remove(bad_magic_path);
    std::filesystem::remove(tiny_path);
    std::filesystem::remove(valid_path);
}

void test_gguf_parser_sha256_and_metadata(const std::filesystem::path& tmp_dir) {
    const auto abc_path = tmp_dir / "abc.bin";
    {
        std::ofstream f(abc_path, std::ios::binary);
        f.write("abc", 3);
    }
    std::string sha;
    std::string err;
    QERLIO_CHECK(qerlio::GGUFParser::compute_sha256(abc_path.string(), sha, err) == QERLIO_OK);
    QERLIO_CHECK(sha == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad");
    std::filesystem::remove(abc_path);

    const auto gguf_path = tmp_dir / "meta_test.gguf";
    {
        const auto bytes = build_synthetic_seed_oss_gguf();
        std::ofstream f(gguf_path, std::ios::binary);
        f.write(reinterpret_cast<const char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
    }
    qerlio::GGUFHeader hdr;
    qerlio::GGUFMetadata meta;
    QERLIO_CHECK(qerlio::GGUFParser::parse_header(gguf_path.string(), hdr, meta, err) == QERLIO_OK);
    QERLIO_CHECK(hdr.version == 3);
    QERLIO_CHECK(hdr.metadata_kv_count == 7);
    QERLIO_CHECK(meta.architecture == "seed_oss");
    QERLIO_CHECK(meta.name == "Hermes 4_3_36b");
    QERLIO_CHECK(meta.context_length == 524288);
    QERLIO_CHECK(meta.embedding_length == 5120);
    QERLIO_CHECK(meta.block_count == 64);
    QERLIO_CHECK(meta.feed_forward_length == 27648);
    QERLIO_CHECK(meta.quantization_type == "Q4_K_M");
    std::filesystem::remove(gguf_path);
}

void test_memory_scheduler_and_quant() {
    qerlio::ModelArchitectureDimensions dims;
    dims.layers = 64;
    dims.kv_heads = 8;
    dims.head_dim = 128;
    dims.bytes_per_element = 4;

#if SIZE_MAX > 0xFFFFFFFFULL
    const size_t kv_bytes = qerlio::MemoryPlanner::calculate_kv_cache_bytes(dims, 524288);
    QERLIO_CHECK(kv_bytes == static_cast<size_t>(274877906944ULL));
#endif

    // OOM check when model + margins exceed available RAM
    const auto oom = qerlio::MemoryPlanner::plan_memory(
        static_cast<size_t>(20ULL) * 1024 * 1024 * 1024,
        dims,
        4096,
        static_cast<size_t>(1024ULL) * 1024 * 1024
    );
    QERLIO_CHECK(!oom.fits_in_available_ram);
    QERLIO_CHECK(oom.recommended_context_length == 0);

    // Scheduler profiles & thermal states
    const auto crit = qerlio::AdaptiveScheduler::compute_policy(
        8, QERLIO_POWER_PERFORMANCE, qerlio::ThermalState::Critical);
    QERLIO_CHECK(!crit.allow_generation);
    QERLIO_CHECK(crit.active_threads == 1);

    const auto saver = qerlio::AdaptiveScheduler::compute_policy(
        8, QERLIO_POWER_SAVER, qerlio::ThermalState::Normal);
    QERLIO_CHECK(saver.allow_generation);
    QERLIO_CHECK(saver.active_threads == 4);

    const auto hot = qerlio::AdaptiveScheduler::compute_policy(
        8, QERLIO_POWER_BALANCED, qerlio::ThermalState::Hot);
    QERLIO_CHECK(hot.throttle_active);

    // UniversalQuantizer profiles
    const auto profiles = qerlio::UniversalQuantizer::get_all_profiles();
    QERLIO_CHECK(!profiles.empty());
    QERLIO_CHECK(qerlio::UniversalQuantizer::find_by_name("Q4_K_M") != nullptr);
    QERLIO_CHECK(qerlio::UniversalQuantizer::find_by_name("IQ3_S") != nullptr);
    QERLIO_CHECK(qerlio::UniversalQuantizer::find_by_name("IQ4_XS") != nullptr);
    QERLIO_CHECK(qerlio::UniversalQuantizer::estimate_model_bytes(1000000ULL, qerlio::QuantType::Q4_K_M) > 0);
}

void test_native_gemm_kernels_and_bounds() {
    const int has_avx2 = qerlio_gemm_has_avx2();
    QERLIO_CHECK(has_avx2 == 0 || has_avx2 == 1);

    std::vector<float> x(256, 1.0f);
    std::vector<uint8_t> w_q4k(2 * 144, 0);
    std::vector<uint8_t> w_q6k(2 * 210, 0);
    std::vector<uint8_t> w_q8_0(2 * 34, 0);
    std::vector<float> w_f32(2 * 16, 0.5f);
    std::vector<float> y(2, -1.0f);

    // Null & bounds rejections
    QERLIO_CHECK(qerlio_gemm_q4_k_f32(nullptr, w_q4k.data(), y.data(), 1, 2, 256, 0, 2) == -1);
    QERLIO_CHECK(qerlio_gemm_q4_k_f32(x.data(), w_q4k.data(), y.data(), 0, 2, 256, 0, 2) == -2);
    QERLIO_CHECK(qerlio_gemm_q4_k_f32(x.data(), w_q4k.data(), y.data(), 1, 2, 256, 0, 3) == -3);
    QERLIO_CHECK(qerlio_gemm_q4_k_f32(x.data(), w_q4k.data(), y.data(), 1, 2, 128, 0, 2) == -4);
    QERLIO_CHECK(qerlio_gemm_q6_k_f32(x.data(), w_q6k.data(), y.data(), 1, 2, 128, 0, 2) == -4);
    QERLIO_CHECK(qerlio_gemm_q8_0_f32(x.data(), w_q8_0.data(), y.data(), 1, 2, 16, 0, 2) == -4);
    QERLIO_CHECK(qerlio_gemm_f32(nullptr, w_f32.data(), y.data(), 1, 2, 16, 0, 2) == -1);

    // Row 0 of Q4_K: d=1.0 (0x3C00), dmin=0, sc[0..7]=1 -> subblock0 weights=0x11 -> q=1 -> dot(all_ones)=64.0
    for (int r = 0; r < 2; ++r) {
        uint8_t* blk = w_q4k.data() + r * 144;
        blk[0] = 0x00;
        blk[1] = 0x3C;  // fp16 1.0
        for (int j = 0; j < 4; ++j) {
            blk[4 + j] = 1;
            blk[4 + 4 + j] = 0;
        }
        blk[12] = 0x01;
        blk[13] = 0x01;
        blk[14] = 0x01;
        blk[15] = 0x01;
        std::memset(blk + 16, (r == 0) ? 0x11 : 0x22, 128);
    }
    QERLIO_CHECK(qerlio_gemm_q4_k_f32(x.data(), w_q4k.data(), y.data(), 1, 2, 256, 0, 2) == 0);
    QERLIO_CHECK(std::fabs(y[0] - 256.0f) < 1e-3f);
    QERLIO_CHECK(std::fabs(y[1] - 512.0f) < 1e-3f);

    // Row slice [1..2) of Q6_K: d=1.0, scales=1, all ql/qh zero -> q = -32 -> dot(all_ones) = 256 * (-32) = -8192.0
    for (int r = 0; r < 2; ++r) {
        uint8_t* blk = w_q6k.data() + r * 210;
        std::memset(blk + 192, (r == 1) ? 1 : 2, 16);
        blk[208] = 0x00;
        blk[209] = 0x3C;  // fp16 1.0
    }
    std::vector<float> y_q6k(2, 0.0f);
    QERLIO_CHECK(qerlio_gemm_q6_k_f32(x.data(), w_q6k.data() + 210, y_q6k.data(), 1, 2, 256, 1, 2) == 0);
    QERLIO_CHECK(std::fabs(y_q6k[0]) < 1e-6f);
    QERLIO_CHECK(std::fabs(y_q6k[1] - (-8192.0f)) < 1e-2f);

    // Q8_0: d=1.0, qs=3 -> 32 * 3 = 96.0
    for (int r = 0; r < 2; ++r) {
        uint8_t* blk = w_q8_0.data() + r * 34;
        blk[0] = 0x00;
        blk[1] = 0x3C;
        std::memset(blk + 2, (r == 0) ? 3 : 5, 32);
    }
    std::vector<float> y_q8(2, 0.0f);
    QERLIO_CHECK(qerlio_gemm_q8_0_f32(x.data(), w_q8_0.data(), y_q8.data(), 1, 2, 32, 0, 2) == 0);
    QERLIO_CHECK(std::fabs(y_q8[0] - 96.0f) < 1e-3f);
    QERLIO_CHECK(std::fabs(y_q8[1] - 160.0f) < 1e-3f);

    // F32
    std::vector<float> y_f32(2, 0.0f);
    QERLIO_CHECK(qerlio_gemm_f32(x.data(), w_f32.data(), y_f32.data(), 1, 2, 16, 0, 2) == 0);
    QERLIO_CHECK(std::fabs(y_f32[0] - 8.0f) < 1e-4f);
    QERLIO_CHECK(std::fabs(y_f32[1] - 8.0f) < 1e-4f);

    // Multi-row (m = 5): exercises both the 4-row tile path and the remainder path in native_gemm.c under ASan/UBSan
    std::vector<float> x5(5 * 256, 1.0f);
    std::vector<float> y5(5 * 2, 0.0f);
    QERLIO_CHECK(qerlio_gemm_q4_k_f32(x5.data(), w_q4k.data(), y5.data(), 5, 2, 256, 0, 2) == 0);
    for (int t = 0; t < 5; ++t) {
        QERLIO_CHECK(std::fabs(y5[t * 2 + 0] - 256.0f) < 1e-3f);
        QERLIO_CHECK(std::fabs(y5[t * 2 + 1] - 512.0f) < 1e-3f);
    }

    std::fill(y5.begin(), y5.end(), 0.0f);
    QERLIO_CHECK(qerlio_gemm_q6_k_f32(x5.data(), w_q6k.data(), y5.data(), 5, 2, 256, 0, 2) == 0);
    for (int t = 0; t < 5; ++t) {
        QERLIO_CHECK(std::fabs(y5[t * 2 + 0] - (-16384.0f)) < 1e-2f);
        QERLIO_CHECK(std::fabs(y5[t * 2 + 1] - (-8192.0f)) < 1e-2f);
    }

    std::fill(y5.begin(), y5.end(), 0.0f);
    QERLIO_CHECK(qerlio_gemm_q8_0_f32(x5.data(), w_q8_0.data(), y5.data(), 5, 2, 32, 0, 2) == 0);
    for (int t = 0; t < 5; ++t) {
        QERLIO_CHECK(std::fabs(y5[t * 2 + 0] - 96.0f) < 1e-3f);
        QERLIO_CHECK(std::fabs(y5[t * 2 + 1] - 160.0f) < 1e-3f);
    }

    std::fill(y5.begin(), y5.end(), 0.0f);
    QERLIO_CHECK(qerlio_gemm_f32(x5.data(), w_f32.data(), y5.data(), 5, 2, 16, 0, 2) == 0);
    for (int t = 0; t < 5; ++t) {
        QERLIO_CHECK(std::fabs(y5[t * 2 + 0] - 8.0f) < 1e-4f);
        QERLIO_CHECK(std::fabs(y5[t * 2 + 1] - 8.0f) < 1e-4f);
    }
}

}  // namespace

int main() {
    const auto tmp_dir = std::filesystem::temp_directory_path();
    test_abi_and_null_guards(tmp_dir);
    test_gguf_parser_sha256_and_metadata(tmp_dir);
    test_memory_scheduler_and_quant();
    test_native_gemm_kernels_and_bounds();
    std::cout << "[PASS] qerlio_cpp_safety_test passed all C++20 memory-safety checks.\n";
    return 0;
}
