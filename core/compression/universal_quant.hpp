#ifndef QERLIO_UNIVERSAL_QUANT_HPP
#define QERLIO_UNIVERSAL_QUANT_HPP

#include <cstdint>
#include <cstddef>
#include <string>
#include <vector>

namespace qerlio {

enum class QuantizationFamily {
    StandardGGUF,
    KQuants,
    ExtremeIQuants,
    ZaqerisRVQ,
    HardwareFP8,
    KVCacheFormat
};

enum class QuantType {
    F32,
    F16,
    BF16,
    Q8_0,
    Q5_1,
    Q5_0,
    Q4_1,
    Q4_0,
    Q2_K,
    Q3_K_S,
    Q3_K_M,
    Q3_K_L,
    Q4_K_S,
    Q4_K_M,
    Q5_K_S,
    Q5_K_M,
    Q6_K,
    IQ1_S,
    IQ1_M,
    IQ2_XXS,
    IQ2_XS,
    IQ2_S,
    IQ3_XXS,
    IQ3_S,
    IQ4_NL,
    IQ4_XS,
    ZAQERIS_M3_GPTQ,
    FP8_E4M3,
    FP8_E5M2,
    KV_Q8_0,
    KV_Q4_0
};

struct QuantProfile {
    QuantType type;
    const char* name;
    QuantizationFamily family;
    float bits_per_weight;
    uint32_t block_size;
    float typical_perplexity_delta;
    bool hardware_accelerated;
};

class UniversalQuantizer {
public:
    static const QuantProfile& get_profile(QuantType type);
    static const QuantProfile* find_by_name(const std::string& name);
    static std::vector<QuantProfile> get_all_profiles();

    static size_t estimate_model_bytes(uint64_t parameter_count, QuantType type);
    static float calculate_compression_ratio(QuantType type);
};

}

#endif
