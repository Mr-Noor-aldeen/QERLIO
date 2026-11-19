#include "universal_quant.hpp"
#include <cstring>

namespace qerlio {

static const QuantProfile PROFILES[] = {
    {QuantType::F32, "F32", QuantizationFamily::StandardGGUF, 32.0f, 1, 0.00f, true},
    {QuantType::F16, "F16", QuantizationFamily::StandardGGUF, 16.0f, 1, 0.00f, true},
    {QuantType::BF16, "BF16", QuantizationFamily::StandardGGUF, 16.0f, 1, 0.00f, true},

    {QuantType::Q8_0, "Q8_0", QuantizationFamily::StandardGGUF, 8.50f, 32, 0.0004f, true},
    {QuantType::Q5_1, "Q5_1", QuantizationFamily::StandardGGUF, 6.00f, 32, 0.0150f, true},
    {QuantType::Q5_0, "Q5_0", QuantizationFamily::StandardGGUF, 5.50f, 32, 0.0200f, true},
    {QuantType::Q4_1, "Q4_1", QuantizationFamily::StandardGGUF, 5.00f, 32, 0.0400f, true},
    {QuantType::Q4_0, "Q4_0", QuantizationFamily::StandardGGUF, 4.50f, 32, 0.0500f, true},

    {QuantType::Q2_K, "Q2_K", QuantizationFamily::KQuants, 2.56f, 256, 0.6710f, false},
    {QuantType::Q3_K_S, "Q3_K_S", QuantizationFamily::KQuants, 3.44f, 256, 0.2200f, false},
    {QuantType::Q3_K_M, "Q3_K_M", QuantizationFamily::KQuants, 3.91f, 256, 0.1400f, false},
    {QuantType::Q3_K_L, "Q3_K_L", QuantizationFamily::KQuants, 4.27f, 256, 0.0800f, false},
    {QuantType::Q4_K_S, "Q4_K_S", QuantizationFamily::KQuants, 4.50f, 256, 0.0350f, false},
    {QuantType::Q4_K_M, "Q4_K_M", QuantizationFamily::KQuants, 4.85f, 256, 0.0210f, false},
    {QuantType::Q5_K_S, "Q5_K_S", QuantizationFamily::KQuants, 5.50f, 256, 0.0100f, false},
    {QuantType::Q5_K_M, "Q5_K_M", QuantizationFamily::KQuants, 5.69f, 256, 0.0070f, false},
    {QuantType::Q6_K, "Q6_K", QuantizationFamily::KQuants, 6.56f, 256, 0.0030f, false},

    {QuantType::IQ1_S, "IQ1_S", QuantizationFamily::ExtremeIQuants, 1.56f, 256, 1.2500f, false},
    {QuantType::IQ1_M, "IQ1_M", QuantizationFamily::ExtremeIQuants, 1.75f, 256, 0.9800f, false},
    {QuantType::IQ2_XXS, "IQ2_XXS", QuantizationFamily::ExtremeIQuants, 2.06f, 256, 0.5200f, false},
    {QuantType::IQ2_XS, "IQ2_XS", QuantizationFamily::ExtremeIQuants, 2.31f, 256, 0.3800f, false},
    {QuantType::IQ2_S, "IQ2_S", QuantizationFamily::ExtremeIQuants, 2.50f, 256, 0.2900f, false},
    {QuantType::IQ3_XXS, "IQ3_XXS", QuantizationFamily::ExtremeIQuants, 3.06f, 256, 0.1600f, false},
    {QuantType::IQ3_S, "IQ3_S", QuantizationFamily::ExtremeIQuants, 3.44f, 256, 0.1100f, false},
    {QuantType::IQ4_NL, "IQ4_NL", QuantizationFamily::ExtremeIQuants, 4.50f, 32, 0.0300f, false},
    {QuantType::IQ4_XS, "IQ4_XS", QuantizationFamily::ExtremeIQuants, 4.25f, 256, 0.0380f, false},

    {QuantType::ZAQERIS_M3_GPTQ, "M3_GPTQ", QuantizationFamily::ZaqerisRVQ, 3.12f, 8, 0.0850f, true},

    {QuantType::FP8_E4M3, "FP8_E4M3", QuantizationFamily::HardwareFP8, 8.00f, 1, 0.0020f, true},
    {QuantType::FP8_E5M2, "FP8_E5M2", QuantizationFamily::HardwareFP8, 8.00f, 1, 0.0050f, true},

    {QuantType::KV_Q8_0, "KV_Q8_0", QuantizationFamily::KVCacheFormat, 8.00f, 32, 0.0010f, true},
    {QuantType::KV_Q4_0, "KV_Q4_0", QuantizationFamily::KVCacheFormat, 4.00f, 32, 0.0120f, true}
};

const QuantProfile& UniversalQuantizer::get_profile(QuantType type) {
    for (const auto& p : PROFILES) {
        if (p.type == type) {
            return p;
        }
    }
    return PROFILES[1];
}

const QuantProfile* UniversalQuantizer::find_by_name(const std::string& name) {
    for (const auto& p : PROFILES) {
        if (p.name == name) {
            return &p;
        }
    }
    return nullptr;
}

std::vector<QuantProfile> UniversalQuantizer::get_all_profiles() {
    std::vector<QuantProfile> list;
    for (const auto& p : PROFILES) {
        list.push_back(p);
    }
    return list;
}

size_t UniversalQuantizer::estimate_model_bytes(uint64_t parameter_count, QuantType type) {
    const auto& p = get_profile(type);
    double bits = static_cast<double>(parameter_count) * p.bits_per_weight;
    return static_cast<size_t>(bits / 8.0);
}

float UniversalQuantizer::calculate_compression_ratio(QuantType type) {
    const auto& p = get_profile(type);
    return 16.0f / p.bits_per_weight;
}

}
