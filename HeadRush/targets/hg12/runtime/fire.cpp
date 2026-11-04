#include "fire.h"
#include "NAM/a2_fast.h"
#include "NAM/player_manifest.h"
#include <array>
#include <cmath>
#include <memory>

struct fire_instance {
    std::unique_ptr<nam::DSP> dsp;
    int max_block;
};

extern "C" fire_instance *fire_create(const unsigned char *dess, size_t bytes, int max_block) {
    constexpr size_t header_size = 96;
    constexpr size_t fire_size = 273558;
    constexpr char fire_hash[] = "83636a24f0217e888495be26e7cf0e8f790a39e430317b6059366a6f72393e1d";
    if (!dess || bytes != fire_size || max_block < 1 || max_block > 1024) return nullptr;
    try {
        using namespace nam::player::detail;
        if (DigestHex(Sha256Bytes(dess, bytes)) != fire_hash) return nullptr;
        // Matching the entire immutable package binds its two component hashes, controls and order.
        const size_t manifest_size = ReadLittleEndian64(dess + 16);
        const size_t model_size = ReadLittleEndian64(dess + 24);
        if (manifest_size > bytes - header_size || model_size != bytes - header_size - manifest_size) return nullptr;
        const auto *model = dess + header_size + manifest_size;
        const auto manifest = nlohmann::json::parse(dess + header_size, model);
        const auto nam_model = nlohmann::json::parse(model, model + model_size);
        std::array<double, 5> defaults{};
        const std::array<const char *, 5> ids = {"lead_gain", "bass", "middle", "treble", "presence"};
        const auto &controls = manifest.at("controls");
        if (controls.size() != ids.size()) return nullptr;
        for (const auto &control : controls) {
            const auto index = control.at("model_parameter_index").get<size_t>();
            if (ids.at(index) != control.at("id").get<std::string>()) return nullptr;
            defaults[index] = control.at("default_value").get<double>();
        }
        auto dsp = nam::wavenet::a2_fast::Factory(nam_model.at("config"), nam_model.at("weights").get<std::vector<float>>(), 48000);
        if (!dsp || dsp->NumGlobalParams() != 5) return nullptr;
        dsp->SetDefaultParams(defaults.data(), 5);
        dsp->ResetAndPrewarm(48000, max_block);
        return new fire_instance{std::move(dsp),max_block};
    } catch (...) {
        return nullptr;
    }
}

extern "C" int fire_process(fire_instance *amp, float *input, float *output, int frames, const float controls[5]) {
    if (!amp || !input || !output || !controls || frames < 1 || frames > amp->max_block) return 0;
    std::array<double, 5> vector;
    for (size_t i = 0; i < vector.size(); i++) {
        const float value = controls[i];
        if (!std::isfinite(value) || value < 0 || value > 1) return 0;
        vector[i] = value;
    }
    try {
        amp->dsp->process(&input, &output, frames, vector.data(), vector.size());
        return 1;
    } catch (...) {
        return 0;
    }
}

extern "C" void fire_destroy(fire_instance *amp) {
    delete amp;
}
