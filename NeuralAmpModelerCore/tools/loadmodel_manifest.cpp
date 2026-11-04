#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <vector>

#include "NAM/get_dsp.h"
#include "NAM/player_manifest.h"
#include "json.hpp"

namespace
{
constexpr int kBlockSize = 64;
constexpr int kMaxCoreManifestConditions = 64;

void SmokeRender(nam::DSP& model, const std::vector<double>& parameters)
{
  double sampleRate = model.GetExpectedSampleRate();
  if (!std::isfinite(sampleRate) || sampleRate <= 0.0)
    sampleRate = 48000.0;
  model.SetDefaultParams(parameters.empty() ? nullptr : parameters.data(), static_cast<int>(parameters.size()));
  model.Reset(sampleRate, kBlockSize);
  const int inputChannels = model.NumInputChannels();
  const int outputChannels = model.NumOutputChannels();
  if (inputChannels <= 0 || outputChannels <= 0 || inputChannels > 64 || outputChannels > 64)
    throw std::runtime_error("Model channel counts are outside the player limit");

  std::vector<std::vector<NAM_SAMPLE>> inputs(
    static_cast<std::size_t>(inputChannels), std::vector<NAM_SAMPLE>(kBlockSize, 0.0));
  std::vector<std::vector<NAM_SAMPLE>> outputs(
    static_cast<std::size_t>(outputChannels), std::vector<NAM_SAMPLE>(kBlockSize, 0.0));
  std::vector<NAM_SAMPLE*> inputPointers(static_cast<std::size_t>(inputChannels));
  std::vector<NAM_SAMPLE*> outputPointers(static_cast<std::size_t>(outputChannels));
  for (int channel = 0; channel < inputChannels; ++channel)
    inputPointers[static_cast<std::size_t>(channel)] = inputs[static_cast<std::size_t>(channel)].data();
  for (int channel = 0; channel < outputChannels; ++channel)
    outputPointers[static_cast<std::size_t>(channel)] = outputs[static_cast<std::size_t>(channel)].data();

  model.process(inputPointers.data(), outputPointers.data(), kBlockSize,
                parameters.empty() ? nullptr : parameters.data(), static_cast<int>(parameters.size()));
  for (const auto& channel : outputs)
    if (!std::all_of(channel.begin(), channel.end(), [](const NAM_SAMPLE sample) { return std::isfinite(sample); }))
      throw std::runtime_error("Model produced non-finite output during the player smoke render");
}
} // namespace

int main(int argc, char* argv[])
{
  if (argc != 2)
  {
    std::cerr << "Usage: loadmodel_manifest <model.dess|player-manifest.json>\n";
    return EXIT_FAILURE;
  }
  try
  {
    const auto manifest =
      nam::player::LoadManifest(std::filesystem::path(argv[1]), kMaxCoreManifestConditions);
    std::unique_ptr<nam::DSP> model = nam::player::LoadPlayerDSP(manifest);
    const auto parameters = manifest.DefaultModelParameters();
    SmokeRender(*model, parameters);

    const nlohmann::json output = {
      {"status", "valid"},
      {"schema_version", 1},
      {"model_path", manifest.model_path.generic_string()},
      {"model_sha256", manifest.model_sha256},
      {"parameter_count", parameters.size()},
      {"slimmable", manifest.runtime_quality.mode == "slimmable"},
      {"morph", manifest.morph.enabled},
      {"selected_size", manifest.runtime_quality.default_size},
      {"size_breakpoints", manifest.runtime_quality.breakpoints},
    };
    std::cout << output.dump() << "\n";
    return EXIT_SUCCESS;
  }
  catch (const std::exception& error)
  {
    std::cerr << "Player manifest validation failed: " << error.what() << "\n";
    return EXIT_FAILURE;
  }
}
