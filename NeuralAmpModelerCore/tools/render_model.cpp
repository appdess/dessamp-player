#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

#include "../../AudioDSPTools/dsp/wav.h"
#include "NAM/dsp.h"
#include "NAM/get_dsp.h"
#include "NAM/player_manifest.h"
#include "NAM/slimmable.h"
#include "json.hpp"

namespace
{
constexpr int kDefaultBlockSize = 64;
constexpr int kMaximumBlockSize = 8192;
constexpr std::uint64_t kMaximumOfflineRenderInputBytes = 512ULL * 1024ULL * 1024ULL;

double ParseFiniteDouble(const std::string& raw, const char* label)
{
  std::size_t consumed = 0;
  double value = 0.0;
  try
  {
    value = std::stod(raw, &consumed);
  }
  catch (const std::exception&)
  {
    throw std::invalid_argument(std::string(label) + " must be numeric");
  }
  if (consumed != raw.size() || !std::isfinite(value))
    throw std::invalid_argument(std::string(label) + " must be finite");
  return value;
}

int ParseBlockSize(const std::string& raw)
{
  std::size_t consumed = 0;
  long value = 0;
  try
  {
    value = std::stol(raw, &consumed, 10);
  }
  catch (const std::exception&)
  {
    throw std::invalid_argument("Block size must be an integer");
  }
  if (consumed != raw.size() || value <= 0 || value > kMaximumBlockSize)
    throw std::invalid_argument("Block size must be between 1 and 8192");
  return static_cast<int>(value);
}

void WriteUInt16(std::ofstream& stream, std::uint16_t value)
{
  const std::array<char, 2> bytes = {
    static_cast<char>(value & 0xffU),
    static_cast<char>((value >> 8U) & 0xffU),
  };
  stream.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
}

void WriteUInt32(std::ofstream& stream, std::uint32_t value)
{
  const std::array<char, 4> bytes = {
    static_cast<char>(value & 0xffU),
    static_cast<char>((value >> 8U) & 0xffU),
    static_cast<char>((value >> 16U) & 0xffU),
    static_cast<char>((value >> 24U) & 0xffU),
  };
  stream.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
}

void WritePcm24Mono(const std::filesystem::path& path, const std::vector<NAM_SAMPLE>& audio,
                    std::uint32_t sampleRate)
{
  if (audio.empty())
    throw std::invalid_argument("Cannot write an empty render");
  if (audio.size() > (std::numeric_limits<std::uint32_t>::max() - 36U) / 3U)
    throw std::invalid_argument("Rendered WAV exceeds the RIFF size limit");
  if (std::filesystem::exists(path))
    throw std::invalid_argument("Output path already exists");
  const auto parent = path.parent_path();
  if (!parent.empty() && !std::filesystem::is_directory(parent))
    throw std::invalid_argument("Output directory does not exist");

  const auto dataBytes = static_cast<std::uint32_t>(audio.size() * 3U);
  std::ofstream stream(path, std::ios::binary | std::ios::out);
  if (!stream.is_open())
    throw std::runtime_error("Unable to create output WAV");
  stream.write("RIFF", 4);
  WriteUInt32(stream, 36U + dataBytes);
  stream.write("WAVE", 4);
  stream.write("fmt ", 4);
  WriteUInt32(stream, 16U);
  WriteUInt16(stream, 1U);
  WriteUInt16(stream, 1U);
  WriteUInt32(stream, sampleRate);
  WriteUInt32(stream, sampleRate * 3U);
  WriteUInt16(stream, 3U);
  WriteUInt16(stream, 24U);
  stream.write("data", 4);
  WriteUInt32(stream, dataBytes);
  for (const NAM_SAMPLE sample : audio)
  {
    if (!std::isfinite(sample))
      throw std::runtime_error("Model produced non-finite output");
    const double clipped = std::clamp(static_cast<double>(sample), -1.0, 1.0);
    const auto integer = static_cast<std::int32_t>(std::llround(clipped * 8388607.0));
    const std::uint32_t encoded = static_cast<std::uint32_t>(integer);
    const std::array<char, 3> bytes = {
      static_cast<char>(encoded & 0xffU),
      static_cast<char>((encoded >> 8U) & 0xffU),
      static_cast<char>((encoded >> 16U) & 0xffU),
    };
    stream.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
  }
  if (!stream.good())
  {
    stream.close();
    std::filesystem::remove(path);
    throw std::runtime_error("Unable to finish output WAV");
  }
}

struct Options
{
  std::filesystem::path modelPath;
  std::filesystem::path inputPath;
  std::filesystem::path outputPath;
  std::vector<double> parameters;
  bool qualityProvided = false;
  double quality = 1.0;
  int blockSize = kDefaultBlockSize;
};

Options ParseOptions(int argc, char* argv[])
{
  if (argc < 4)
    throw std::invalid_argument(
      "Usage: render_model <model.nam|model.dess> <input.wav> <output.wav> "
      "[--param <value> ...] [--quality <0..1>] [--block-size <frames>]");
  Options options;
  options.modelPath = argv[1];
  options.inputPath = argv[2];
  options.outputPath = argv[3];
  for (int index = 4; index < argc; ++index)
  {
    const std::string argument(argv[index]);
    if (argument == "--param")
    {
      if (++index >= argc)
        throw std::invalid_argument("--param requires a value");
      options.parameters.push_back(ParseFiniteDouble(argv[index], "Parameter"));
    }
    else if (argument == "--quality")
    {
      if (++index >= argc)
        throw std::invalid_argument("--quality requires a value");
      options.quality = ParseFiniteDouble(argv[index], "Quality");
      if (options.quality < 0.0 || options.quality > 1.0)
        throw std::invalid_argument("Quality must be in [0, 1]");
      options.qualityProvided = true;
    }
    else if (argument == "--block-size")
    {
      if (++index >= argc)
        throw std::invalid_argument("--block-size requires a value");
      options.blockSize = ParseBlockSize(argv[index]);
    }
    else
    {
      throw std::invalid_argument("Unknown argument: " + argument);
    }
  }
  return options;
}
} // namespace

int main(int argc, char* argv[])
{
  try
  {
    const Options options = ParseOptions(argc, argv);
    if (!std::filesystem::is_regular_file(options.modelPath))
      throw std::invalid_argument("Model path is not a regular file");
    if (!std::filesystem::is_regular_file(options.inputPath))
      throw std::invalid_argument("Input WAV path is not a regular file");

    std::string extension = options.modelPath.extension().string();
    std::transform(extension.begin(), extension.end(), extension.begin(),
                   [](const unsigned char character) { return static_cast<char>(std::tolower(character)); });
    const bool playerPackage = extension == ".dess";
    std::unique_ptr<nam::DSP> model;
    bool morph = false;
    if (playerPackage)
    {
      const auto manifest = nam::player::LoadManifest(options.modelPath, 64);
      morph = manifest.morph.enabled;
      model = nam::player::LoadPlayerDSP(manifest);
    }
    else
      model = nam::get_dsp(options.modelPath);
    if (model == nullptr)
      throw std::runtime_error("NAM Core returned no model");
    if (model->NumInputChannels() != 1 || model->NumOutputChannels() != 1)
      throw std::invalid_argument("The standalone renderer currently supports mono models only");
    if (model->NumGlobalParams() != static_cast<int>(options.parameters.size()))
      throw std::invalid_argument(
        "Model expects " + std::to_string(model->NumGlobalParams()) + " parameters; received "
        + std::to_string(options.parameters.size()));

    auto* slimmable = dynamic_cast<nam::SlimmableModel*>(model.get());
    if (options.qualityProvided && slimmable == nullptr)
      throw std::invalid_argument("--quality is only valid for a slimmable model");
    if (slimmable != nullptr)
      slimmable->SetSlimmableSize(options.quality);

    std::vector<float> inputFloat;
    double sampleRate = 0.0;
    const auto wavStatus = dsp::wav::LoadWithMaxDataBytes(
      options.inputPath.string().c_str(), inputFloat, sampleRate, kMaximumOfflineRenderInputBytes);
    if (wavStatus != dsp::wav::LoadReturnCode::SUCCESS)
      throw std::invalid_argument("Unable to load input WAV: " + dsp::wav::GetMsgForLoadReturnCode(wavStatus));
    const double expectedRate = model->GetExpectedSampleRate();
    if (expectedRate != NAM_UNKNOWN_EXPECTED_SAMPLE_RATE && std::abs(expectedRate - sampleRate) > 0.5)
      throw std::invalid_argument(
        "Input WAV sample rate does not match the model's expected sample rate");

    model->SetDefaultParams(
      options.parameters.empty() ? nullptr : options.parameters.data(),
      static_cast<int>(options.parameters.size()));
    model->ResetAndPrewarm(sampleRate, options.blockSize);

    std::vector<NAM_SAMPLE> input(inputFloat.begin(), inputFloat.end());
    std::vector<NAM_SAMPLE> output(input.size(), 0.0);
    for (std::size_t offset = 0; offset < input.size(); offset += static_cast<std::size_t>(options.blockSize))
    {
      const int frames = static_cast<int>(
        std::min(static_cast<std::size_t>(options.blockSize), input.size() - offset));
      NAM_SAMPLE* inputPointer = input.data() + offset;
      NAM_SAMPLE* outputPointer = output.data() + offset;
      model->process(
        &inputPointer,
        &outputPointer,
        frames,
        options.parameters.empty() ? nullptr : options.parameters.data(),
        static_cast<int>(options.parameters.size()));
    }
    if (!std::all_of(output.begin(), output.end(), [](const NAM_SAMPLE sample) { return std::isfinite(sample); }))
      throw std::runtime_error("Model produced non-finite output");
    const double peak = std::accumulate(
      output.begin(), output.end(), 0.0,
      [](const double current, const NAM_SAMPLE sample) {
        return std::max(current, std::abs(static_cast<double>(sample)));
      });
    WritePcm24Mono(options.outputPath, output, static_cast<std::uint32_t>(std::llround(sampleRate)));

    const nlohmann::json result = {
      {"status", "rendered"},
      {"model_path", std::filesystem::weakly_canonical(options.modelPath).generic_string()},
      {"input_path", std::filesystem::weakly_canonical(options.inputPath).generic_string()},
      {"output_path", std::filesystem::weakly_canonical(options.outputPath).generic_string()},
      {"sample_rate", static_cast<int>(std::llround(sampleRate))},
      {"frames", output.size()},
      {"parameters", options.parameters},
      {"quality", options.quality},
      {"player_package", playerPackage},
      {"morph", morph},
      {"peak", peak},
    };
    std::cout << result.dump() << "\n";
    return EXIT_SUCCESS;
  }
  catch (const std::exception& error)
  {
    std::cerr << "Model render failed: " << error.what() << "\n";
    return EXIT_FAILURE;
  }
}
