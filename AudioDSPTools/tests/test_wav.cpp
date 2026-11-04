#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

#include "../dsp/Resample.h"
#include "../dsp/wav.h"

namespace
{
void put_u16(std::ofstream& out, const std::uint16_t value)
{
  const unsigned char bytes[] = {static_cast<unsigned char>(value), static_cast<unsigned char>(value >> 8)};
  out.write(reinterpret_cast<const char*>(bytes), sizeof(bytes));
}

void put_u32(std::ofstream& out, const std::uint32_t value)
{
  const unsigned char bytes[] = {static_cast<unsigned char>(value), static_cast<unsigned char>(value >> 8),
                                 static_cast<unsigned char>(value >> 16), static_cast<unsigned char>(value >> 24)};
  out.write(reinterpret_cast<const char*>(bytes), sizeof(bytes));
}

void write_pcm16(const std::filesystem::path& path, const std::uint32_t declared_data_size,
                 const std::vector<unsigned char>& data, const std::uint32_t sample_rate = 48000)
{
  std::ofstream out(path, std::ios::binary);
  out.write("RIFF", 4);
  put_u32(out, 36 + declared_data_size);
  out.write("WAVEfmt ", 8);
  put_u32(out, 16);
  put_u16(out, 1);
  put_u16(out, 1);
  put_u32(out, sample_rate);
  put_u32(out, sample_rate * 2);
  put_u16(out, 2);
  put_u16(out, 16);
  out.write("data", 4);
  put_u32(out, declared_data_size);
  out.write(reinterpret_cast<const char*>(data.data()), static_cast<std::streamsize>(data.size()));
}

void write_ieee32(const std::filesystem::path& path, const float sample)
{
  std::ofstream out(path, std::ios::binary);
  out.write("RIFF", 4);
  put_u32(out, 40);
  out.write("WAVEfmt ", 8);
  put_u32(out, 16);
  put_u16(out, 3);
  put_u16(out, 1);
  put_u32(out, 48000);
  put_u32(out, 192000);
  put_u16(out, 4);
  put_u16(out, 32);
  out.write("data", 4);
  put_u32(out, 4);
  out.write(reinterpret_cast<const char*>(&sample), sizeof(sample));
}

void write_pcm24(const std::filesystem::path& path, const std::uint32_t declared_data_size,
                 const std::vector<unsigned char>& data)
{
  std::ofstream out(path, std::ios::binary);
  out.write("RIFF", 4);
  put_u32(out, 36 + declared_data_size);
  out.write("WAVEfmt ", 8);
  put_u32(out, 16);
  put_u16(out, 1);
  put_u16(out, 1);
  put_u32(out, 48000);
  put_u32(out, 144000);
  put_u16(out, 3);
  put_u16(out, 24);
  out.write("data", 4);
  put_u32(out, declared_data_size);
  out.write(reinterpret_cast<const char*>(data.data()), static_cast<std::streamsize>(data.size()));
}

dsp::wav::LoadReturnCode load(const std::filesystem::path& path, std::vector<float>& samples)
{
  double sample_rate = 0.0;
  return dsp::wav::Load(path.string().c_str(), samples, sample_rate);
}
} // namespace

int main(const int argc, char** argv)
{
  const auto directory = std::filesystem::temp_directory_path() / "dessmetal-wav-tests";
  std::filesystem::create_directories(directory);

  std::vector<float> samples;
  const auto badWave = directory / "bad-wave.wav";
  {
    std::ofstream out(badWave, std::ios::binary);
    out.write("RIFF", 4);
    put_u32(out, 4);
    out.write("NOPE", 4);
  }
  assert(load(badWave, samples) == dsp::wav::LoadReturnCode::ERROR_NOT_WAVE);

  const auto valid = directory / "valid.wav";
  write_pcm16(valid, 4, {0, 0, 0xff, 0x7f});
  assert(load(valid, samples) == dsp::wav::LoadReturnCode::SUCCESS);
  assert(samples.size() == 2);
  double explicitlyBoundedSampleRate = 0.0;
  samples.clear();
  assert(dsp::wav::LoadWithMaxDataBytes(valid.string().c_str(), samples, explicitlyBoundedSampleRate, 3)
         == dsp::wav::LoadReturnCode::ERROR_INVALID_FILE);
  samples.clear();
  assert(dsp::wav::LoadWithMaxDataBytes(valid.string().c_str(), samples, explicitlyBoundedSampleRate, 4)
         == dsp::wav::LoadReturnCode::SUCCESS);

  const auto implausibleRate = directory / "implausible-rate.wav";
  write_pcm16(implausibleRate, 4, {0, 0, 0xff, 0x7f}, 1);
  samples.clear();
  assert(load(implausibleRate, samples) == dsp::wav::LoadReturnCode::ERROR_INVALID_FILE);

  // A hostile rate ratio must not grow the intermediate resampling vector
  // beyond the convolution engine's tap budget.
  const std::vector<float> lowRateInput{0.0f, 1.0f, 0.5f, 0.0f};
  std::vector<float> boundedOutput{123.0f, 456.0f};
  dsp::ResampleCubic(lowRateInput, 1.0, 192000.0, 0.0, boundedOutput, 8192);
  assert(boundedOutput.size() == 8192);
  assert(boundedOutput.front() != 123.0f);

  for (std::size_t shortSize = 0; shortSize < 3; ++shortSize)
  {
    const auto shortSample = directory / ("short-24-sample-" + std::to_string(shortSize));
    {
      std::ofstream out(shortSample, std::ios::binary);
      const unsigned char bytes[] = {0x00, 0x00};
      out.write(reinterpret_cast<const char*>(bytes), static_cast<std::streamsize>(shortSize));
    }
    std::ifstream in(shortSample, std::ios::binary);
    int value = 123;
    assert(!dsp::wav::_ReadSigned24BitInt(in, value));
    assert(value == 0);
  }

  const auto truncated24 = directory / "truncated-24.wav";
  write_pcm24(truncated24, 3, {0, 0});
  samples.clear();
  assert(load(truncated24, samples) == dsp::wav::LoadReturnCode::ERROR_INVALID_FILE);

  const auto includedPad24 = directory / "included-pad-24.wav";
  write_pcm24(includedPad24, 4, {0, 0, 0, 0});
  samples.clear();
  assert(load(includedPad24, samples) == dsp::wav::LoadReturnCode::SUCCESS);
  assert(samples.size() == 1);

  const auto nonzeroPad24 = directory / "nonzero-pad-24.wav";
  write_pcm24(nonzeroPad24, 4, {0, 0, 0, 1});
  samples.clear();
  assert(load(nonzeroPad24, samples) == dsp::wav::LoadReturnCode::ERROR_INVALID_FILE);

  const auto nonFinite = directory / "non-finite.wav";
  write_ieee32(nonFinite, std::numeric_limits<float>::quiet_NaN());
  samples.clear();
  assert(load(nonFinite, samples) == dsp::wav::LoadReturnCode::ERROR_INVALID_FILE);

  const auto odd = directory / "odd.wav";
  write_pcm16(odd, 3, {0, 0, 0});
  samples.clear();
  assert(load(odd, samples) == dsp::wav::LoadReturnCode::ERROR_INVALID_FILE);

  const auto truncated = directory / "truncated.wav";
  write_pcm16(truncated, 4, {0, 0});
  samples.clear();
  assert(load(truncated, samples) == dsp::wav::LoadReturnCode::ERROR_INVALID_FILE);

  const auto negative = directory / "negative.wav";
  write_pcm16(negative, 0xffffffffU, {});
  samples.clear();
  assert(load(negative, samples) == dsp::wav::LoadReturnCode::ERROR_INVALID_FILE);

  const auto empty = directory / "empty.wav";
  write_pcm16(empty, 0, {});
  samples.clear();
  assert(load(empty, samples) == dsp::wav::LoadReturnCode::ERROR_INVALID_FILE);

  for (int i = 1; i < argc; ++i)
  {
    double sampleRate = 0.0;
    samples.clear();
    const auto result = dsp::wav::Load(argv[i], samples, sampleRate);
    if (result != dsp::wav::LoadReturnCode::SUCCESS)
      std::cerr << "Failed shipped IR: " << argv[i] << std::endl;
    assert(result == dsp::wav::LoadReturnCode::SUCCESS);
    assert(sampleRate >= 8000.0 && sampleRate <= 768000.0);
    assert(!samples.empty());
    assert(std::all_of(samples.begin(), samples.end(), [](const float sample) { return std::isfinite(sample); }));
  }

  std::filesystem::remove_all(directory);
  return 0;
}
