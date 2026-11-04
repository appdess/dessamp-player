//
//  wav.h
//  NeuralAmpModeler-macOS
//
//  Created by Steven Atkinson on 12/31/22.
//

#pragma once

#include <cstdint>
#include <fstream>
#include <string>
#include <vector>

namespace dsp
{
namespace wav
{
enum class LoadReturnCode
{
  SUCCESS = 0,
  ERROR_OPENING,
  ERROR_NOT_RIFF,
  ERROR_NOT_WAVE,
  ERROR_MISSING_FMT,
  ERROR_INVALID_FILE,
  ERROR_UNSUPPORTED_FORMAT_ALAW,
  ERROR_UNSUPPORTED_FORMAT_MULAW,
  ERROR_UNSUPPORTED_FORMAT_OTHER,
  ERROR_UNSUPPORTED_BITS_PER_SAMPLE,
  ERROR_NOT_MONO,
  ERROR_OTHER
};

// Get a string describing the error
std::string GetMsgForLoadReturnCode(LoadReturnCode rc);

// Load a WAV file into a provided array of doubles,
// And note the sample rate.
//
// Returns: as per return cases above
LoadReturnCode Load(const char* fileName, std::vector<float>& audio, double& sampleRate);

// Load with an explicit data-chunk ceiling for bounded offline tools. The
// ordinary loader retains its conservative default limit for product/runtime
// callers.
LoadReturnCode LoadWithMaxDataBytes(const char* fileName, std::vector<float>& audio, double& sampleRate,
                                    std::uint64_t maxAudioDataBytes);

// Load samples, 16-bit
void _LoadSamples16(std::ifstream& wavFile, const int chunkSize, std::vector<float>& samples);
// Load samples, 24-bit
bool _LoadSamples24(std::ifstream& wavFile, const int chunkSize, std::vector<float>& samples);
// Load samples, 32-bit
void _LoadSamples32FloatingPoint(std::ifstream& wavFile, const int chunkSize, std::vector<float>& samples);
// Load samples, 32-bit fixed point
void _LoadSamples32FixedPoint(std::ifstream& wavFile, const int chunkSize, std::vector<float>& samples);

// Read in a 24-bit sample and convert it to an int
bool _ReadSigned24BitInt(std::ifstream& stream, int& value);
}; // namespace wav
}; // namespace dsp
