//
//  wav.cpp
//  NeuralAmpModeler-macOS
//
//  Created by Steven Atkinson on 12/31/22.
//

#include <algorithm>
#include <cstdint>
#include <cstring> // strncmp
#include <cmath> // pow
#include <limits>
#include <fstream>
#include <iostream>
#include <string>
#include <sstream>
#include <unordered_set>
#include <vector>

#include "wav.h"

struct WaveFileData
{
  // TODO use types like uint32_t, etc
  struct RiffChunk
  {
    bool valid = false; // Have we gotten this info yet?
    int size; // NB: Of the rest of the file
    char format[4];
  } riffChunk;

  struct FmtChunk
  {
    bool valid = false;
    int size;
    // PCM: 1
    // IEEE: 3
    // A-law: 6
    // mu-law: 7
    // Extensible: 65534
    unsigned short audioFormat;
    short numChannels;
    int sampleRate;
    int byteRate;
    short blockAlign;
    short bitsPerSample;
    struct Extensible
    {
      std::uint16_t validBitsPerSample;
      std::uint32_t channelMask;
      std::uint32_t subFormat; // PCM, IEEE
    } extensible;
  } fmtChunk;

  struct FactChunk
  {
    bool valid = false;
    int size;
    int numSamples;
  } factChunk;

  struct DataChunk
  {
    bool valid = false;
    char id[4];
    int size;
  } dataChunk;
};

const unsigned short AUDIO_FORMAT_PCM = 1;
const unsigned short AUDIO_FORMAT_IEEE = 3;
const unsigned short AUDIO_FORMAT_ALAW = 6;
const unsigned short AUDIO_FORMAT_MULAW = 7;
const unsigned short AUDIO_FORMAT_EXTENSIBLE = 65534;

bool idIsNotJunk(char* id)
{
  return strncmp(id, "RIFF", 4) == 0 || strncmp(id, "WAVE", 4) == 0 || strncmp(id, "fmt ", 4) == 0
         || strncmp(id, "data", 4) == 0;
}

int ReadInt(std::ifstream& file)
{
  int value = 0;
  file.read(reinterpret_cast<char*>(&value), 4);
  return value;
}

short ReadShort(std::ifstream& file)
{
  short value = 0;
  file.read(reinterpret_cast<char*>(&value), 2);
  return value;
}

unsigned short ReadUnsignedShort(std::ifstream& file)
{
  unsigned short value = 0;
  file.read(reinterpret_cast<char*>(&value), 2);
  return value;
}

namespace
{
// The loader is used for mono cabinet IRs, whose convolution path consumes at
// most 8192 taps. Generous but finite format bounds keep malformed inputs from
// turning a plug-in load into an excessive allocation or resampling job.
constexpr std::uint64_t kDefaultMaxAudioDataBytes = 16ULL * 1024ULL * 1024ULL;
constexpr int kMinAudioSampleRate = 8000;
constexpr int kMaxAudioSampleRate = 768000;

bool HasBytesRemaining(std::ifstream& file, const std::uint64_t count)
{
  const std::streampos current = file.tellg();
  if (current < 0)
    return false;
  file.seekg(0, std::ios::end);
  const std::streampos end = file.tellg();
  file.seekg(current);
  return end >= current && static_cast<std::uint64_t>(end - current) >= count;
}
} // namespace

dsp::wav::LoadReturnCode ReadJunk(std::ifstream& file)
{
  const int chunkSize = ReadInt(file);
  if (!file.good() || chunkSize < 0)
    return dsp::wav::LoadReturnCode::ERROR_INVALID_FILE;
  const std::uint64_t paddedSize = static_cast<std::uint64_t>(chunkSize) + (chunkSize % 2);
  if (!HasBytesRemaining(file, paddedSize))
    return dsp::wav::LoadReturnCode::ERROR_INVALID_FILE;
  file.seekg(static_cast<std::streamoff>(paddedSize), std::ios::cur);
  return file.good() ? dsp::wav::LoadReturnCode::SUCCESS : dsp::wav::LoadReturnCode::ERROR_INVALID_FILE;
}

std::string dsp::wav::GetMsgForLoadReturnCode(LoadReturnCode retCode)
{
  std::stringstream message;

  switch (retCode)
  {
    case (LoadReturnCode::ERROR_OPENING):
      message << "Failed to open file (is it being used by another "
                 "program?)";
      break;
    case (LoadReturnCode::ERROR_NOT_RIFF): message << "File is not a WAV file."; break;
    case (LoadReturnCode::ERROR_NOT_WAVE): message << "File is not a WAV file."; break;
    case (LoadReturnCode::ERROR_MISSING_FMT): message << "File is missing expected format chunk."; break;
    case (LoadReturnCode::ERROR_INVALID_FILE): message << "WAV file contents are invalid."; break;
    case (LoadReturnCode::ERROR_UNSUPPORTED_FORMAT_ALAW): message << "Unsupported file format \"A-law\""; break;
    case (LoadReturnCode::ERROR_UNSUPPORTED_FORMAT_MULAW): message << "Unsupported file format \"mu-law\""; break;
    case (LoadReturnCode::ERROR_UNSUPPORTED_FORMAT_OTHER): message << "Unsupported file format."; break;
    case (LoadReturnCode::ERROR_NOT_MONO): message << "File is not mono."; break;
    case (LoadReturnCode::ERROR_UNSUPPORTED_BITS_PER_SAMPLE): message << "Unsupported bits per sample"; break;
    case (dsp::wav::LoadReturnCode::ERROR_OTHER): message << "???"; break;
    default: message << "???"; break;
  }

  return message.str();
}

dsp::wav::LoadReturnCode ReadRiffChunk(std::ifstream& wavFile, WaveFileData::RiffChunk& chunk)
{
  if (chunk.valid)
  {
    std::cerr << "Error: RIFF chunk already read." << std::endl;
    return dsp::wav::LoadReturnCode::ERROR_INVALID_FILE;
  }
  chunk.size = ReadInt(wavFile);
  if (!wavFile.good() || chunk.size < 4 || !HasBytesRemaining(wavFile, 4))
    return dsp::wav::LoadReturnCode::ERROR_INVALID_FILE;
  wavFile.read(chunk.format, 4);
  if (strncmp(chunk.format, "WAVE", 4) != 0)
  {
    std::cerr << "Error: File format is not expected 'WAVE'. Got '" << std::string(chunk.format, 4)
              << "' instead." << std::endl;
    return dsp::wav::LoadReturnCode::ERROR_NOT_WAVE;
  }
  chunk.valid = true;
  return dsp::wav::LoadReturnCode::SUCCESS;
}

dsp::wav::LoadReturnCode ReadFmtChunk(std::ifstream& wavFile, WaveFileData& wfd, double& sampleRate)
{
  if (wfd.fmtChunk.valid)
  {
    std::cerr << "Error: Format chunk already read." << std::endl;
    return dsp::wav::LoadReturnCode::ERROR_INVALID_FILE;
  }
  if (!wfd.riffChunk.valid)
  {
    std::cerr << "Error: Missing RIFF chunk." << std::endl;
    return dsp::wav::LoadReturnCode::ERROR_INVALID_FILE;
  }

  wfd.fmtChunk.size = ReadInt(wavFile);
  if (!wavFile.good() || wfd.fmtChunk.size < 16
      || !HasBytesRemaining(wavFile, static_cast<std::uint64_t>(wfd.fmtChunk.size)))
  {
    std::cerr << "WAV chunk 1 size is " << wfd.fmtChunk.size
              << ", which is smaller than the requried 16 to fit the expected "
                 "information."
              << std::endl;
    return dsp::wav::LoadReturnCode::ERROR_INVALID_FILE;
  }

  wfd.fmtChunk.audioFormat = ReadUnsignedShort(wavFile);
  std::unordered_set<unsigned short> supportedFormats{AUDIO_FORMAT_PCM, AUDIO_FORMAT_IEEE, AUDIO_FORMAT_EXTENSIBLE};
  if (supportedFormats.find(wfd.fmtChunk.audioFormat) == supportedFormats.end())
  {
    std::cerr << "Error: Unsupported WAV format detected. ";
    switch (wfd.fmtChunk.audioFormat)
    {
      case AUDIO_FORMAT_ALAW:
        std::cerr << "(Got: A-law)" << std::endl;
        return dsp::wav::LoadReturnCode::ERROR_UNSUPPORTED_FORMAT_ALAW;
      case AUDIO_FORMAT_MULAW:
        std::cerr << "(Got: mu-law)" << std::endl;
        return dsp::wav::LoadReturnCode::ERROR_UNSUPPORTED_FORMAT_MULAW;
      default:
        std::cerr << "(Got unknown format " << wfd.fmtChunk.audioFormat << ")" << std::endl;
        return dsp::wav::LoadReturnCode::ERROR_INVALID_FILE;
    }
  }

  wfd.fmtChunk.numChannels = ReadShort(wavFile);
  // HACK
  // Note for future: for multi-channel files, samples are laid out with channel in the inner loop.
  if (wfd.fmtChunk.numChannels != 1)
  {
    std::cerr << "Require mono (using for IR loading)" << std::endl;
    return dsp::wav::LoadReturnCode::ERROR_NOT_MONO;
  }

  wfd.fmtChunk.sampleRate = ReadInt(wavFile);
  wfd.fmtChunk.byteRate = ReadInt(wavFile);
  wfd.fmtChunk.blockAlign = ReadShort(wavFile);
  wfd.fmtChunk.bitsPerSample = ReadShort(wavFile);
  if (!wavFile.good() || wfd.fmtChunk.sampleRate < kMinAudioSampleRate
      || wfd.fmtChunk.sampleRate > kMaxAudioSampleRate || wfd.fmtChunk.byteRate <= 0
      || wfd.fmtChunk.blockAlign <= 0 || wfd.fmtChunk.bitsPerSample <= 0)
    return dsp::wav::LoadReturnCode::ERROR_INVALID_FILE;
  const std::uint64_t expectedByteRate = static_cast<std::uint64_t>(wfd.fmtChunk.sampleRate)
                                         * static_cast<std::uint64_t>(wfd.fmtChunk.blockAlign);
  if (expectedByteRate != static_cast<std::uint64_t>(wfd.fmtChunk.byteRate))
    return dsp::wav::LoadReturnCode::ERROR_INVALID_FILE;
  int bytesRead = 16;

  if (wfd.fmtChunk.audioFormat == AUDIO_FORMAT_EXTENSIBLE)
  {
    unsigned short cbSize = ReadUnsignedShort(wavFile);
    if (!wavFile.good() || cbSize < 22 || wfd.fmtChunk.size < 18 + cbSize)
      return dsp::wav::LoadReturnCode::ERROR_INVALID_FILE;
    // Do we need to assert or modify the data loading below if this doesn't match bitsPerSample?
    wfd.fmtChunk.extensible.validBitsPerSample = ReadUnsignedShort(wavFile);
    auto read_u32 = [&]() -> std::uint32_t {
      std::uint8_t b[4] = {};
      wavFile.read((char*)b, 4);
      return static_cast<std::uint32_t>(b[0]) | (static_cast<std::uint32_t>(b[1]) << 8)
             | (static_cast<std::uint32_t>(b[2]) << 16) | (static_cast<std::uint32_t>(b[3]) << 24);
    };
    wfd.fmtChunk.extensible.channelMask = read_u32();
    std::uint8_t guid[16];
    wavFile.read((char*)guid, 16);
    if (!wavFile.good())
      return dsp::wav::LoadReturnCode::ERROR_INVALID_FILE;
    wfd.fmtChunk.extensible.subFormat = guid[1] << 8 | guid[0];
    bytesRead += cbSize + 2; // Don't forget the 2 for the cbSize itself!
  }

  // Skip any extra bytes in the fmt chunk
  // This should probably be a remainder of a dword so that we're mod-4
  if (wfd.fmtChunk.size > bytesRead)
  {
    const int extraBytes = wfd.fmtChunk.size - bytesRead;
    if (extraBytes >= 4)
    {
      std::cerr << "More than 4 extra bytes in fmt chunk." << std::endl;
      return dsp::wav::LoadReturnCode::ERROR_INVALID_FILE;
    }
    wavFile.ignore(extraBytes);
  }

  // Store SR for final return
  sampleRate = (double)wfd.fmtChunk.sampleRate;

  wfd.fmtChunk.valid = true;
  return dsp::wav::LoadReturnCode::SUCCESS;
}

dsp::wav::LoadReturnCode ReadFactChunk(std::ifstream& wavFile, WaveFileData& wfd)
{
  if (wfd.factChunk.valid)
  {
    std::cerr << "Error: Duplicate fact chunk." << std::endl;
    return dsp::wav::LoadReturnCode::ERROR_INVALID_FILE;
  }
  if (!wfd.riffChunk.valid)
  {
    std::cerr << "Error: Missing RIFF chunk." << std::endl;
    return dsp::wav::LoadReturnCode::ERROR_INVALID_FILE;
  }
  // We could assert that the fmt chunk was also read first but I'm not sure that's necessary for the file to be valid.

  wfd.factChunk.size = ReadInt(wavFile);
  if (wfd.factChunk.size != 4)
  {
    std::cerr << "Error: Invalid fact chunk size. Only 4 is supported; got " << wfd.factChunk.size << " instead."
              << std::endl;
    return dsp::wav::LoadReturnCode::ERROR_INVALID_FILE;
  }
  wfd.factChunk.numSamples = ReadInt(wavFile);
  if (!wavFile.good() || wfd.factChunk.numSamples < 0)
    return dsp::wav::LoadReturnCode::ERROR_INVALID_FILE;

  wfd.factChunk.valid = true;
  return dsp::wav::LoadReturnCode::SUCCESS;
}

int GetAudioFormat(WaveFileData& wfd)
{
  return wfd.fmtChunk.audioFormat == AUDIO_FORMAT_EXTENSIBLE ? wfd.fmtChunk.extensible.subFormat
                                                             : wfd.fmtChunk.audioFormat;
}

dsp::wav::LoadReturnCode ReadDataChunk(std::ifstream& wavFile, WaveFileData& wfd, std::vector<float>& audio,
                                      const std::uint64_t maxAudioDataBytes)
{
  if (wfd.dataChunk.valid)
  {
    std::cerr << "Error: Already read data chunk." << std::endl;
    return dsp::wav::LoadReturnCode::ERROR_INVALID_FILE;
  }
  if (!wfd.riffChunk.valid)
  {
    std::cerr << "Error: Missing RIFF chunk." << std::endl;
    return dsp::wav::LoadReturnCode::ERROR_INVALID_FILE;
  }
  if (!wfd.fmtChunk.valid) // fmt chunk must come before data chunk
  {
    std::cerr << "Error: Tried to read data chunk before fmt chunk." << std::endl;
    return dsp::wav::LoadReturnCode::ERROR_INVALID_FILE;
  }
  if (wfd.fmtChunk.audioFormat == AUDIO_FORMAT_EXTENSIBLE
      && !wfd.factChunk.valid) // fact chunk must come before data chunk
  {
    std::cerr << "Error: Tried to read data chunk before fact chunk for extensible format WAVE file." << std::endl;
    return dsp::wav::LoadReturnCode::ERROR_INVALID_FILE;
  }

  // Size of the data chunk, in bits.
  wfd.dataChunk.size = ReadInt(wavFile);
  if (!wavFile.good() || wfd.dataChunk.size <= 0
      || maxAudioDataBytes == 0
      || static_cast<std::uint64_t>(wfd.dataChunk.size) > maxAudioDataBytes
      || !HasBytesRemaining(wavFile, static_cast<std::uint64_t>(wfd.dataChunk.size)))
  {
    return dsp::wav::LoadReturnCode::ERROR_INVALID_FILE;
  }

  const int bytesPerSample = wfd.fmtChunk.bitsPerSample / 8;
  if (wfd.fmtChunk.bitsPerSample % 8 != 0 || bytesPerSample <= 0
      || wfd.fmtChunk.blockAlign != bytesPerSample)
  {
    return dsp::wav::LoadReturnCode::ERROR_INVALID_FILE;
  }
  const int audioFormat = GetAudioFormat(wfd);
  const int trailingBytes = wfd.dataChunk.size % bytesPerSample;
  const bool hasCompatibleIncludedPadding = audioFormat == AUDIO_FORMAT_PCM
                                            && wfd.fmtChunk.bitsPerSample == 24 && trailingBytes == 1;
  if (trailingBytes != 0 && !hasCompatibleIncludedPadding)
    return dsp::wav::LoadReturnCode::ERROR_INVALID_FILE;
  const int sampleDataSize = wfd.dataChunk.size - trailingBytes;
  if (sampleDataSize <= 0)
    return dsp::wav::LoadReturnCode::ERROR_INVALID_FILE;

  if (audioFormat == AUDIO_FORMAT_IEEE)
  {
    if (wfd.fmtChunk.bitsPerSample == 32)
      dsp::wav::_LoadSamples32FloatingPoint(wavFile, sampleDataSize, audio);
    else
    {
      std::cerr << "Error: Unsupported bits per sample for IEEE files: " << wfd.fmtChunk.bitsPerSample << std::endl;
      return dsp::wav::LoadReturnCode::ERROR_UNSUPPORTED_BITS_PER_SAMPLE;
    }
  }
  else if (audioFormat == AUDIO_FORMAT_PCM)
  {
    if (wfd.fmtChunk.bitsPerSample == 16)
      dsp::wav::_LoadSamples16(wavFile, sampleDataSize, audio);
    else if (wfd.fmtChunk.bitsPerSample == 24)
    {
      if (!dsp::wav::_LoadSamples24(wavFile, sampleDataSize, audio))
        return dsp::wav::LoadReturnCode::ERROR_INVALID_FILE;
    }
    else if (wfd.fmtChunk.bitsPerSample == 32)
      dsp::wav::_LoadSamples32FixedPoint(wavFile, sampleDataSize, audio);
    else
    {
      std::cerr << "Error: Unsupported bits per sample for PCM files: " << wfd.fmtChunk.bitsPerSample << std::endl;
      return dsp::wav::LoadReturnCode::ERROR_UNSUPPORTED_BITS_PER_SAMPLE;
    }
  }
  else
  {
    std::cerr << "Error: Unsupported audio format: " << audioFormat << std::endl;
    return dsp::wav::LoadReturnCode::ERROR_UNSUPPORTED_FORMAT_OTHER;
  }
  if (trailingBytes == 1)
  {
    char includedPadding = 0;
    wavFile.read(&includedPadding, 1);
    if (!wavFile.good() || includedPadding != 0)
      return dsp::wav::LoadReturnCode::ERROR_INVALID_FILE;
  }
  if (!wavFile.good() || audio.empty()
      || !std::all_of(audio.begin(), audio.end(), [](const float sample) { return std::isfinite(sample); }))
    return dsp::wav::LoadReturnCode::ERROR_INVALID_FILE;
  wfd.dataChunk.valid = true;
  return dsp::wav::LoadReturnCode::SUCCESS;
}

dsp::wav::LoadReturnCode dsp::wav::LoadWithMaxDataBytes(const char* fileName, std::vector<float>& audio,
                                                       double& sampleRate,
                                                       const std::uint64_t maxAudioDataBytes)
{
  // FYI: https://www.mmsp.ece.mcgill.ca/Documents/AudioFormats/WAVE/WAVE.html
  // Open the WAV file for reading
  std::ifstream wavFile(fileName, std::ios::binary);

  // Check if the file was opened successfully
  if (!wavFile.is_open())
  {
    std::cerr << "Error opening WAV file" << std::endl;
    return dsp::wav::LoadReturnCode::ERROR_OPENING;
  }

  char chunkId[4] = {};
  auto ReadChunkID = [&]() { return static_cast<bool>(wavFile.read(chunkId, 4)); };

  WaveFileData wfd;
  dsp::wav::LoadReturnCode returnCode;
  while (!wfd.dataChunk.valid && !wavFile.eof())
  {
    if (!ReadChunkID())
      return dsp::wav::LoadReturnCode::ERROR_INVALID_FILE;
    if (!wfd.riffChunk.valid && strncmp(chunkId, "RIFF", 4) != 0)
    {
      {
        std::cerr << "Error: File does not start with expected RIFF chunk." << std::endl;
        wavFile.close();
        return dsp::wav::LoadReturnCode::ERROR_NOT_RIFF;
      }
    }
    // Read the various chunks
    if (strncmp(chunkId, "RIFF", 4) == 0)
    {
      returnCode = ReadRiffChunk(wavFile, wfd.riffChunk);
    }
    else if (strncmp(chunkId, "fmt ", 4) == 0)
    {
      returnCode = ReadFmtChunk(wavFile, wfd, sampleRate);
    }
    else if (strncmp(chunkId, "fact", 4) == 0)
    {
      returnCode = ReadFactChunk(wavFile, wfd);
    }
    else if (strncmp(chunkId, "data", 4) == 0)
    {
      returnCode = ReadDataChunk(wavFile, wfd, audio, maxAudioDataBytes);
    }
    else
    { // There might be junk chunks; just ignore them.
      returnCode = ReadJunk(wavFile);
    }
    if (returnCode != dsp::wav::LoadReturnCode::SUCCESS)
    {
      wavFile.close();
      return returnCode;
    }
  }
  wavFile.close();
  if (!wfd.dataChunk.valid)
  { // This implicitly asserts that the fmt chunk was read and gave us the sample rate
    std::cerr << "Error: File does not contain expected data chunk." << std::endl;
    return dsp::wav::LoadReturnCode::ERROR_INVALID_FILE;
  }
  return dsp::wav::LoadReturnCode::SUCCESS;
}

dsp::wav::LoadReturnCode dsp::wav::Load(const char* fileName, std::vector<float>& audio, double& sampleRate)
{
  return dsp::wav::LoadWithMaxDataBytes(fileName, audio, sampleRate, kDefaultMaxAudioDataBytes);
}

void dsp::wav::_LoadSamples16(std::ifstream& wavFile, const int chunkSize, std::vector<float>& samples)
{
  if (chunkSize <= 0 || chunkSize % 2 != 0)
    return;
  // Allocate an array to hold the samples
  std::vector<short> tmp(chunkSize / 2); // 16 bits (2 bytes) per sample

  // Read the samples from the file into the array
  wavFile.read(reinterpret_cast<char*>(tmp.data()), chunkSize);

  // Copy into the return array
  const float scale = 1.0 / ((double)(1 << 15));
  samples.resize(tmp.size());
  for (size_t i = 0; i < samples.size(); i++)
    samples[i] = scale * ((float)tmp[i]); // 2^16
}

bool dsp::wav::_LoadSamples24(std::ifstream& wavFile, const int chunkSize, std::vector<float>& samples)
{
  if (chunkSize <= 0 || chunkSize % 3 != 0)
    return false;
  // Allocate an array to hold the samples
  std::vector<int> tmp(chunkSize / 3); // 24 bits (3 bytes) per sample
  // Read in and convert the samples
  for (int& x : tmp)
  {
    if (!dsp::wav::_ReadSigned24BitInt(wavFile, x))
    {
      samples.clear();
      return false;
    }
  }

  // Copy into the return array
  const float scale = 1.0 / ((double)(1 << 23));
  samples.resize(tmp.size());
  for (size_t i = 0; i < samples.size(); i++)
    samples[i] = scale * ((float)tmp[i]);
  return true;
}

bool dsp::wav::_ReadSigned24BitInt(std::ifstream& stream, int& value)
{
  // Read the three bytes of the 24-bit integer.
  std::uint8_t bytes[3] = {};
  value = 0;
  stream.read(reinterpret_cast<char*>(bytes), 3);
  if (!stream)
    return false;

  // Combine the three bytes into a single integer using bit shifting and
  // masking. This works by isolating each byte using a bit mask (0xff) and then
  // shifting the byte to the correct position in the final integer.
  value = bytes[0] | (bytes[1] << 8) | (bytes[2] << 16);

  // The value is stored in two's complement format, so if the most significant
  // bit (the 24th bit) is set, then the value is negative. In this case, we
  // need to extend the sign bit to get the correct negative value.
  if (value & (1 << 23))
  {
    value |= ~((1 << 24) - 1);
  }

  return true;
}

void dsp::wav::_LoadSamples32FloatingPoint(std::ifstream& wavFile, const int chunkSize, std::vector<float>& samples)
{
  if (chunkSize <= 0 || chunkSize % 4 != 0)
    return;
  // NOTE: 32-bit is float.
  samples.resize(chunkSize / 4); // 32 bits (4 bytes) per sample
  // Read the samples from the file into the array
  wavFile.read(reinterpret_cast<char*>(samples.data()), chunkSize);
}

void dsp::wav::_LoadSamples32FixedPoint(std::ifstream& wavFile, const int chunkSize, std::vector<float>& samples)
{
  if (chunkSize <= 0 || chunkSize % 4 != 0)
    return;
  // Allocate an array to hold the samples
  std::vector<int> tmp(chunkSize / 4); // 32 bits (4 bytes) per sample

  // Read the samples from the file into the array
  wavFile.read(reinterpret_cast<char*>(tmp.data()), chunkSize);

  // Copy into the return array
  const float scale = 1.0 / 2147483648.0; // 2^31 for 32-bit fixed point
  samples.resize(tmp.size());
  for (size_t i = 0; i < samples.size(); i++)
    samples[i] = scale * ((float)tmp[i]);
}
