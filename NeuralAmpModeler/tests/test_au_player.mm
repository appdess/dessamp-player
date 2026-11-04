#include <AudioToolbox/AudioToolbox.h>
#include <AudioUnit/AudioUnit.h>
#include <CoreFoundation/CoreFoundation.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <dlfcn.h>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "NAM/player_manifest.h"
#include "../DessPresetFile.h"

namespace
{
constexpr UInt32 kDessAmpSubtype = 'DsAP';
constexpr UInt32 kDessAmpManufacturer = 'AdMs';
constexpr AudioUnitParameterID kFirstPedalLevel = 29;
constexpr std::size_t kParameterCount = 33;
constexpr AudioUnitParameterID kInputLevelParameter = 0;
constexpr AudioUnitParameterID kCalibrateInputParameter = 15;
constexpr AudioUnitParameterID kInputCalibrationLevelParameter = 16;
constexpr AudioUnitParameterID kAmpEnabled = 19;
constexpr AudioUnitParameterID kPlayerMode = 20;
constexpr AudioUnitParameterID kFirstPlayerCondition = 21;
constexpr AudioUnitParameterID kIRToggleParameter = 13;
constexpr AudioUnitParameterID kBoostToneParameter = 8;
constexpr AudioUnitParameterID kBoostActiveParameter = 11;
constexpr AudioUnitParameterID kBoostModelParameter = 14;
constexpr double kSampleRate = 48000.0;
constexpr UInt32 kBlockSize = 64;

void RequireStatus(const OSStatus status, const char* operation)
{
  if (status != noErr)
    throw std::runtime_error(std::string(operation) + " failed with OSStatus " + std::to_string(status));
}

template <typename T>
void AppendNative(std::vector<std::uint8_t>& bytes, const T& value)
{
  const auto offset = bytes.size();
  bytes.resize(offset + sizeof(T));
  std::memcpy(bytes.data() + offset, &value, sizeof(T));
}

void AppendString(std::vector<std::uint8_t>& bytes, const std::string& value)
{
  const auto length = static_cast<std::int32_t>(value.size());
  AppendNative(bytes, length);
  bytes.insert(bytes.end(), value.begin(), value.end());
}

std::vector<std::uint8_t> MakeStateChunk(const std::string& manifestPath, const std::string& irPath,
                                         const std::array<double, kParameterCount>& parameters)
{
  std::vector<std::uint8_t> bytes;
  AppendString(bytes, "###NeuralAmpModeler###");
  AppendString(bytes, "0.2.0");
  AppendString(bytes, manifestPath);
  AppendString(bytes, irPath);
  for (const double parameter : parameters)
    AppendNative(bytes, parameter);
  return bytes;
}

CFStringRef MakeString(const char* value)
{
  return CFStringCreateWithCString(kCFAllocatorDefault, value, kCFStringEncodingUTF8);
}

void SetDictionaryNumber(CFMutableDictionaryRef dictionary, const char* key, const SInt32 value)
{
  CFStringRef keyString = MakeString(key);
  CFNumberRef number = CFNumberCreate(kCFAllocatorDefault, kCFNumberSInt32Type, &value);
  CFDictionarySetValue(dictionary, keyString, number);
  CFRelease(number);
  CFRelease(keyString);
}

void SetDictionaryString(CFMutableDictionaryRef dictionary, const char* key, const char* value)
{
  CFStringRef keyString = MakeString(key);
  CFStringRef stringValue = MakeString(value);
  CFDictionarySetValue(dictionary, keyString, stringValue);
  CFRelease(stringValue);
  CFRelease(keyString);
}

CFMutableDictionaryRef MakeStateDictionary(const std::string& manifestPath, const std::string& irPath,
                                           const std::array<double, kParameterCount>& parameters)
{
  const auto bytes = MakeStateChunk(manifestPath, irPath, parameters);
  CFMutableDictionaryRef dictionary =
    CFDictionaryCreateMutable(kCFAllocatorDefault, 0, &kCFTypeDictionaryKeyCallBacks, &kCFTypeDictionaryValueCallBacks);
  SetDictionaryNumber(dictionary, kAUPresetVersionKey, 0x000200);
  SetDictionaryNumber(dictionary, kAUPresetTypeKey, static_cast<SInt32>(kAudioUnitType_Effect));
  SetDictionaryNumber(dictionary, kAUPresetSubtypeKey, static_cast<SInt32>(kDessAmpSubtype));
  SetDictionaryNumber(dictionary, kAUPresetManufacturerKey, static_cast<SInt32>(kDessAmpManufacturer));
  SetDictionaryString(dictionary, kAUPresetNameKey, "Empty");
  CFDataRef data = CFDataCreate(kCFAllocatorDefault, bytes.data(), static_cast<CFIndex>(bytes.size()));
  CFStringRef dataKey = MakeString(kAUPresetDataKey);
  CFDictionarySetValue(dictionary, dataKey, data);
  CFRelease(dataKey);
  CFRelease(data);
  return dictionary;
}

AudioComponentDescription MakeDescription()
{
  AudioComponentDescription description{};
  description.componentType = kAudioUnitType_Effect;
  description.componentSubType = kDessAmpSubtype;
  description.componentManufacturer = kDessAmpManufacturer;
  return description;
}

AudioComponent RegisterComponentBinary(const char* binaryPath, void*& libraryHandle)
{
  libraryHandle = dlopen(binaryPath, RTLD_NOW | RTLD_LOCAL);
  if (libraryHandle == nullptr)
    throw std::runtime_error(std::string("Could not load staged AU binary: ") + dlerror());
  dlerror();
  void* symbol = dlsym(libraryHandle, "DessAmp_Factory");
  if (const char* error = dlerror())
    throw std::runtime_error(std::string("Could not find DessAmp_Factory: ") + error);
  const auto factory = reinterpret_cast<AudioComponentFactoryFunction>(symbol);
  auto description = MakeDescription();
  description.componentFlags = kAudioComponentFlag_SandboxSafe;
  const AudioComponent component =
    AudioComponentRegister(&description, CFSTR("DessAmp staged integration test"), 0x000200, factory);
  if (component == nullptr)
    throw std::runtime_error("AudioComponentRegister returned no staged component");
  return component;
}

AudioUnit CreateUnit(AudioComponent component)
{
  if (component == nullptr)
  {
    auto description = MakeDescription();
    component = AudioComponentFindNext(nullptr, &description);
  }
  if (component == nullptr)
    throw std::runtime_error("DessAmp AU is not registered");
  AudioUnit unit = nullptr;
  RequireStatus(AudioComponentInstanceNew(component, &unit), "AudioComponentInstanceNew");
  if (unit == nullptr)
    throw std::runtime_error("AudioComponentInstanceNew returned no unit");
  return unit;
}

void ApplyState(AudioUnit unit, CFPropertyListRef state)
{
  RequireStatus(
    AudioUnitSetProperty(unit, kAudioUnitProperty_ClassInfo, kAudioUnitScope_Global, 0, &state, sizeof(state)),
    "AudioUnitSetProperty(ClassInfo)");
}

CFPropertyListRef CopyState(AudioUnit unit)
{
  CFPropertyListRef state = nullptr;
  UInt32 size = sizeof(state);
  RequireStatus(AudioUnitGetProperty(unit, kAudioUnitProperty_ClassInfo, kAudioUnitScope_Global, 0, &state, &size),
                "AudioUnitGetProperty(ClassInfo)");
  if (state == nullptr || CFGetTypeID(state) != CFDictionaryGetTypeID())
    throw std::runtime_error("DessAmp returned no class-info dictionary");
  return state;
}

// Reload the exact AU state bytes via the same disk format used by Save/New.
void ApplyPedalPreset(AudioUnit unit, CFPropertyListRef state)
{
  const auto dictionary = static_cast<CFDictionaryRef>(state);
  const auto key = CFSTR("data");
  const auto bytes = static_cast<CFDataRef>(CFDictionaryGetValue(dictionary, key));
  if (!bytes || CFGetTypeID(bytes) != CFDataGetTypeID())
    throw std::runtime_error("Missing session bytes for pedal preset");
  std::filesystem::path path;
  std::string error;
  const auto requested = std::filesystem::temp_directory_path() / ("dessamp-au-pedals-" + std::to_string(getpid()));
  if (!dessamp::preset::WriteAtomic(requested, CFDataGetBytePtr(bytes), CFDataGetLength(bytes), path, error))
    throw std::runtime_error(error);
  const auto decoded = dessamp::preset::Read(path);
  std::filesystem::remove(path);
  if (!decoded) throw std::runtime_error(decoded.error);
  auto recalled = CFDictionaryCreateMutableCopy(nullptr, 0, dictionary);
  auto data = CFDataCreate(nullptr, decoded.state.data(), decoded.state.size());
  CFDictionarySetValue(recalled, key, data);
  CFRelease(data);
  ApplyState(unit, recalled);
  CFRelease(recalled);
}

std::string ReadString(const std::uint8_t* bytes, const std::size_t size, std::size_t& offset)
{
  if (offset + sizeof(std::int32_t) > size)
    throw std::runtime_error("Serialized state is truncated before a string length");
  std::int32_t length = 0;
  std::memcpy(&length, bytes + offset, sizeof(length));
  offset += sizeof(length);
  if (length < 0 || static_cast<std::size_t>(length) > size - offset)
    throw std::runtime_error("Serialized state contains an invalid string length");
  std::string value(reinterpret_cast<const char*>(bytes + offset), static_cast<std::size_t>(length));
  offset += static_cast<std::size_t>(length);
  return value;
}

void VerifySerializedState(CFPropertyListRef state, const std::string& expectedManifest, const std::string& expectedIR,
                           const std::array<double, kParameterCount>& expectedParameters)
{
  const auto dictionary = static_cast<CFDictionaryRef>(state);
  CFStringRef dataKey = MakeString(kAUPresetDataKey);
  const auto data = static_cast<CFDataRef>(CFDictionaryGetValue(dictionary, dataKey));
  CFRelease(dataKey);
  if (data == nullptr || CFGetTypeID(data) != CFDataGetTypeID())
    throw std::runtime_error("DessAmp class info contains no serialized data");
  const auto* bytes = reinterpret_cast<const std::uint8_t*>(CFDataGetBytePtr(data));
  const auto size = static_cast<std::size_t>(CFDataGetLength(data));
  std::size_t offset = 0;
  if (ReadString(bytes, size, offset) != "###NeuralAmpModeler###")
    throw std::runtime_error("Serialized state header changed");
  if (ReadString(bytes, size, offset) != "0.2.0")
    throw std::runtime_error("Serialized state version is not 0.2.0");
  if (ReadString(bytes, size, offset) != expectedManifest)
    throw std::runtime_error("Player manifest path did not round-trip");
  if (ReadString(bytes, size, offset) != expectedIR)
    throw std::runtime_error("Cabinet IR path did not round-trip");
  for (std::size_t index = 0; index < expectedParameters.size(); ++index)
  {
    if (offset + sizeof(double) > size)
      throw std::runtime_error("Serialized state is truncated in its parameter payload");
    double actual = 0.0;
    std::memcpy(&actual, bytes + offset, sizeof(actual));
    offset += sizeof(actual);
    if (!std::isfinite(actual) || std::abs(actual - expectedParameters[index]) > 1.0e-9)
      throw std::runtime_error("Serialized parameter " + std::to_string(index) + " did not round-trip");
  }
  if (offset != size)
    throw std::runtime_error("Serialized AU state contains unexpected trailing bytes");
}

void VerifyHostParameters(AudioUnit unit, const std::array<double, kParameterCount>& expected)
{
  for (AudioUnitParameterID parameter = 0; parameter < expected.size(); ++parameter)
  {
    AudioUnitParameterValue actual = 0.0f;
    RequireStatus(AudioUnitGetParameter(unit, parameter, kAudioUnitScope_Global, 0, &actual), "AudioUnitGetParameter");
    if (std::abs(static_cast<double>(actual) - expected[parameter]) > 1.0e-4)
      throw std::runtime_error("Host parameter " + std::to_string(parameter) + " did not round-trip");
  }
}

std::array<double, kParameterCount> MakeParameters(const double conditionPercent)
{
  std::array<double, kParameterCount> parameters{
    -3.0, 5.0, -70.0, 4.0, 6.0, 7.0, 1.5,  2.0,  7.0,  1.0,  0.0,  0.0,  5.0,  0.0,  2.0,
    0.0, 12.0, 0.0,   0.0, 1.0, 1.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0,
    -4.0, 2.0, 3.0, -1.0,
  };
  for (std::size_t index = kFirstPlayerCondition; index < kFirstPedalLevel; ++index)
    parameters[index] = conditionPercent;
  return parameters;
}

struct InputState
{
  std::uint64_t frame = 0;
};

OSStatus InputCallback(void* refCon, AudioUnitRenderActionFlags*, const AudioTimeStamp*, UInt32, UInt32 frames,
                       AudioBufferList* data)
{
  auto& state = *static_cast<InputState*>(refCon);
  if (data == nullptr)
    return kAudio_ParamError;
  constexpr double pi = 3.14159265358979323846;
  for (UInt32 bufferIndex = 0; bufferIndex < data->mNumberBuffers; ++bufferIndex)
  {
    auto& buffer = data->mBuffers[bufferIndex];
    if (buffer.mData == nullptr)
      return kAudio_ParamError;
    auto* samples = static_cast<Float32*>(buffer.mData);
    for (UInt32 index = 0; index < frames; ++index)
    {
      const double time = static_cast<double>(state.frame + index) / kSampleRate;
      const double pulse = ((state.frame + index) % 521U) == 0U ? 0.08 : 0.0;
      samples[index] = static_cast<Float32>(0.045 * std::sin(2.0 * pi * 97.0 * time)
                                            + 0.025 * std::sin(2.0 * pi * 431.0 * time) + pulse);
    }
    buffer.mDataByteSize = frames * sizeof(Float32);
  }
  state.frame += frames;
  return noErr;
}

void ConfigureUnit(AudioUnit unit, InputState& inputState)
{
  AudioStreamBasicDescription format{};
  format.mSampleRate = kSampleRate;
  format.mFormatID = kAudioFormatLinearPCM;
  format.mFormatFlags = kAudioFormatFlagsNativeFloatPacked | kAudioFormatFlagIsNonInterleaved;
  format.mBytesPerPacket = sizeof(Float32);
  format.mFramesPerPacket = 1;
  format.mBytesPerFrame = sizeof(Float32);
  format.mChannelsPerFrame = 1;
  format.mBitsPerChannel = 8U * sizeof(Float32);
  RequireStatus(
    AudioUnitSetProperty(unit, kAudioUnitProperty_StreamFormat, kAudioUnitScope_Input, 0, &format, sizeof(format)),
    "set input stream format");
  RequireStatus(
    AudioUnitSetProperty(unit, kAudioUnitProperty_StreamFormat, kAudioUnitScope_Output, 0, &format, sizeof(format)),
    "set output stream format");
  UInt32 maximumFrames = kBlockSize;
  RequireStatus(AudioUnitSetProperty(unit, kAudioUnitProperty_MaximumFramesPerSlice, kAudioUnitScope_Global, 0,
                                     &maximumFrames, sizeof(maximumFrames)),
                "set maximum frames per slice");
  AURenderCallbackStruct callback{InputCallback, &inputState};
  RequireStatus(AudioUnitSetProperty(
                  unit, kAudioUnitProperty_SetRenderCallback, kAudioUnitScope_Input, 0, &callback, sizeof(callback)),
                "set input render callback");
}

void PumpMainRunLoop()
{
  const CFAbsoluteTime deadline = CFAbsoluteTimeGetCurrent() + 1.5;
  while (CFAbsoluteTimeGetCurrent() < deadline)
    CFRunLoopRunInMode(kCFRunLoopDefaultMode, 0.025, true);
}

std::vector<double> RenderRealtime(AudioUnit unit, const std::size_t totalFrames,
                                   const std::array<double, 8>* liveConditions = nullptr,
                                   const bool varyFrames = false, const double* liveBoostTone = nullptr)
{
  InputState inputState;
  ConfigureUnit(unit, inputState);
  RequireStatus(AudioUnitInitialize(unit), "AudioUnitInitialize");
  PumpMainRunLoop();

  AudioUnitParameterValue ampEnabled = 0.0f;
  AudioUnitParameterValue playerMode = 0.0f;
  RequireStatus(AudioUnitGetParameter(unit, kAmpEnabled, kAudioUnitScope_Global, 0, &ampEnabled), "read Amp Enabled");
  RequireStatus(AudioUnitGetParameter(unit, kPlayerMode, kAudioUnitScope_Global, 0, &playerMode), "read Player Mode");
  if (ampEnabled < 0.5f || playerMode < 0.5f)
    throw std::runtime_error("Conditioned player was disabled while loading its manifest");
  if (liveConditions != nullptr)
  {
    for (AudioUnitParameterID parameter = kFirstPlayerCondition; parameter < kFirstPedalLevel; ++parameter)
      RequireStatus(AudioUnitSetParameter(
                      unit, parameter, kAudioUnitScope_Global, 0,
                      static_cast<AudioUnitParameterValue>(liveConditions->at(parameter - kFirstPlayerCondition)), 0),
                    "AudioUnitSetParameter(Player Condition)");
  }

  if (liveBoostTone != nullptr)
    RequireStatus(AudioUnitSetParameter(unit, kBoostToneParameter, kAudioUnitScope_Global, 0,
                                        static_cast<AudioUnitParameterValue>(*liveBoostTone), 0), "set live Boost Tone");

  std::vector<double> result;
  result.reserve(totalFrames);
  std::array<Float32, kBlockSize> block{};
  AudioTimeStamp timestamp{};
  timestamp.mFlags = kAudioTimeStampSampleTimeValid;
  std::size_t blockIndex = 0;
  while (result.size() < totalFrames)
  {
    constexpr std::array<UInt32, 4> sizes{{kBlockSize, 17, 31, 1}};
    const UInt32 requested = varyFrames ? sizes[blockIndex++ % sizes.size()] : kBlockSize;
    const UInt32 frames = static_cast<UInt32>(std::min<std::size_t>(requested, totalFrames - result.size()));
    AudioBufferList output{};
    output.mNumberBuffers = 1;
    output.mBuffers[0].mNumberChannels = 1;
    output.mBuffers[0].mDataByteSize = frames * sizeof(Float32);
    output.mBuffers[0].mData = block.data();
    AudioUnitRenderActionFlags flags = 0;
    timestamp.mSampleTime = static_cast<Float64>(result.size());
    RequireStatus(AudioUnitRender(unit, &flags, &timestamp, 0, frames, &output), "AudioUnitRender");
    for (UInt32 index = 0; index < frames; ++index)
    {
      if (!std::isfinite(block[index]))
        throw std::runtime_error("Conditioned AU produced non-finite audio");
      result.push_back(block[index]);
    }
  }
  RequireStatus(AudioUnitUninitialize(unit), "AudioUnitUninitialize");
  return result;
}

std::vector<double> RenderState(AudioComponent component, const std::string& manifestPath, const std::string& irPath,
                                const std::array<double, kParameterCount>& parameters, bool varyFrames = false)
{
  AudioUnit unit = nullptr;
  CFMutableDictionaryRef state = nullptr;
  try
  {
    unit = CreateUnit(component);
    state = MakeStateDictionary(manifestPath, irPath, parameters);
    ApplyState(unit, state);
    VerifyHostParameters(unit, parameters);
    auto audio = RenderRealtime(unit, 48000, nullptr, varyFrames);
    CFRelease(state);
    AudioComponentInstanceDispose(unit);
    return audio;
  }
  catch (...)
  {
    if (state != nullptr)
      CFRelease(state);
    if (unit != nullptr)
      AudioComponentInstanceDispose(unit);
    throw;
  }
}

double RootMeanSquare(const std::vector<double>& values)
{
  long double sum = 0.0;
  for (const double value : values)
    sum += static_cast<long double>(value) * value;
  return std::sqrt(static_cast<double>(sum / std::max<std::size_t>(values.size(), 1)));
}

double DifferenceRMS(const std::vector<double>& first, const std::vector<double>& second)
{
  if (first.size() != second.size())
    throw std::runtime_error("Render sizes differ");
  long double sum = 0.0;
  for (std::size_t index = 0; index < first.size(); ++index)
  {
    const double difference = first[index] - second[index];
    sum += static_cast<long double>(difference) * difference;
  }
  return std::sqrt(static_cast<double>(sum / std::max<std::size_t>(first.size(), 1)));
}

int FindMorphConditionIndex(const nam::player::Manifest& manifest)
{
  int learnedMorphIndex = -1;
  for (const auto& control : manifest.controls)
  {
    if (control.id != "morph" || control.model_parameter_index >= manifest.global_condition_size)
      continue;
    if (learnedMorphIndex >= 0)
      throw std::runtime_error("Manifest exposes more than one learned Morph condition");
    if (control.model_parameter_index < 0 || control.IsSwitch() || control.minimum != 0.0 || control.maximum != 1.0)
      throw std::runtime_error("Learned Morph must be one continuous [0, 1] model condition");
    learnedMorphIndex = control.model_parameter_index;
  }
  if (manifest.morph.enabled && learnedMorphIndex >= 0)
    throw std::runtime_error("Manifest combines learned Morph with the dual-model Morph slot");
  return manifest.morph.enabled ? manifest.global_condition_size : learnedMorphIndex;
}
} // namespace

int main(int argc, char* argv[])
{
  if (argc < 2)
  {
    std::cerr << "usage: test_au_player <model.dess|player-manifest.json> [--snapshot] [--test-boost-bank] [--test-verdant-tone] "
                 "[--expect-control-label index=label] "
                 "[--expect-input-calibration-reference-dbu value] "
                 "[--ir-path /absolute/cabinet.wav] "
                 "[--component-binary /absolute/DessAmp]\n";
    return 2;
  }

  bool snapshotMode = false;
  bool testBoostBank = false;
  bool testVerdantTone = false;
  bool testInputCalibration = false;
  double expectedInputCalibrationReferenceDBU = 0.0;
  const char* componentBinary = nullptr;
  std::string irPath;
  std::vector<std::pair<int, std::string>> expectedControlLabels;
  for (int index = 2; index < argc; ++index)
  {
    const std::string option(argv[index]);
    if (option == "--snapshot" && !snapshotMode)
      snapshotMode = true;
    else if (option == "--test-verdant-tone")
      testVerdantTone = true;
    else if (option == "--test-boost-bank" && !testBoostBank)
      testBoostBank = true;
    else if (option == "--expect-input-calibration-reference-dbu" && !testInputCalibration && index + 1 < argc)
    {
      try
      {
        expectedInputCalibrationReferenceDBU = std::stod(argv[++index]);
        if (!std::isfinite(expectedInputCalibrationReferenceDBU))
          throw std::runtime_error("non-finite");
        testInputCalibration = true;
      }
      catch (const std::exception&)
      {
        std::cerr << "Invalid input-calibration reference\n";
        return 2;
      }
    }
    else if (option == "--expect-control-label" && index + 1 < argc)
    {
      const std::string expectation(argv[++index]);
      const auto separator = expectation.find('=');
      if (separator == std::string::npos || separator == 0U || separator + 1U >= expectation.size())
      {
        std::cerr << "Invalid control-label expectation: " << expectation << '\n';
        return 2;
      }
      try
      {
        const int controlIndex = std::stoi(expectation.substr(0U, separator));
        expectedControlLabels.emplace_back(controlIndex, expectation.substr(separator + 1U));
      }
      catch (const std::exception&)
      {
        std::cerr << "Invalid control-label expectation: " << expectation << '\n';
        return 2;
      }
    }
    else if (option == "--component-binary" && componentBinary == nullptr && index + 1 < argc)
      componentBinary = argv[++index];
    else if (option == "--ir-path" && irPath.empty() && index + 1 < argc)
      irPath = argv[++index];
    else
    {
      std::cerr << "Invalid or duplicate option: " << option << '\n';
      return 2;
    }
  }

  AudioUnit lowUnit = nullptr;
  AudioUnit restoredUnit = nullptr;
  AudioUnit highUnit = nullptr;
  AudioUnit cabBypassUnit = nullptr;
  AudioComponent stagedComponent = nullptr;
  void* stagedLibrary = nullptr;
  CFMutableDictionaryRef lowInputState = nullptr;
  CFMutableDictionaryRef cabBypassInputState = nullptr;
  CFPropertyListRef savedState = nullptr;
  try
  {
    const std::string manifestPath(argv[1]);
    const auto manifest = nam::player::LoadManifest(manifestPath, 8);
    for (const auto& expectation : expectedControlLabels)
    {
      if (expectation.first < 0 || expectation.first >= static_cast<int>(manifest.controls.size()))
        throw std::runtime_error("Expected control-label index is outside the manifest vector");
      const auto& actual = manifest.controls[static_cast<std::size_t>(expectation.first)].label;
      if (actual != expectation.second)
        throw std::runtime_error("Manifest control label differs from the expected arbitrary label");
    }
    if (snapshotMode != (manifest.global_condition_size == 0))
      throw std::runtime_error("--snapshot disagrees with the manifest parameter mode");
    if (componentBinary != nullptr)
      stagedComponent = RegisterComponentBinary(componentBinary, stagedLibrary);
    auto lowParameters = MakeParameters(0.0);
    auto highParameters = MakeParameters(100.0);
    if (!irPath.empty())
    {
      lowParameters[kIRToggleParameter] = 1.0;
      highParameters[kIRToggleParameter] = 1.0;
    }
    std::array<double, 8> highConditions{};
    highConditions.fill(100.0);

    lowUnit = CreateUnit(stagedComponent);
    lowInputState = MakeStateDictionary(manifestPath, irPath, lowParameters);
    ApplyState(lowUnit, lowInputState);
    VerifyHostParameters(lowUnit, lowParameters);
    savedState = CopyState(lowUnit);
    VerifySerializedState(savedState, manifestPath, irPath, lowParameters);
    const auto lowAudio = RenderRealtime(lowUnit, 48000);
    const auto variableAudio = RenderState(stagedComponent, manifestPath, irPath, lowParameters, true);
    if (DifferenceRMS(lowAudio, variableAudio) > std::max(1e-8, RootMeanSquare(lowAudio) * 1e-5))
      throw std::runtime_error("Variable host block sizes changed the rendered signal");

    restoredUnit = CreateUnit(stagedComponent);
    ApplyState(restoredUnit, savedState);
    VerifyHostParameters(restoredUnit, lowParameters);
    const auto restoredAudio = RenderRealtime(restoredUnit, 48000);

    highUnit = CreateUnit(stagedComponent);
    ApplyState(highUnit, lowInputState);
    const auto highAudio = RenderRealtime(highUnit, 48000, &highConditions);
    VerifyHostParameters(highUnit, highParameters);

    double cabinetDifference = 0.0;
    if (!irPath.empty())
    {
      auto cabBypassParameters = lowParameters;
      cabBypassParameters[kIRToggleParameter] = 0.0;
      cabBypassUnit = CreateUnit(stagedComponent);
      cabBypassInputState = MakeStateDictionary(manifestPath, irPath, cabBypassParameters);
      ApplyState(cabBypassUnit, cabBypassInputState);
      const auto cabBypassAudio = RenderRealtime(cabBypassUnit, 48000);
      VerifyHostParameters(cabBypassUnit, cabBypassParameters);
      cabinetDifference = DifferenceRMS(lowAudio, cabBypassAudio);
      if (cabinetDifference <= 1.0e-7)
        throw std::runtime_error("Cabinet IR enable/bypass did not change exact-AU output");
    }

    const double lowRMS = RootMeanSquare(lowAudio);
    const double highRMS = RootMeanSquare(highAudio);
    const double recallDifference = DifferenceRMS(lowAudio, restoredAudio);
    const double conditionDifference = DifferenceRMS(lowAudio, highAudio);
    if (lowRMS <= 1.0e-6 || highRMS <= 1.0e-6)
      throw std::runtime_error("Conditioned AU render is silent");
    if (recallDifference > 1.0e-6)
      throw std::runtime_error("Reopened AU state did not reproduce the same deterministic render");
    if (!snapshotMode && conditionDifference <= 1.0e-5)
      throw std::runtime_error("Changing all learned controls did not change the AU render");
    if (snapshotMode && conditionDifference > 1.0e-6)
      throw std::runtime_error("Snapshot model unexpectedly responded to inactive condition parameters");

    double inputCalibrationDifference = 0.0;
    double inputCalibrationEquivalentDifference = 0.0;
    if (testInputCalibration)
    {
      constexpr double interfaceInputLevelDBU = 16.0;
      auto calibratedParameters = lowParameters;
      calibratedParameters[kCalibrateInputParameter] = 1.0;
      calibratedParameters[kInputCalibrationLevelParameter] = interfaceInputLevelDBU;
      auto equivalentParameters = lowParameters;
      equivalentParameters[kInputLevelParameter] +=
        interfaceInputLevelDBU - expectedInputCalibrationReferenceDBU;
      const auto calibratedAudio =
        RenderState(stagedComponent, manifestPath, irPath, calibratedParameters);
      const auto equivalentAudio =
        RenderState(stagedComponent, manifestPath, irPath, equivalentParameters);
      inputCalibrationDifference = DifferenceRMS(lowAudio, calibratedAudio);
      inputCalibrationEquivalentDifference = DifferenceRMS(calibratedAudio, equivalentAudio);
      if (inputCalibrationDifference <= 1.0e-7)
        throw std::runtime_error("Input calibration did not change exact-AU output");
      if (inputCalibrationEquivalentDifference > 1.0e-6)
        throw std::runtime_error("Input calibration does not equal the expected model-reference gain correction");
    }

    std::array<double, 4> boostRMS{};
    std::array<double, 4> boostDifference{};
    std::array<double, 4> boostRecallDifference{};
    if (testBoostBank || testVerdantTone)
    {
      for (int boostIndex = 0; boostIndex < 4; ++boostIndex)
      {
        AudioUnit boostUnit = nullptr;
        AudioUnit boostRestoredUnit = nullptr;
        CFMutableDictionaryRef boostInputState = nullptr;
        CFPropertyListRef boostSavedState = nullptr;
        try
        {
          auto boostParameters = lowParameters;
          boostParameters[kBoostActiveParameter] = 1.0;
          boostParameters[kBoostModelParameter] = static_cast<double>(boostIndex);
          boostUnit = CreateUnit(stagedComponent);
          boostInputState = MakeStateDictionary(manifestPath, irPath, boostParameters);
          ApplyState(boostUnit, boostInputState);
          const double highTone = 10.0;
          const auto boostAudio = RenderRealtime(boostUnit, 48000, nullptr, false, testVerdantTone ? &highTone : nullptr);
          if (testVerdantTone)
          {
            boostParameters[kBoostToneParameter] = 0.0;
            const auto minimum = RenderState(stagedComponent, manifestPath, irPath, boostParameters);
            const double delta = DifferenceRMS(minimum, boostAudio);
            std::cout << "tone_slot=" << boostIndex << " min_max_diff_rms=" << delta << '\n';
            if (boostIndex == 2 ? delta <= 1.e-5 : delta > 1.e-7)
              throw std::runtime_error("Unexpected response to live boost Tone");
            boostParameters[kBoostToneParameter] = highTone;
          }
          VerifyHostParameters(boostUnit, boostParameters);
          auto otherLevels = boostParameters;
          for (int slot = 0; slot < 4; ++slot)
            if (slot != boostIndex) otherLevels[kFirstPedalLevel + slot] += 8.0;
          const auto otherAudio = RenderState(stagedComponent, manifestPath, irPath, otherLevels);
          otherLevels[kFirstPedalLevel + boostIndex] -= 8.0;
          const auto ownAudio = RenderState(stagedComponent, manifestPath, irPath, otherLevels);
          if (DifferenceRMS(boostAudio, otherAudio) > 1.e-7 || DifferenceRMS(boostAudio, ownAudio) <= 1.e-6)
            throw std::runtime_error("Pedal Level failed isolation or active gain check for slot " + std::to_string(boostIndex));
          boostSavedState = CopyState(boostUnit);
          VerifySerializedState(boostSavedState, manifestPath, irPath, boostParameters);
          boostRestoredUnit = CreateUnit(stagedComponent);
          ApplyPedalPreset(boostRestoredUnit, boostSavedState);
          VerifyHostParameters(boostRestoredUnit, boostParameters);
          const auto boostRestoredAudio = RenderRealtime(boostRestoredUnit, 48000);
          boostRMS[static_cast<std::size_t>(boostIndex)] = RootMeanSquare(boostAudio);
          boostDifference[static_cast<std::size_t>(boostIndex)] = DifferenceRMS(lowAudio, boostAudio);
          boostRecallDifference[static_cast<std::size_t>(boostIndex)] =
            DifferenceRMS(boostAudio, boostRestoredAudio);
          if (boostRMS[static_cast<std::size_t>(boostIndex)] <= 1.0e-6)
            throw std::runtime_error("Boost slot " + std::to_string(boostIndex) + " rendered silence");
          if (boostDifference[static_cast<std::size_t>(boostIndex)] <= 1.0e-7)
            throw std::runtime_error("Boost slot " + std::to_string(boostIndex) + " did not change exact-AU output");
          if (boostRecallDifference[static_cast<std::size_t>(boostIndex)] > 1.0e-6)
            throw std::runtime_error("Boost slot " + std::to_string(boostIndex)
                                     + " did not survive an exact state reopen");
        }
        catch (...)
        {
          if (boostSavedState != nullptr)
            CFRelease(boostSavedState);
          if (boostInputState != nullptr)
            CFRelease(boostInputState);
          if (boostUnit != nullptr)
            AudioComponentInstanceDispose(boostUnit);
          if (boostRestoredUnit != nullptr)
            AudioComponentInstanceDispose(boostRestoredUnit);
          throw;
        }
        CFRelease(boostSavedState);
        CFRelease(boostInputState);
        AudioComponentInstanceDispose(boostUnit);
        AudioComponentInstanceDispose(boostRestoredUnit);
      }
      auto bypassed = lowParameters;
      std::fill(bypassed.begin() + kFirstPedalLevel, bypassed.end(), 16.0);
      bypassed[kBoostToneParameter] = 10.0;
      if (DifferenceRMS(lowAudio, RenderState(stagedComponent, manifestPath, irPath, bypassed)) > 1.e-7)
        throw std::runtime_error("Pedal levels or Tone colored the bypass path");
    }

    double morphDifference = 0.0;
    const int resolvedMorphConditionIndex = FindMorphConditionIndex(manifest);
    if (resolvedMorphConditionIndex >= 0)
    {
      const auto morphConditionIndex = static_cast<std::size_t>(resolvedMorphConditionIndex);
      if (morphConditionIndex >= highConditions.size())
        throw std::runtime_error("Morph has no valid learned host slot");
      std::array<double, 8> morphConditions{};
      morphConditions.fill(0.0);
      morphConditions[morphConditionIndex] = 100.0;
      auto morphParameters = lowParameters;
      morphParameters[kFirstPlayerCondition + morphConditionIndex] = 100.0;

      // Morph must remain continuous at host precision. A whole-percentage
      // parameter step makes slow automation visibly and audibly staircase,
      // even though the DSP itself smooths each target transition.
      constexpr AudioUnitParameterValue fineMorphValue = 50.37f;
      const auto morphParameterID =
        static_cast<AudioUnitParameterID>(kFirstPlayerCondition + morphConditionIndex);
      RequireStatus(AudioUnitSetParameter(
                      highUnit, morphParameterID, kAudioUnitScope_Global, 0, fineMorphValue, 0),
                    "AudioUnitSetParameter fine Morph");
      AudioUnitParameterValue fineMorphReadback = 0.0f;
      RequireStatus(AudioUnitGetParameter(
                      highUnit, morphParameterID, kAudioUnitScope_Global, 0, &fineMorphReadback),
                    "AudioUnitGetParameter fine Morph");
      if (std::abs(fineMorphReadback - fineMorphValue) > 0.005f)
        throw std::runtime_error("Morph host parameter was quantized more coarsely than 0.01 percent");

      ApplyState(highUnit, lowInputState);
      const auto morphAudio = RenderRealtime(highUnit, 48000, &morphConditions);
      VerifyHostParameters(highUnit, morphParameters);
      morphDifference = DifferenceRMS(lowAudio, morphAudio);
      if (morphDifference <= 1.0e-5)
        throw std::runtime_error("Morph did not change exact-AU output independently");
    }

    std::size_t verifiedSwitches = 0U;
    for (std::size_t conditionIndex = 0U; conditionIndex < manifest.controls.size(); ++conditionIndex)
    {
      const auto& control = manifest.controls[conditionIndex];
      if (!control.IsSwitch())
        continue;
      if (control.model_parameter_index < 0 || control.model_parameter_index >= static_cast<int>(highConditions.size()))
        throw std::runtime_error("Binary control has no valid learned host slot");
      const auto hostConditionIndex = static_cast<std::size_t>(control.model_parameter_index);
      std::array<double, 8> switchConditions{};
      switchConditions.fill(0.0);
      switchConditions[hostConditionIndex] = 100.0;
      auto switchParameters = lowParameters;
      switchParameters[kFirstPlayerCondition + hostConditionIndex] = 100.0;
      ApplyState(highUnit, lowInputState);
      const auto switchAudio = RenderRealtime(highUnit, 48000, &switchConditions);
      VerifyHostParameters(highUnit, switchParameters);
      if (DifferenceRMS(lowAudio, switchAudio) <= 1.0e-7)
        throw std::runtime_error("Binary control " + control.label + " did not change exact-AU output independently");
      ++verifiedSwitches;
    }

    std::cout << "DessAmp AU " << (snapshotMode ? "snapshot" : "conditioned")
              << " player passed: 33-parameter state/path round-trip, headless real-time manifest load, "
                 "finite non-silent audio, deterministic reopen, and "
              << (snapshotMode ? "inactive condition parameters\n" : "live host-parameter condition-dependent output, ")
              << (snapshotMode ? "" : std::to_string(verifiedSwitches) + " independent binary controls")
              << (resolvedMorphConditionIndex >= 0
                    ? manifest.morph.enabled
                        ? ", independent dual-model Morph with 0.01-percent host precision"
                        : ", independent learned Morph with 0.01-percent host precision and no appended slot"
                    : "")
              << (irPath.empty() ? "\n" : ", cabinet IR state/path and audible bypass delta\n");
    std::cout << "low_rms=" << lowRMS << " high_rms=" << highRMS << " recall_diff_rms=" << recallDifference
              << " condition_diff_rms=" << conditionDifference << " morph_diff_rms=" << morphDifference
              << " cabinet_diff_rms=" << cabinetDifference
              << " input_calibration_diff_rms=" << inputCalibrationDifference
              << " input_calibration_equivalent_diff_rms=" << inputCalibrationEquivalentDifference << '\n';
    if (testBoostBank || testVerdantTone)
    {
      for (std::size_t boostIndex = 0; boostIndex < boostRMS.size(); ++boostIndex)
        std::cout << "boost_slot=" << boostIndex << " rms=" << boostRMS[boostIndex]
                  << " dry_diff_rms=" << boostDifference[boostIndex]
                  << " recall_diff_rms=" << boostRecallDifference[boostIndex] << '\n';
    }
  }
  catch (const std::exception& error)
  {
    std::cerr << error.what() << '\n';
    if (savedState != nullptr)
      CFRelease(savedState);
    if (lowInputState != nullptr)
      CFRelease(lowInputState);
    if (cabBypassInputState != nullptr)
      CFRelease(cabBypassInputState);
    if (lowUnit != nullptr)
      AudioComponentInstanceDispose(lowUnit);
    if (restoredUnit != nullptr)
      AudioComponentInstanceDispose(restoredUnit);
    if (highUnit != nullptr)
      AudioComponentInstanceDispose(highUnit);
    if (cabBypassUnit != nullptr)
      AudioComponentInstanceDispose(cabBypassUnit);
    if (stagedLibrary != nullptr)
      dlclose(stagedLibrary);
    return 1;
  }

  CFRelease(savedState);
  CFRelease(lowInputState);
  if (cabBypassInputState != nullptr)
    CFRelease(cabBypassInputState);
  AudioComponentInstanceDispose(lowUnit);
  AudioComponentInstanceDispose(restoredUnit);
  AudioComponentInstanceDispose(highUnit);
  if (cabBypassUnit != nullptr)
    AudioComponentInstanceDispose(cabBypassUnit);
  if (stagedLibrary != nullptr)
    dlclose(stagedLibrary);
  return 0;
}
