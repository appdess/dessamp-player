#include <AudioToolbox/AudioToolbox.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <vector>

namespace
{
constexpr UInt32 kDessMetalSubtype = '1YEo';
constexpr UInt32 kDessMetalManufacturer = 'AdMs';
constexpr UInt32 kFramesPerBlock = 512;
// Exercise the resampling path; 48 kHz would legitimately report zero latency.
constexpr double kSampleRate = 44100.0;
constexpr AudioUnitParameterID kDriveLevelParameter = 7;
constexpr AudioUnitParameterID kDriveActiveParameter = 11;
constexpr AudioUnitParameterID kIRToggleParameter = 13;
constexpr AudioUnitParameterID kDriveModelParameter = 14;
constexpr AudioUnitParameterID kAmpModelParameter = 18;
constexpr AudioUnitParameterID kAmpActiveParameter = 19;

struct InputState
{
  double phase = 0.0;
};

OSStatus RenderInput(void* context, AudioUnitRenderActionFlags*, const AudioTimeStamp*, UInt32, UInt32 frames,
                     AudioBufferList* ioData)
{
  auto& state = *static_cast<InputState*>(context);
  if (ioData == nullptr || ioData->mNumberBuffers == 0 || ioData->mBuffers[0].mData == nullptr)
    return kAudio_ParamError;

  auto* samples = static_cast<float*>(ioData->mBuffers[0].mData);
  for (UInt32 frame = 0; frame < frames; ++frame)
  {
    samples[frame] = static_cast<float>(0.1 * std::sin(state.phase));
    state.phase += 2.0 * M_PI * 220.0 / kSampleRate;
  }
  ioData->mBuffers[0].mDataByteSize = frames * sizeof(float);
  return noErr;
}

bool Check(const OSStatus status, const char* operation)
{
  if (status == noErr)
    return true;
  std::cerr << operation << " failed with OSStatus " << status << '\n';
  return false;
}

AudioStreamBasicDescription MakeFormat(const UInt32 channels)
{
  AudioStreamBasicDescription format{};
  format.mSampleRate = kSampleRate;
  format.mFormatID = kAudioFormatLinearPCM;
  format.mFormatFlags = kAudioFormatFlagIsFloat | kAudioFormatFlagIsNonInterleaved | kAudioFormatFlagsNativeEndian;
  format.mBytesPerPacket = sizeof(float);
  format.mFramesPerPacket = 1;
  format.mBytesPerFrame = sizeof(float);
  format.mChannelsPerFrame = channels;
  format.mBitsPerChannel = 8 * sizeof(float);
  return format;
}
} // namespace

int main()
{
  AudioComponentDescription description{};
  description.componentType = kAudioUnitType_Effect;
  description.componentSubType = kDessMetalSubtype;
  description.componentManufacturer = kDessMetalManufacturer;

  const AudioComponent component = AudioComponentFindNext(nullptr, &description);
  if (component == nullptr)
  {
    std::cerr << "DessMetal AU is not registered\n";
    return 1;
  }

  AudioUnit unit = nullptr;
  if (!Check(AudioComponentInstanceNew(component, &unit), "AudioComponentInstanceNew"))
    return 1;

  const auto dispose = [&]() {
    if (unit != nullptr)
    {
      AudioUnitUninitialize(unit);
      AudioComponentInstanceDispose(unit);
      unit = nullptr;
    }
  };

  const auto inputFormat = MakeFormat(1);
  const auto outputFormat = MakeFormat(2);
  InputState inputState;
  AURenderCallbackStruct callback{RenderInput, &inputState};
  UInt32 offline = 1;
  UInt32 maxFrames = kFramesPerBlock;

  if (!Check(AudioUnitSetProperty(unit, kAudioUnitProperty_StreamFormat, kAudioUnitScope_Input, 0, &inputFormat,
                                  sizeof(inputFormat)),
             "set input format")
      || !Check(AudioUnitSetProperty(unit, kAudioUnitProperty_StreamFormat, kAudioUnitScope_Output, 0, &outputFormat,
                                     sizeof(outputFormat)),
                "set output format")
      || !Check(AudioUnitSetProperty(unit, kAudioUnitProperty_SetRenderCallback, kAudioUnitScope_Input, 0, &callback,
                                     sizeof(callback)),
                "set input callback")
      || !Check(AudioUnitSetProperty(unit, kAudioUnitProperty_MaximumFramesPerSlice, kAudioUnitScope_Global, 0,
                                     &maxFrames, sizeof(maxFrames)),
                "set maximum frames")
      || !Check(AudioUnitSetProperty(unit, kAudioUnitProperty_OfflineRender, kAudioUnitScope_Global, 0, &offline,
                                     sizeof(offline)),
                "enable offline rendering")
      || !Check(AudioUnitInitialize(unit), "AudioUnitInitialize"))
  {
    dispose();
    return 1;
  }

  std::vector<float> left(kFramesPerBlock);
  std::vector<float> right(kFramesPerBlock);
  struct StereoBufferList
  {
    AudioBufferList list;
    AudioBuffer second;
  } output{};
  output.list.mNumberBuffers = 2;
  output.list.mBuffers[0] = AudioBuffer{1, static_cast<UInt32>(left.size() * sizeof(float)), left.data()};
  output.second = AudioBuffer{1, static_cast<UInt32>(right.size() * sizeof(float)), right.data()};

  const auto start = std::chrono::steady_clock::now();
  double peak = 0.0;
  bool finite = true;
  UInt32 renderedBlocks = 0;
  const auto renderBlocks = [&](const UInt32 blockCount, double& measuredPeak, double* maxAdjacentStep = nullptr,
                                double* previousLeftSample = nullptr) {
    measuredPeak = 0.0;
    if (maxAdjacentStep != nullptr)
      *maxAdjacentStep = 0.0;
    bool havePreviousLeftSample = previousLeftSample != nullptr;
    double previousLeft = havePreviousLeftSample ? *previousLeftSample : 0.0;
    for (UInt32 block = 0; block < blockCount; ++block, ++renderedBlocks)
    {
      std::fill(left.begin(), left.end(), 0.0f);
      std::fill(right.begin(), right.end(), 0.0f);
      AudioTimeStamp timestamp{};
      timestamp.mFlags = kAudioTimeStampSampleTimeValid;
      timestamp.mSampleTime = static_cast<Float64>(renderedBlocks * kFramesPerBlock);
      AudioUnitRenderActionFlags flags = 0;
      if (!Check(AudioUnitRender(unit, &flags, &timestamp, 0, kFramesPerBlock, &output.list), "AudioUnitRender"))
        return false;
      for (const float sample : left)
      {
        finite = finite && std::isfinite(sample);
        measuredPeak = std::max(measuredPeak, std::abs(static_cast<double>(sample)));
        if (maxAdjacentStep != nullptr && havePreviousLeftSample)
          *maxAdjacentStep = std::max(*maxAdjacentStep, std::abs(static_cast<double>(sample) - previousLeft));
        previousLeft = sample;
        havePreviousLeftSample = true;
      }
      for (const float sample : right)
      {
        finite = finite && std::isfinite(sample);
        measuredPeak = std::max(measuredPeak, std::abs(static_cast<double>(sample)));
      }
    }
    if (previousLeftSample != nullptr)
      *previousLeftSample = previousLeft;
    return true;
  };

  if (!renderBlocks(24, peak))
  {
    dispose();
    return 1;
  }

  std::vector<double> drivePeaks;
  for (int driveModel = 0; driveModel < 4; ++driveModel)
  {
    double transitionPeak = 0.0;
    double drivePeak = 0.0;
    if (!Check(AudioUnitSetParameter(unit, kDriveActiveParameter, kAudioUnitScope_Global, 0, 0.0f, 0),
               "disable drive before model selection")
        || !renderBlocks(2, transitionPeak)
        || !Check(AudioUnitSetParameter(unit, kDriveModelParameter, kAudioUnitScope_Global, 0,
                                        static_cast<AudioUnitParameterValue>(driveModel), 0),
                  "select drive model")
        || !Check(AudioUnitSetParameter(unit, kDriveLevelParameter, kAudioUnitScope_Global, 0, 0.0f, 0),
                  "set drive level")
        || !Check(AudioUnitSetParameter(unit, kDriveActiveParameter, kAudioUnitScope_Global, 0, 1.0f, 0),
                  "enable drive")
        || !renderBlocks(16, transitionPeak))
    {
      dispose();
      return 1;
    }

    AudioUnitParameterValue driveActive = 0.0f;
    AudioUnitParameterValue selectedDriveModel = -1.0f;
    if (!Check(AudioUnitGetParameter(unit, kDriveActiveParameter, kAudioUnitScope_Global, 0, &driveActive),
               "read drive active state")
        || !Check(AudioUnitGetParameter(unit, kDriveModelParameter, kAudioUnitScope_Global, 0, &selectedDriveModel),
                  "read drive model state")
        || driveActive < 0.5f || std::abs(selectedDriveModel - driveModel) > 0.01f)
    {
      std::cerr << "Drive model failed to remain loaded: requested=" << driveModel
                << " selected=" << selectedDriveModel << " active=" << driveActive << '\n';
      dispose();
      return 1;
    }

    inputState.phase = 0.0;
    if (!renderBlocks(8, drivePeak) || drivePeak < 1.0e-8)
    {
      std::cerr << "Drive model " << driveModel << " rendered silence\n";
      dispose();
      return 1;
    }
    drivePeaks.push_back(drivePeak);
  }

  const auto [minDrivePeak, maxDrivePeak] = std::minmax_element(drivePeaks.begin(), drivePeaks.end());
  if (*maxDrivePeak - *minDrivePeak < 1.0e-6)
  {
    std::cerr << "Drive model selections did not produce measurably distinct output\n";
    dispose();
    return 1;
  }

  double sickDessPeak = 0.0;
  if (!Check(AudioUnitSetParameter(unit, kAmpModelParameter, kAudioUnitScope_Global, 0, 4.0f, 0),
             "select SickDess amp model")
      || !renderBlocks(24, sickDessPeak) || sickDessPeak < 1.0e-8)
  {
    std::cerr << "SickDess rendered invalid or silent output: peak=" << sickDessPeak << '\n';
    dispose();
    return 1;
  }

  Float64 latencySeconds = 0.0;
  UInt32 latencySize = sizeof(latencySeconds);
  if (!Check(AudioUnitGetProperty(unit, kAudioUnitProperty_Latency, kAudioUnitScope_Global, 0, &latencySeconds,
                                  &latencySize),
             "get offline latency")
      || !std::isfinite(latencySeconds) || latencySeconds <= 0.0 || latencySeconds >= 0.1)
  {
    std::cerr << "Offline resampling latency was not published: " << latencySeconds << " seconds\n";
    dispose();
    return 1;
  }

  double irBypassPeak = 0.0;
  double fullBypassPeak = 0.0;
  double modelSwitchBypassPeak = 0.0;
  double modelSwitchBypassMaxStep = 0.0;
  double bypassPreviousSample = 0.0;
  double reloadPeak = 0.0;
  if (!Check(AudioUnitSetParameter(unit, kIRToggleParameter, kAudioUnitScope_Global, 0, 0.0f, 0),
             "disable IR")
      || !renderBlocks(8, irBypassPeak)
      || !Check(AudioUnitSetParameter(unit, kAmpActiveParameter, kAudioUnitScope_Global, 0, 0.0f, 0),
                "disable amp")
      || !renderBlocks(8, fullBypassPeak, nullptr, &bypassPreviousSample)
      || !Check(AudioUnitSetParameter(unit, kAmpModelParameter, kAudioUnitScope_Global, 0, 1.0f, 0),
                "select replacement model while amp is bypassed")
      || !renderBlocks(8, modelSwitchBypassPeak, &modelSwitchBypassMaxStep, &bypassPreviousSample))
  {
    dispose();
    return 1;
  }

  Float64 latencyDuringBypassSelectionSeconds = 0.0;
  UInt32 latencyDuringBypassSelectionSize = sizeof(latencyDuringBypassSelectionSeconds);
  if (!Check(AudioUnitGetProperty(unit, kAudioUnitProperty_Latency, kAudioUnitScope_Global, 0,
                                  &latencyDuringBypassSelectionSeconds, &latencyDuringBypassSelectionSize),
             "get latency after bypassed model selection")
      || std::abs(latencyDuringBypassSelectionSeconds - latencySeconds) > (0.5 / kSampleRate))
  {
    std::cerr << "Bypassed model selection changed the published latency from " << latencySeconds * kSampleRate
              << " to " << latencyDuringBypassSelectionSeconds * kSampleRate << " samples\n";
    dispose();
    return 1;
  }

  if (!Check(AudioUnitSetParameter(unit, kAmpActiveParameter, kAudioUnitScope_Global, 0, 1.0f, 0),
                "reenable amp")
      || !Check(AudioUnitSetParameter(unit, kIRToggleParameter, kAudioUnitScope_Global, 0, 1.0f, 0),
                "reenable IR")
      || !renderBlocks(24, reloadPeak))
  {
    dispose();
    return 1;
  }

  Float64 latencyAfterBypassSeconds = 0.0;
  UInt32 latencyAfterBypassSize = sizeof(latencyAfterBypassSeconds);
  if (!Check(AudioUnitGetProperty(unit, kAudioUnitProperty_Latency, kAudioUnitScope_Global, 0,
                                  &latencyAfterBypassSeconds, &latencyAfterBypassSize),
             "get latency after amp bypass cycle")
      || std::abs(latencyAfterBypassSeconds - latencySeconds) > (0.5 / kSampleRate))
  {
    std::cerr << "Amp bypass changed the published latency from " << latencySeconds * kSampleRate << " to "
              << latencyAfterBypassSeconds * kSampleRate << " samples\n";
    dispose();
    return 1;
  }
  const auto elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
  dispose();

  if (!finite || peak < 1.0e-8 || irBypassPeak < 1.0e-8 || fullBypassPeak < 1.0e-8
      || modelSwitchBypassPeak < 1.0e-8 || reloadPeak < 1.0e-8)
  {
    std::cerr << "Offline render produced invalid or silent output; peaks=" << peak << ", " << irBypassPeak << ", "
              << fullBypassPeak << ", " << modelSwitchBypassPeak << ", " << reloadPeak << '\n';
    return 1;
  }
  // A 220 Hz, 0.1-amplitude sine advances by about 0.0032 per sample at
  // 44.1 kHz. Leave generous resampler tolerance while rejecting a reset,
  // zero-latency fallback, or discontinuity during a bypassed model change.
  if (modelSwitchBypassMaxStep > 0.02)
  {
    std::cerr << "Bypassed model selection broke resampler continuity; maximum adjacent step="
              << modelSwitchBypassMaxStep << '\n';
    return 1;
  }
  if (elapsed > 10.0)
  {
    std::cerr << "Offline render took too long: " << elapsed << " seconds\n";
    return 1;
  }

  std::cout << "DessMetal AU offline render passed: peaks=" << peak << ", " << irBypassPeak << ", "
            << fullBypassPeak << ", " << modelSwitchBypassPeak << ", " << reloadPeak
            << ", SickDess=" << sickDessPeak << ", drive models=" << drivePeaks[0] << "/" << drivePeaks[1]
            << "/" << drivePeaks[2] << "/" << drivePeaks[3]
            << ", bypass model-switch max-step=" << modelSwitchBypassMaxStep << ", latency="
            << latencySeconds * kSampleRate << " samples, elapsed=" << elapsed << "s\n";
  return 0;
}
