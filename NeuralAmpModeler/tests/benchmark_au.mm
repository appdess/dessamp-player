// Reuse the exact-binary registration and versioned-state fixture from integration QA.
#define main DessAmpIntegrationTestMain
#include "test_au_player.mm"
#undef main
#include <chrono>
#include <fstream>
#include <json.hpp>

namespace {
struct BenchmarkInput { double rate; std::uint64_t frame = 0; bool impulse = false; };
OSStatus BenchmarkCallback(void* context, AudioUnitRenderActionFlags*, const AudioTimeStamp*, UInt32,
                           UInt32 count, AudioBufferList* buffers)
{
  auto& input = *static_cast<BenchmarkInput*>(context);
  for (UInt32 c = 0; c < buffers->mNumberBuffers; ++c)
  {
    auto* samples = static_cast<Float32*>(buffers->mBuffers[c].mData);
    for (UInt32 s = 0; s < count; ++s)
      samples[s] = input.impulse ? (input.frame + s == 0 ? 0.1f : 0.f)
                                : 0.05f * std::sin(6.283185307179586 * 197.0 * (input.frame + s) / input.rate);
    buffers->mBuffers[c].mDataByteSize = count * sizeof(Float32);
  }
  input.frame += count;
  return noErr;
}

nlohmann::json Benchmark(AudioComponent component, const std::string& model, const std::string& ir,
                         double rate, UInt32 blockSize, bool boost, bool cabinet, bool verifyLatency)
{
  AudioUnit unit = CreateUnit(component);
  auto parameters = MakeParameters(50.0);
  parameters[0] = parameters[6] = 0.0;
  parameters[9] = 0.0; // isolate gate cost from the model benchmark
  parameters[11] = boost ? 1.0 : 0.0;
  parameters[14] = 0.0;
  parameters[13] = cabinet ? 1.0 : 0.0;
  auto state = MakeStateDictionary(model, ir, parameters);
  ApplyState(unit, state);
  CFRelease(state);
  BenchmarkInput input{rate};
  AudioStreamBasicDescription format{};
  format.mSampleRate = rate;
  format.mFormatID = kAudioFormatLinearPCM;
  format.mFormatFlags = kAudioFormatFlagsNativeFloatPacked | kAudioFormatFlagIsNonInterleaved;
  format.mBytesPerPacket = format.mBytesPerFrame = sizeof(Float32);
  format.mFramesPerPacket = format.mChannelsPerFrame = 1;
  format.mBitsPerChannel = 32;
  for (auto scope : {kAudioUnitScope_Input, kAudioUnitScope_Output})
    RequireStatus(AudioUnitSetProperty(unit, kAudioUnitProperty_StreamFormat, scope, 0, &format, sizeof(format)), "format");
  RequireStatus(AudioUnitSetProperty(unit, kAudioUnitProperty_MaximumFramesPerSlice, kAudioUnitScope_Global,
                                     0, &blockSize, sizeof(blockSize)), "block size");
  AURenderCallbackStruct callback{BenchmarkCallback, &input};
  RequireStatus(AudioUnitSetProperty(unit, kAudioUnitProperty_SetRenderCallback, kAudioUnitScope_Input,
                                     0, &callback, sizeof(callback)), "input callback");
  RequireStatus(AudioUnitInitialize(unit), "initialize");
  PumpMainRunLoop();
  std::vector<Float32> audio(blockSize);
  AudioBufferList buffers{};
  buffers.mNumberBuffers = 1;
  buffers.mBuffers[0] = {1, static_cast<UInt32>(blockSize * sizeof(Float32)), audio.data()};
  AudioTimeStamp time{};
  time.mFlags = kAudioTimeStampSampleTimeValid;
  auto render = [&]() {
    AudioUnitRenderActionFlags flags = 0;
    time.mSampleTime = input.frame;
    RequireStatus(AudioUnitRender(unit, &flags, &time, 0, blockSize, &buffers), "render");
  };
  for (int n = 0; n < 32; ++n) render();
  PumpMainRunLoop(); // publish latency after DSP adoption, outside the timed region
  double latency = 0;
  UInt32 size = sizeof(latency);
  RequireStatus(AudioUnitGetProperty(unit, kAudioUnitProperty_Latency, kAudioUnitScope_Global, 0,
                                     &latency, &size), "latency");
  std::vector<double> times;
  const int count = static_cast<int>(rate / blockSize);
  times.reserve(count);
  double energy = 0;
  for (int n = 0; n < count; ++n)
  {
    const auto start = std::chrono::steady_clock::now();
    render();
    times.push_back(std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - start).count());
    for (auto value : audio)
    {
      if (!std::isfinite(value)) throw std::runtime_error("non-finite benchmark output");
      energy += value * value;
    }
  }
  if (energy <= 1e-12) throw std::runtime_error("silent benchmark output");
  double sum = 0; for (double value : times) sum += value;
  std::sort(times.begin(), times.end());
  const double budget = 1e6 * blockSize / rate;
  nlohmann::json result = {{"sample_rate", rate}, {"block_frames", blockSize}, {"boost", boost},
    {"cabinet", cabinet}, {"reported_latency_samples", std::lround(latency * rate)},
    {"reported_latency_ms", latency * 1000}, {"mean_callback_us", sum / times.size()},
    {"p99_callback_us", times[static_cast<std::size_t>(0.99 * (times.size() - 1))]},
    {"max_callback_us", times.back()}, {"mean_budget_percent", 100 * sum / times.size() / budget},
    {"deadline_exceedances", std::count_if(times.begin(), times.end(), [&](double t) {return t > budget;})},
    {"output_rms", std::sqrt(energy / (count * blockSize))}};
  if (verifyLatency)
  {
    // Current 12-lobe resampler's causal input padding at these rates. A
    // native 48 kHz model has no conversion delay; each enabled model adds
    // one independent conversion pair. Cabinet FIR adds no block buffering.
    const long perModel = rate == 44100.0 ? 27 : rate == 96000.0 ? 42 : 0;
    if (std::lround(latency * rate) != perModel * (boost ? 2 : 1))
      throw std::runtime_error("host latency omitted an active resampler");
    if (boost)
    {
      RequireStatus(AudioUnitSetParameter(unit, 11, kAudioUnitScope_Global, 0, 0.0f, 0), "disable boost");
      PumpMainRunLoop();
      render();
      PumpMainRunLoop();
      RequireStatus(AudioUnitGetProperty(unit, kAudioUnitProperty_Latency, kAudioUnitScope_Global, 0,
                                         &latency, &size), "disabled latency");
      if (std::lround(latency * rate) != perModel)
        throw std::runtime_error("host latency retained disabled Boost delay");
    }
    result["latency_regression_passed"] = true;
  }
  RequireStatus(AudioUnitUninitialize(unit), "uninitialize");
  AudioComponentInstanceDispose(unit);
  return result;
}
}

int main(int argc, char** argv)
{
  try
  {
    if (argc != 5 && argc != 6) throw std::runtime_error("usage: benchmark_au component-binary model.dess cabinet.wav report.json [--verify-latency]");
    const bool verifyLatency = argc == 6 && std::string(argv[5]) == "--verify-latency";
    if (argc == 6 && !verifyLatency) throw std::runtime_error("unknown option");
    void* handle = nullptr;
    auto component = RegisterComponentBinary(argv[1], handle);
    nlohmann::json results = nlohmann::json::array();
    for (double rate : {44100.0, 48000.0, 96000.0})
      for (UInt32 frames : {32U, 64U, 128U})
        for (bool fullChain : {false, true})
        {
          auto result = Benchmark(component, argv[2], argv[3], rate, frames, fullChain, fullChain, verifyLatency);
          std::cerr << rate << " / " << frames << " / full=" << fullChain << ": "
                    << result["mean_budget_percent"] << "%\n";
          results.push_back(result);
        }
    std::ofstream stream(argv[4]);
    stream << nlohmann::json({{"component_binary", argv[1]}, {"model", argv[2]},
      {"method", "Sequential in-process realtime-mode AudioUnitRender; one second per case after warmup. Not hardware round-trip or a hard real-time scheduler guarantee."},
      {"cases", results}}).dump(2) << '\n';
    if (!stream) throw std::runtime_error("failed to write report");
  }
  catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
}
