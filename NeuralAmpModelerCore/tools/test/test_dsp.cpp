// Tests for dsp

#include "NAM/dsp.h"
#include <limits>
#include <stdexcept>
#include <vector>

namespace test_dsp
{
class CountingPrewarmDSP : public nam::DSP
{
public:
  explicit CountingPrewarmDSP(const int prewarmSamples)
    : nam::DSP(1, 1, 48000.0), mPrewarmSamples(prewarmSamples), mProcessedFrames()
  {
  }

  void process(NAM_SAMPLE**, NAM_SAMPLE**, const int numFrames) override
  {
    mProcessedFrames.push_back(numFrames);
  }

  const std::vector<int>& ProcessedFrames() const { return mProcessedFrames; }

protected:
  int PrewarmSamples() override { return mPrewarmSamples; }

private:
  int mPrewarmSamples;
  std::vector<int> mProcessedFrames;
};

// Simplest test: can I construct something!
void test_construct()
{
  const int in_channels = 1;
  const int out_channels = 1;
  nam::DSP myDsp(in_channels, out_channels, 48000.0);
}

void test_channels()
{
  const int in_channels = 2;
  const int out_channels = 3;
  nam::DSP myDsp(in_channels, out_channels, 48000.0);
  assert(myDsp.NumInputChannels() == in_channels);
  assert(myDsp.NumOutputChannels() == out_channels);
}

void test_get_input_level()
{
  const int in_channels = 2;
  const int out_channels = 1;
  nam::DSP myDsp(in_channels, out_channels, 48000.0);
  const double expected = 19.0;
  myDsp.SetInputLevel(expected);
  assert(myDsp.HasInputLevel());
  const double actual = myDsp.GetInputLevel();

  assert(actual == expected);
}

void test_get_output_level()
{
  const int in_channels = 1;
  const int out_channels = 2;
  nam::DSP myDsp(in_channels, out_channels, 48000.0);
  const double expected = 12.0;
  myDsp.SetOutputLevel(expected);
  assert(myDsp.HasOutputLevel());
  const double actual = myDsp.GetOutputLevel();

  assert(actual == expected);
}

// Test correct function of DSP::HasInputLevel()
void test_has_input_level()
{
  const int in_channels = 2;
  const int out_channels = 1;
  nam::DSP myDsp(in_channels, out_channels, 48000.0);
  assert(!myDsp.HasInputLevel());

  const double level = 19.0;
  myDsp.SetInputLevel(level);
  assert(myDsp.HasInputLevel());
}

void test_has_output_level()
{
  const int in_channels = 1;
  const int out_channels = 2;
  nam::DSP myDsp(in_channels, out_channels, 48000.0);

  assert(!myDsp.HasOutputLevel());

  const double level = 12.0;
  myDsp.SetOutputLevel(level);
  assert(myDsp.HasOutputLevel());
}

// Test correct function of DSP::HasInputLevel()
void test_set_input_level()
{
  const int in_channels = 2;
  const int out_channels = 1;
  nam::DSP myDsp(in_channels, out_channels, 48000.0);
  myDsp.SetInputLevel(19.0);
}

void test_set_output_level()
{
  const int in_channels = 1;
  const int out_channels = 2;
  nam::DSP myDsp(in_channels, out_channels, 48000.0);
  myDsp.SetOutputLevel(19.0);
}

void test_audio_metadata_rejects_unsafe_values()
{
  nam::DSP myDsp(1, 1, 48000.0);

  const auto rejects = [](const auto& setter) {
    for (const double value : {std::numeric_limits<double>::infinity(),
                               -std::numeric_limits<double>::infinity(),
                               std::numeric_limits<double>::quiet_NaN(), 161.0, -161.0})
    {
      bool threw = false;
      try
      {
        setter(value);
      }
      catch (const std::invalid_argument&)
      {
        threw = true;
      }
      assert(threw);
    }
  };

  rejects([&myDsp](const double value) { myDsp.SetLoudness(value); });
  rejects([&myDsp](const double value) { myDsp.SetInputLevel(value); });
  rejects([&myDsp](const double value) { myDsp.SetOutputLevel(value); });

  myDsp.SetLoudness(-37.8406867980957);
  myDsp.SetInputLevel(18.3);
  myDsp.SetOutputLevel(12.3);
  assert(myDsp.GetLoudness() == -37.8406867980957);
  assert(myDsp.GetInputLevel() == static_cast<float>(18.3));
  assert(myDsp.GetOutputLevel() == static_cast<float>(12.3));
}

void test_prewarm_processes_only_requested_frames()
{
  CountingPrewarmDSP oneSample(1);
  oneSample.Reset(48000.0, 4096);
  assert((oneSample.ProcessedFrames() == std::vector<int>{1}));

  CountingPrewarmDSP boundaryPlusOne(4097);
  boundaryPlusOne.Reset(48000.0, 4096);
  assert((boundaryPlusOne.ProcessedFrames() == std::vector<int>{4096, 1}));
}

void test_process_multi_channel()
{
  const int in_channels = 2;
  const int out_channels = 2;
  nam::DSP myDsp(in_channels, out_channels, 48000.0);
  const int num_frames = 64;

  // Allocate buffers
  std::vector<std::vector<double>> inputBuffers(in_channels);
  std::vector<std::vector<double>> outputBuffers(out_channels);
  std::vector<double*> inputPtrs(in_channels);
  std::vector<double*> outputPtrs(out_channels);

  for (int ch = 0; ch < in_channels; ch++)
  {
    inputBuffers[ch].resize(num_frames);
    outputBuffers[ch].resize(num_frames);
    inputPtrs[ch] = inputBuffers[ch].data();
    outputPtrs[ch] = outputBuffers[ch].data();

    // Fill input with test data
    for (int i = 0; i < num_frames; i++)
    {
      inputBuffers[ch][i] = (ch + 1) * 0.5 + i * 0.01;
    }
  }
  for (int ch = 0; ch < out_channels; ch++)
  {
    outputBuffers[ch].resize(num_frames);
    outputPtrs[ch] = outputBuffers[ch].data();
  }

  // Process
  myDsp.process(inputPtrs.data(), outputPtrs.data(), num_frames);

  // Check that default implementation copied input to output
  const int channelsToCheck = std::min(in_channels, out_channels);
  for (int ch = 0; ch < channelsToCheck; ch++)
  {
    for (int i = 0; i < num_frames; i++)
    {
      assert(outputBuffers[ch][i] == inputBuffers[ch][i]);
    }
  }
}
}; // namespace test_dsp
