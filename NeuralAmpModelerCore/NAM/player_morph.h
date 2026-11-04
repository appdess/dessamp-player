#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <memory>
#include <stdexcept>
#include <vector>

#include "dsp.h"

namespace nam::player
{
// Runtime-only description of a two-head morph. The source models remain
// ordinary NAM payloads; this composite owns their alignment and blend law.
struct MorphSettings
{
  int delay_samples = 0;
  int polarity = 1;
  double level_scale = 1.0;
  double center_gain = 1.0;
  double smoothing_ms = 30.0;
};

// Render two independently loaded amp heads from the same input and expose one
// additional block-constant parameter after the primary model's condition
// vector. All memory is reserved by Reset(), so process() remains allocation-
// free. Alignment is applied at the models' native sample rate, before the
// plug-in's outer resampler.
class MorphingDSP final : public nam::DSP
{
public:
  MorphingDSP(std::unique_ptr<nam::DSP> primary, std::unique_ptr<nam::DSP> secondary,
              const double expected_sample_rate, const MorphSettings& settings)
  : nam::DSP(1, 1, expected_sample_rate)
  , mPrimary(std::move(primary))
  , mSecondary(std::move(secondary))
  , mSettings(settings)
  {
    if (mPrimary == nullptr || mSecondary == nullptr)
      throw std::invalid_argument("Morph models must both be present");
    if (mPrimary->NumInputChannels() != 1 || mPrimary->NumOutputChannels() != 1
        || mSecondary->NumInputChannels() != 1 || mSecondary->NumOutputChannels() != 1)
      throw std::invalid_argument("Morph models must both be mono");
    if (mSecondary->NumGlobalParams() != 0)
      throw std::invalid_argument("The secondary morph model must be a snapshot");
    if (mSettings.delay_samples < 0 || mSettings.delay_samples > 4096)
      throw std::invalid_argument("Morph delay is outside the supported range");
    if (mSettings.polarity != -1 && mSettings.polarity != 1)
      throw std::invalid_argument("Morph polarity must be -1 or 1");
    if (!std::isfinite(mSettings.level_scale) || mSettings.level_scale <= 0.0
        || mSettings.level_scale > 100.0)
      throw std::invalid_argument("Morph level scale is outside the supported range");
    if (!std::isfinite(mSettings.center_gain) || mSettings.center_gain < 0.25
        || mSettings.center_gain > 4.0)
      throw std::invalid_argument("Morph center gain is outside the supported range");
    if (!std::isfinite(mSettings.smoothing_ms) || mSettings.smoothing_ms < 0.0
        || mSettings.smoothing_ms > 1000.0)
      throw std::invalid_argument("Morph smoothing time is outside the supported range");

    mPrimaryParameterCount = mPrimary->NumGlobalParams();
    if (mPrimaryParameterCount < 0 || mPrimaryParameterCount >= 64)
      throw std::invalid_argument("Primary morph parameter count is outside the supported range");
    mDefaultParameters.assign(static_cast<std::size_t>(NumGlobalParams()), 0.5);

    // The primary capture defines calibration for the composite. The fixed
    // center compensation keeps the selected 50/50 blend close to that level.
    if (mPrimary->HasInputLevel())
      SetInputLevel(mPrimary->GetInputLevel());
    if (mPrimary->HasOutputLevel())
      SetOutputLevel(mPrimary->GetOutputLevel());
    if (mPrimary->HasLoudness())
      SetLoudness(mPrimary->GetLoudness());
  }

  int NumGlobalParams() const override { return mPrimaryParameterCount + 1; }

  std::size_t EstimatedOperationsPerSample() const override
  {
    return mPrimary->EstimatedOperationsPerSample() + mSecondary->EstimatedOperationsPerSample() + 12U;
  }

  void SetDefaultParams(const double* params, const int num_params) override
  {
    if (num_params != NumGlobalParams() || params == nullptr)
      throw std::invalid_argument("Morph default parameter count is invalid");
    for (int index = 0; index < num_params; ++index)
    {
      if (!std::isfinite(params[index]))
        throw std::invalid_argument("Morph default parameters must be finite");
      mDefaultParameters[static_cast<std::size_t>(index)] = params[index];
    }
    mPrimary->SetDefaultParams(
      mPrimaryParameterCount == 0 ? nullptr : mDefaultParameters.data(), mPrimaryParameterCount);
    mSecondary->SetDefaultParams(nullptr, 0);
    mDefaultMorph = std::clamp(mDefaultParameters.back(), 0.0, 1.0);
  }

  void Reset(const double sample_rate, const int max_buffer_size) override
  {
    if (!std::isfinite(sample_rate) || sample_rate <= 0.0 || max_buffer_size <= 0)
      throw std::invalid_argument("Morph reset configuration is invalid");
    if (std::isfinite(GetExpectedSampleRate()) && GetExpectedSampleRate() > 0.0
        && std::abs(sample_rate - GetExpectedSampleRate()) > 1.0e-6)
      throw std::invalid_argument("Morph composite must run at its native sample rate");

    mExternalSampleRate = sample_rate;
    mHaveExternalSampleRate = true;
    SetMaxBufferSize(max_buffer_size);
    mPrimaryOutput.assign(static_cast<std::size_t>(max_buffer_size), 0.0);
    mSecondaryOutput.assign(static_cast<std::size_t>(max_buffer_size), 0.0);
    mPrimaryOutputPointer = mPrimaryOutput.data();
    mSecondaryOutputPointer = mSecondaryOutput.data();
    mDelayLine.assign(static_cast<std::size_t>(mSettings.delay_samples), 0.0);
    mDelayPosition = 0U;
    mCurrentMorph = mDefaultMorph;
    mSmoothingCoefficient = mSettings.smoothing_ms <= 0.0
                              ? 0.0
                              : std::exp(-1.0 / (0.001 * mSettings.smoothing_ms * sample_rate));

    mPrimary->Reset(sample_rate, max_buffer_size);
    mSecondary->Reset(sample_rate, max_buffer_size);
  }

  void prewarm() override
  {
    mPrimary->prewarm();
    mSecondary->prewarm();
    std::fill(mDelayLine.begin(), mDelayLine.end(), 0.0);
    mDelayPosition = 0U;
    mCurrentMorph = mDefaultMorph;
  }

  void process(NAM_SAMPLE** input, NAM_SAMPLE** output, const int num_frames) override
  {
    process(input, output, num_frames, mDefaultParameters.data(), NumGlobalParams());
  }

  void process(NAM_SAMPLE** input, NAM_SAMPLE** output, const int num_frames, const double* params,
               const int num_params) override
  {
    if (num_frames < 0 || static_cast<std::size_t>(num_frames) > mPrimaryOutput.size())
      throw std::invalid_argument("Morph process block exceeds the prepared size");
    if (num_params != NumGlobalParams() || params == nullptr)
      throw std::invalid_argument("Morph process parameter count is invalid");
    if (input == nullptr || output == nullptr || input[0] == nullptr || output[0] == nullptr)
      throw std::invalid_argument("Morph process buffers are invalid");

    mPrimary->process(
      input, &mPrimaryOutputPointer, num_frames, mPrimaryParameterCount == 0 ? nullptr : params,
      mPrimaryParameterCount);
    mSecondary->process(input, &mSecondaryOutputPointer, num_frames);

    const double target_morph = std::clamp(params[mPrimaryParameterCount], 0.0, 1.0);
    for (int frame = 0; frame < num_frames; ++frame)
    {
      double secondary = mSecondaryOutput[static_cast<std::size_t>(frame)];
      if (!mDelayLine.empty())
      {
        const double delayed = mDelayLine[mDelayPosition];
        mDelayLine[mDelayPosition] = secondary;
        mDelayPosition = (mDelayPosition + 1U) % mDelayLine.size();
        secondary = delayed;
      }
      secondary *= static_cast<double>(mSettings.polarity) * mSettings.level_scale;

      mCurrentMorph = target_morph + mSmoothingCoefficient * (mCurrentMorph - target_morph);
      const double compensation =
        1.0 + 4.0 * (mSettings.center_gain - 1.0) * mCurrentMorph * (1.0 - mCurrentMorph);
      output[0][frame] = static_cast<NAM_SAMPLE>(
        compensation
        * ((1.0 - mCurrentMorph) * mPrimaryOutput[static_cast<std::size_t>(frame)]
           + mCurrentMorph * secondary));
    }
  }

private:
  std::unique_ptr<nam::DSP> mPrimary;
  std::unique_ptr<nam::DSP> mSecondary;
  MorphSettings mSettings;
  int mPrimaryParameterCount = 0;
  std::vector<double> mDefaultParameters;
  std::vector<NAM_SAMPLE> mPrimaryOutput;
  std::vector<NAM_SAMPLE> mSecondaryOutput;
  NAM_SAMPLE* mPrimaryOutputPointer = nullptr;
  NAM_SAMPLE* mSecondaryOutputPointer = nullptr;
  std::vector<NAM_SAMPLE> mDelayLine;
  std::size_t mDelayPosition = 0U;
  double mDefaultMorph = 0.5;
  double mCurrentMorph = 0.5;
  double mSmoothingCoefficient = 0.0;
};
} // namespace nam::player
