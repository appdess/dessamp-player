//
//  ImpulseResponse.h
//  NeuralAmpModeler-macOS
//
//  Created by Steven Atkinson on 12/30/22.
//
// Impulse response processing

#pragma once

#include <cstdint>
#include <filesystem>

#include <Eigen/Dense>

#include "dsp.h"
#include "wav.h"

namespace dsp
{
class ImpulseResponse : public History
{
public:
  struct IRData;
  ImpulseResponse(const char* fileName, const double sampleRate);
  ImpulseResponse(const IRData& irData, const double sampleRate);
  double** Process(double** inputs, const size_t numChannels, const size_t numFrames) override;
  IRData GetData();
  double GetSampleRate() const { return mSampleRate; };
  void SetPreparedEpoch(const std::uint64_t epoch) { mPreparedEpoch = epoch; }
  std::uint64_t GetPreparedEpoch() const { return mPreparedEpoch; }
  void SetPreparedRequestToken(const unsigned int token) { mPreparedRequestToken = token; }
  unsigned int GetPreparedRequestToken() const { return mPreparedRequestToken; }
  // TODO states for the IR class
  dsp::wav::LoadReturnCode GetWavState() const { return this->mWavState; };

private:
  // Set the weights, given that the plugin is running at the provided sample
  // rate.
  void _SetWeights();

  // State of audio
  dsp::wav::LoadReturnCode mWavState;
  // Keep a copy of the raw audio that was loaded so that it can be resampled
  std::vector<float> mRawAudio;
  double mRawAudioSampleRate;
  // Resampled to the required sample rate.
  std::vector<float> mResampled;
  double mSampleRate;
  std::uint64_t mPreparedEpoch = 0;
  unsigned int mPreparedRequestToken = 0;

  const size_t mMaxLength = 8192;
  // The weights
  Eigen::VectorXf mWeight;
};

struct dsp::ImpulseResponse::IRData
{
  std::vector<float> mRawAudio;
  double mRawAudioSampleRate;
};

}; // namespace dsp
