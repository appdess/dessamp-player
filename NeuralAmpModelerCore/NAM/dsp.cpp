#include <algorithm> // std::max_element
#include <cmath> // pow, tanh, expf
#include <filesystem>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>

#include "dsp.h"
#include "model_validation.h"
#include "registry.h"

#define tanh_impl_ std::tanh
// #define tanh_impl_ fast_tanh_

constexpr const long _INPUT_BUFFER_SAFETY_FACTOR = 32;
constexpr double _MAX_EXPECTED_SAMPLE_RATE = 768000.0;
constexpr int _MAX_MODEL_IO_CHANNELS = 4096;
constexpr int _MAX_PROCESS_BUFFER_FRAMES = 1 << 20;
constexpr std::size_t _MAX_PREWARM_BUFFER_VALUES = 16U << 20;

namespace
{
constexpr std::size_t kMaxLinearBufferValues = 16U << 20;
constexpr double kMaxAbsMetadataDb = 160.0;

double checked_metadata_db_value(const double value, const char* name)
{
  // Express this as a closed-range check so unordered NaN comparisons fail
  // closed even in optimized builds; the finite endpoints also reject
  // infinities and extreme finite values.
  if (!(value >= -kMaxAbsMetadataDb && value <= kMaxAbsMetadataDb))
    throw std::invalid_argument(std::string(name) + " is outside the supported audio metadata range");
  return value;
}

int validate_linear_model(const int in_channels, const int out_channels, const int receptive_field, const bool bias,
                          const std::size_t actual_weights)
{
  using nam::model_validation::checked_add;
  using nam::model_validation::checked_multiply;
  using nam::model_validation::positive_dimension;

  const std::size_t input_channels = positive_dimension(in_channels, "Linear in_channels");
  positive_dimension(out_channels, "Linear out_channels");
  const std::size_t field = positive_dimension(receptive_field, "Linear receptive_field");
  const std::size_t expected_weights = checked_add(field, bias ? 1U : 0U, "Linear weights");
  nam::model_validation::require_exact_weight_count("Linear", expected_weights, actual_weights);

  const std::size_t buffer_values = checked_multiply(
    input_channels, checked_multiply(static_cast<std::size_t>(_INPUT_BUFFER_SAFETY_FACTOR), field, "Linear buffer"),
    "Linear buffer");
  if (buffer_values > kMaxLinearBufferValues
      || field > static_cast<std::size_t>(std::numeric_limits<int>::max() / _INPUT_BUFFER_SAFETY_FACTOR))
  {
    throw std::invalid_argument("Linear buffer exceeds the supported resource limit");
  }
  return receptive_field;
}
}

nam::DSP::DSP(const int in_channels, const int out_channels, const double expected_sample_rate)
: mExpectedSampleRate(expected_sample_rate)
, mInChannels(in_channels)
, mOutChannels(out_channels)
{
  if (in_channels <= 0 || out_channels <= 0 || in_channels > _MAX_MODEL_IO_CHANNELS
      || out_channels > _MAX_MODEL_IO_CHANNELS)
  {
    throw std::runtime_error("Channel counts are outside the supported range");
  }
  if (expected_sample_rate != NAM_UNKNOWN_EXPECTED_SAMPLE_RATE
      && (!std::isfinite(expected_sample_rate) || expected_sample_rate <= 0.0
          || expected_sample_rate > _MAX_EXPECTED_SAMPLE_RATE))
  {
    throw std::invalid_argument("Expected sample rate is outside the supported range");
  }
}

void nam::DSP::prewarm()
{
  if (mMaxBufferSize == 0)
  {
    SetMaxBufferSize(4096);
  }
  const int prewarmSamples = PrewarmSamples();
  if (prewarmSamples == 0)
    return;

  const size_t bufferSize = std::max(mMaxBufferSize, 1);
  const std::size_t channel_count = model_validation::checked_add(
    static_cast<std::size_t>(mInChannels), static_cast<std::size_t>(mOutChannels), "Prewarm channel count");
  const std::size_t prewarm_values =
    model_validation::checked_multiply(channel_count, bufferSize, "Prewarm buffers");
  if (prewarm_values > _MAX_PREWARM_BUFFER_VALUES)
    throw std::invalid_argument("Prewarm buffers exceed the supported resource limit");
  // Allocate buffers for all channels
  std::vector<std::vector<NAM_SAMPLE>> inputBuffers(mInChannels);
  std::vector<std::vector<NAM_SAMPLE>> outputBuffers(mOutChannels);
  std::vector<NAM_SAMPLE*> inputPtrs(mInChannels);
  std::vector<NAM_SAMPLE*> outputPtrs(mOutChannels);

  for (int ch = 0; ch < mInChannels; ch++)
  {
    inputBuffers[ch].resize(bufferSize, (NAM_SAMPLE)0.0);
    inputPtrs[ch] = inputBuffers[ch].data();
  }
  for (int ch = 0; ch < mOutChannels; ch++)
  {
    outputBuffers[ch].resize(bufferSize, (NAM_SAMPLE)0.0);
    outputPtrs[ch] = outputBuffers[ch].data();
  }

  int samplesProcessed = 0;
  while (samplesProcessed < prewarmSamples)
  {
    const int frames = std::min(mMaxBufferSize, prewarmSamples - samplesProcessed);
    this->process(inputPtrs.data(), outputPtrs.data(), frames);
    samplesProcessed += frames;
  }
}

void nam::DSP::process(NAM_SAMPLE** input, NAM_SAMPLE** output, const int num_frames)
{
  // Default implementation is the null operation: copy input to output
  // For now, assume 1:1 channel mapping (first min(in_channels, out_channels) channels)
  const int channelsToProcess = std::min(mInChannels, mOutChannels);
  for (int ch = 0; ch < channelsToProcess; ch++)
  {
    for (int i = 0; i < num_frames; i++)
      output[ch][i] = input[ch][i];
  }
  // Zero out any extra output channels
  for (int ch = channelsToProcess; ch < mOutChannels; ch++)
  {
    for (int i = 0; i < num_frames; i++)
      output[ch][i] = (NAM_SAMPLE)0.0;
  }
}

double nam::DSP::GetLoudness() const
{
  if (!HasLoudness())
  {
    throw std::runtime_error("Asked for loudness of a model that doesn't know how loud it is!");
  }
  return mLoudness;
}

void nam::DSP::Reset(const double sampleRate, const int maxBufferSize)
{
  // Some subclasses might want to throw an exception if the sample rate is "wrong".
  // This could be under a debugging flag potentially.
  mExternalSampleRate = sampleRate;
  mHaveExternalSampleRate = true;
  SetMaxBufferSize(maxBufferSize);

  prewarm();
}

void nam::DSP::SetLoudness(const double loudness)
{
  mLoudness = checked_metadata_db_value(loudness, "Loudness");
  mHasLoudness = true;
}

void nam::DSP::SetMaxBufferSize(const int maxBufferSize)
{
  if (maxBufferSize <= 0 || maxBufferSize > _MAX_PROCESS_BUFFER_FRAMES)
    throw std::invalid_argument("Maximum buffer size is outside the supported range");
  mMaxBufferSize = maxBufferSize;
}

double nam::DSP::GetInputLevel()
{
  return mInputLevel.level;
}

double nam::DSP::GetOutputLevel()
{
  return mOutputLevel.level;
}

bool nam::DSP::HasInputLevel()
{
  return mInputLevel.haveLevel;
}

bool nam::DSP::HasOutputLevel()
{
  return mOutputLevel.haveLevel;
}

void nam::DSP::SetInputLevel(const double inputLevel)
{
  const float checked_level = static_cast<float>(checked_metadata_db_value(inputLevel, "Input level"));
  mInputLevel.haveLevel = true;
  mInputLevel.level = checked_level;
}

void nam::DSP::SetOutputLevel(const double outputLevel)
{
  const float checked_level = static_cast<float>(checked_metadata_db_value(outputLevel, "Output level"));
  mOutputLevel.haveLevel = true;
  mOutputLevel.level = checked_level;
}

// Buffer =====================================================================

nam::Buffer::Buffer(const int in_channels, const int out_channels, const int receptive_field,
                    const double expected_sample_rate)
: nam::DSP(in_channels, out_channels, expected_sample_rate)
{
  this->_set_receptive_field(receptive_field);
}

void nam::Buffer::_set_receptive_field(const int new_receptive_field)
{
  this->_set_receptive_field(new_receptive_field, _INPUT_BUFFER_SAFETY_FACTOR * new_receptive_field);
};

void nam::Buffer::_set_receptive_field(const int new_receptive_field, const int input_buffer_size)
{
  this->_receptive_field = new_receptive_field;
  const int in_channels = NumInputChannels();
  const int out_channels = NumOutputChannels();

  // Resize buffers for all input channels
  _input_buffers.resize(in_channels);
  for (int ch = 0; ch < in_channels; ch++)
  {
    _input_buffers[ch].resize(input_buffer_size);
    std::fill(_input_buffers[ch].begin(), _input_buffers[ch].end(), 0.0f);
  }

  // Resize output buffers (though they'll be resized per call in _update_buffers_)
  _output_buffers.resize(out_channels);

  this->_reset_input_buffer();
}

void nam::Buffer::_update_buffers_(NAM_SAMPLE** input, const int num_frames)
{
  const int in_channels = NumInputChannels();
  const int out_channels = NumOutputChannels();

  // Make sure that the buffers are big enough for the receptive field and the
  // frames needed. All channels use the same buffer size.
  const long minimum_input_buffer_size = (long)this->_receptive_field + _INPUT_BUFFER_SAFETY_FACTOR * num_frames;

  for (int ch = 0; ch < in_channels; ch++)
  {
    if ((long)this->_input_buffers[ch].size() < minimum_input_buffer_size)
    {
      long new_buffer_size = 2;
      while (new_buffer_size < minimum_input_buffer_size)
        new_buffer_size *= 2;
      this->_input_buffers[ch].resize(new_buffer_size);
      std::fill(this->_input_buffers[ch].begin(), this->_input_buffers[ch].end(), 0.0f);
    }
  }

  // If we'd run off the end of the input buffer, then we need to move the data
  // back to the start of the buffer and start again. All channels move together.
  const long buffer_size = (long)this->_input_buffers[0].size();
  if (this->_input_buffer_offset + num_frames > buffer_size)
    this->_rewind_buffers_();

  // Put the new samples into the input buffer for each channel
  for (int ch = 0; ch < in_channels; ch++)
  {
    for (long i = this->_input_buffer_offset, j = 0; j < num_frames; i++, j++)
      this->_input_buffers[ch][i] = (float)input[ch][j];
  }

  // Resize output buffers for all output channels
  for (int ch = 0; ch < out_channels; ch++)
  {
    this->_output_buffers[ch].resize(num_frames);
    std::fill(this->_output_buffers[ch].begin(), this->_output_buffers[ch].end(), 0.0f);
  }
}

void nam::Buffer::_rewind_buffers_()
{
  const int in_channels = NumInputChannels();

  // Rewind buffers for all input channels (they all move together)
  for (int ch = 0; ch < in_channels; ch++)
  {
    // Copy the input buffer back
    // RF-1 samples because we've got at least one new one inbound.
    for (long i = 0, j = this->_input_buffer_offset - this->_receptive_field; i < this->_receptive_field; i++, j++)
      this->_input_buffers[ch][i] = this->_input_buffers[ch][j];
  }
  // And reset the offset.
  // Even though we could be stingy about that one sample that we won't be using
  // (because a new set is incoming) it's probably not worth the
  // hyper-optimization and liable for bugs. And the code looks way tidier this
  // way.
  this->_input_buffer_offset = this->_receptive_field;
}

void nam::Buffer::_reset_input_buffer()
{
  this->_input_buffer_offset = this->_receptive_field;
}

void nam::Buffer::_advance_input_buffer_(const int num_frames)
{
  this->_input_buffer_offset += num_frames;
}

// Linear =====================================================================

nam::Linear::Linear(const int in_channels, const int out_channels, const int receptive_field, const bool _bias,
                    const std::vector<float>& weights, const double expected_sample_rate)
: nam::Buffer(in_channels, out_channels,
              validate_linear_model(in_channels, out_channels, receptive_field, _bias, weights.size()),
              expected_sample_rate)
{
  this->_weight.resize(this->_receptive_field);
  // Pass in in reverse order so that dot products work out of the box.
  for (int i = 0; i < this->_receptive_field; i++)
    this->_weight(i) = weights[receptive_field - 1 - i];
  this->_bias = _bias ? weights[receptive_field] : (float)0.0;
}

void nam::Linear::process(NAM_SAMPLE** input, NAM_SAMPLE** output, const int num_frames)
{
  this->nam::Buffer::_update_buffers_(input, num_frames);

  const int in_channels = NumInputChannels();
  const int out_channels = NumOutputChannels();

  // For now, Linear processes each input channel independently to corresponding output channel
  // This is a simple implementation - can be extended later for cross-channel mixing
  const int channelsToProcess = std::min(in_channels, out_channels);

  // Main computation!
  for (int ch = 0; ch < channelsToProcess; ch++)
  {
    for (int i = 0; i < num_frames; i++)
    {
      const long offset = this->_input_buffer_offset - this->_weight.size() + i + 1;
      auto input_vec = Eigen::Map<const Eigen::VectorXf>(&this->_input_buffers[ch][offset], this->_receptive_field);
      output[ch][i] = this->_bias + this->_weight.dot(input_vec);
    }
  }

  // Zero out any extra output channels
  for (int ch = channelsToProcess; ch < out_channels; ch++)
  {
    for (int i = 0; i < num_frames; i++)
      output[ch][i] = (NAM_SAMPLE)0.0;
  }

  // Prepare for next call:
  nam::Buffer::_advance_input_buffer_(num_frames);
}

// Factory
std::unique_ptr<nam::DSP> nam::linear::Factory(const nlohmann::json& config, std::vector<float>& weights,
                                               const double expectedSampleRate)
{
  const int receptive_field = model_validation::json_integer_at(config, "receptive_field");
  const bool bias = config.at("bias");
  // Default to 1 channel in/out for backward compatibility
  const int in_channels = model_validation::json_integer_value(config, "in_channels", 1);
  const int out_channels = model_validation::json_integer_value(config, "out_channels", 1);
  return std::make_unique<nam::Linear>(in_channels, out_channels, receptive_field, bias, weights, expectedSampleRate);
}

// NN modules =================================================================

// Conv1x1 ====================================================================

nam::Conv1x1::Conv1x1(const int in_channels, const int out_channels, const bool _bias, const int groups)
{
  model_validation::require_grouped_channels(in_channels, out_channels, groups, "Conv1x1");
  const auto dense_values = model_validation::checked_multiply(
    static_cast<std::size_t>(in_channels), static_cast<std::size_t>(out_channels), "Conv1x1 dense storage");
  model_validation::require_dense_layer_limit(dense_values, "Conv1x1");

  this->_num_groups = groups;
  this->_weight.resize(out_channels, in_channels);
  this->_do_bias = _bias;
  if (_bias)
    this->_bias.resize(out_channels);
}


void nam::Conv1x1::SetMaxBufferSize(const int maxBufferSize)
{
  _output.resize(get_out_channels(), maxBufferSize);
}

void nam::Conv1x1::set_weights_(std::vector<float>::iterator& weights)
{
  if (this->_weight.size() > 0)
  {
    const long out_channels = this->_weight.rows();
    const long in_channels = this->_weight.cols();
    const int numGroups = this->_num_groups;
    const long out_per_group = out_channels / numGroups;
    const long in_per_group = in_channels / numGroups;

    // For grouped convolutions, weights are organized per group
    // Weight layout: weights are [group0, group1, ..., groupN-1]
    // Each group's weight matrix is (out_channels/numGroups, in_channels/numGroups)
    for (int g = 0; g < numGroups; g++)
    {
      for (auto i = 0; i < out_per_group; i++)
      {
        for (auto j = 0; j < in_per_group; j++)
        {
          this->_weight(g * out_per_group + i, g * in_per_group + j) = *(weights++);
        }
      }
    }
  }
  if (this->_do_bias)
    for (int i = 0; i < this->_bias.size(); i++)
      this->_bias(i) = *(weights++);
}

Eigen::MatrixXf nam::Conv1x1::process(const Eigen::MatrixXf& input, const int num_frames) const
{
  const int numGroups = this->_num_groups;
  const long in_channels = get_in_channels();
  const long out_channels = get_out_channels();
  const long in_per_group = in_channels / numGroups;
  const long out_per_group = out_channels / numGroups;

  Eigen::MatrixXf result(out_channels, num_frames);

  if (numGroups == 1)
  {
    // Standard convolution (no grouping)
    if (this->_do_bias)
      result = (this->_weight * input.leftCols(num_frames)).colwise() + this->_bias;
    else
      result = this->_weight * input.leftCols(num_frames);
  }
  else
  {
    // Grouped convolution: process each group separately
    result.setZero();
    for (int g = 0; g < numGroups; g++)
    {
      // Extract input slice for this group
      auto input_group = input.leftCols(num_frames).middleRows(g * in_per_group, in_per_group);

      // Extract weight slice for this group
      auto weight_group = this->_weight.block(g * out_per_group, g * in_per_group, out_per_group, in_per_group);

      // Extract output slice for this group
      auto output_group = result.middleRows(g * out_per_group, out_per_group);

      // Perform grouped convolution: output_group = weight_group * input_group
      output_group.noalias() = weight_group * input_group;
    }

    // Add bias if present
    if (this->_do_bias)
      result.colwise() += this->_bias;
  }

  return result;
}

void nam::Conv1x1::process_(const Eigen::Ref<const Eigen::MatrixXf>& input, const int num_frames)
{
  assert(num_frames <= _output.cols());

  const int numGroups = this->_num_groups;
  const long in_channels = get_in_channels();
  const long out_channels = get_out_channels();
  const long in_per_group = in_channels / numGroups;
  const long out_per_group = out_channels / numGroups;

  if (numGroups == 1)
  {
    // Standard convolution (no grouping)
    _output.leftCols(num_frames).noalias() = this->_weight * input.leftCols(num_frames);
  }
  else
  {
    // Grouped convolution: process each group separately
    _output.leftCols(num_frames).setZero();
    for (int g = 0; g < numGroups; g++)
    {
      // Extract input slice for this group
      auto input_group = input.leftCols(num_frames).middleRows(g * in_per_group, in_per_group);

      // Extract weight slice for this group
      auto weight_group = this->_weight.block(g * out_per_group, g * in_per_group, out_per_group, in_per_group);

      // Extract output slice for this group
      auto output_group = _output.leftCols(num_frames).middleRows(g * out_per_group, out_per_group);

      // Perform grouped convolution: output_group = weight_group * input_group
      output_group.noalias() = weight_group * input_group;
    }
  }

  // Add bias if present
  if (this->_do_bias)
  {
    _output.leftCols(num_frames).colwise() += this->_bias;
  }
}
