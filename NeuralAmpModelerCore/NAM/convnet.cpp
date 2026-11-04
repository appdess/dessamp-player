#include <algorithm> // std::max_element
#include <algorithm>
#include <cmath> // pow, tanh, expf
#include <filesystem>
#include <fstream>
#include <limits>
#include <string>
#include <unordered_map>
#include <unordered_set>

#include "dsp.h"
#include "model_validation.h"
#include "registry.h"
#include "convnet.h"

namespace
{
constexpr std::size_t kMaxConvNetPrewarmSamples = 1U << 20;
constexpr std::size_t kMaxConvNetPrewarmWork = 64U << 20;
constexpr std::size_t kMaxConvNetBufferValues = 16U << 20;
constexpr std::size_t kMaxConvNetDenseValues = 32U << 20;
constexpr std::size_t kMaxConvNetRuntimeValues = 64U << 20;
constexpr std::size_t kValidationBufferFrames = 4096;

std::size_t estimate_convnet_operations_per_sample(const int in_channels, const int out_channels,
                                                    const int channels, const std::size_t layer_count)
{
  using nam::model_validation::checked_add;
  using nam::model_validation::checked_multiply;

  const auto input_channels = static_cast<std::size_t>(in_channels);
  const auto output_channels = static_cast<std::size_t>(out_channels);
  const auto hidden_channels = static_cast<std::size_t>(channels);
  std::size_t dense_values = checked_multiply(
    2U, checked_multiply(hidden_channels, input_channels, "ConvNet dense storage"), "ConvNet dense storage");
  if (layer_count > 1U)
  {
    dense_values = checked_add(
      dense_values,
      checked_multiply(
        layer_count - 1U,
        checked_multiply(
          2U, checked_multiply(hidden_channels, hidden_channels, "ConvNet dense storage"),
          "ConvNet dense storage"),
        "ConvNet dense storage"),
      "ConvNet dense storage");
  }
  return checked_add(
    dense_values, checked_multiply(output_channels, hidden_channels, "ConvNet dense storage"),
    "ConvNet dense storage");
}
}

nam::convnet::BatchNorm::BatchNorm(const int dim, std::vector<float>::iterator& weights)
{
  // Extract from param buffer
  Eigen::VectorXf running_mean(dim);
  Eigen::VectorXf running_var(dim);
  Eigen::VectorXf _weight(dim);
  Eigen::VectorXf _bias(dim);
  for (int i = 0; i < dim; i++)
    running_mean(i) = *(weights++);
  for (int i = 0; i < dim; i++)
    running_var(i) = *(weights++);
  for (int i = 0; i < dim; i++)
    _weight(i) = *(weights++);
  for (int i = 0; i < dim; i++)
    _bias(i) = *(weights++);
  float eps = *(weights++);

  // Convert to scale & loc
  this->scale.resize(dim);
  this->loc.resize(dim);
  for (int i = 0; i < dim; i++)
    this->scale(i) = _weight(i) / sqrt(eps + running_var(i));
  this->loc = _bias - this->scale.cwiseProduct(running_mean);
}

void nam::convnet::BatchNorm::process_(Eigen::MatrixXf& x, const long i_start, const long i_end) const
{
  // todo using colwise?
  // #speed but conv probably dominates
  for (auto i = i_start; i < i_end; i++)
  {
    x.col(i) = x.col(i).cwiseProduct(this->scale);
    x.col(i) += this->loc;
  }
}

void nam::convnet::ConvNetBlock::set_weights_(const int in_channels, const int out_channels, const int _dilation,
                                              const bool batchnorm,
                                              const activations::ActivationConfig& activation_config, const int groups,
                                              std::vector<float>::iterator& weights)
{
  this->_batchnorm = batchnorm;
  // HACK 2 kernel
  this->conv.set_size_and_weights_(in_channels, out_channels, 2, _dilation, !batchnorm, groups, weights);
  if (this->_batchnorm)
    this->batchnorm = BatchNorm(out_channels, weights);
  this->activation = activations::Activation::get_activation(activation_config);
}

void nam::convnet::ConvNetBlock::SetMaxBufferSize(const int maxBufferSize)
{
  this->conv.SetMaxBufferSize(maxBufferSize);
  const long out_channels = get_out_channels();
  this->_output.resize(out_channels, maxBufferSize);
  this->_output.setZero();
}

void nam::convnet::ConvNetBlock::Process(const Eigen::MatrixXf& input, const int num_frames)
{
  // Process input with Conv1D
  this->conv.Process(input, num_frames);

  // Get output from Conv1D (this is a block reference to conv's _output buffer)
  auto conv_output_block = this->conv.GetOutput().leftCols(num_frames);

  // Copy conv output to our own output buffer
  this->_output.leftCols(num_frames) = conv_output_block;

  // Apply batchnorm if needed
  if (this->_batchnorm)
  {
    // Batchnorm expects indices, so we use 0 to num_frames for our output matrix
    this->batchnorm.process_(this->_output, 0, num_frames);
  }

  // Apply activation
  this->activation->apply(this->_output.leftCols(num_frames));
}

Eigen::Block<Eigen::MatrixXf> nam::convnet::ConvNetBlock::GetOutput(const int num_frames)
{
  return this->_output.block(0, 0, this->_output.rows(), num_frames);
}

void nam::convnet::ConvNetBlock::process_(const Eigen::MatrixXf& input, Eigen::MatrixXf& output, const long i_start,
                                          const long i_end)
{
  const long ncols = i_end - i_start;
  // Extract input slice and process with Conv1D
  Eigen::MatrixXf input_slice = input.middleCols(i_start, ncols);
  this->conv.Process(input_slice, (int)ncols);

  // Get output from Conv1D (this is a block reference to _output buffer)
  auto conv_output_block = this->conv.GetOutput().leftCols((int)ncols);

  // For batchnorm, we need a matrix reference (not a block)
  // Create a temporary matrix from the block, process it, then copy back
  Eigen::MatrixXf temp_output = conv_output_block;

  // Apply batchnorm if needed
  if (this->_batchnorm)
  {
    // Batchnorm expects indices, so we use 0 to ncols for our temp matrix
    this->batchnorm.process_(temp_output, 0, ncols);
  }

  // Apply activation
  this->activation->apply(temp_output);

  // Copy to Conv1D's output buffer and to output matrix
  conv_output_block = temp_output;
  output.middleCols(i_start, ncols) = temp_output;
}

long nam::convnet::ConvNetBlock::get_out_channels() const
{
  return this->conv.get_out_channels();
}

nam::convnet::_Head::_Head(const int in_channels, const int out_channels, std::vector<float>::iterator& weights)
{
  // Weights are stored row-major: first row (output 0), then row 1 (output 1), etc.
  // For each output channel: [w0, w1, ..., w_{in_channels-1}]
  // Then biases: [bias0, bias1, ..., bias_{out_channels-1}]
  this->_weight.resize(out_channels, in_channels);
  for (int out_ch = 0; out_ch < out_channels; out_ch++)
  {
    for (int in_ch = 0; in_ch < in_channels; in_ch++)
    {
      this->_weight(out_ch, in_ch) = *(weights++);
    }
  }

  // Biases for each output channel
  this->_bias.resize(out_channels);
  for (int out_ch = 0; out_ch < out_channels; out_ch++)
  {
    this->_bias(out_ch) = *(weights++);
  }
}

void nam::convnet::_Head::process_(const Eigen::MatrixXf& input, Eigen::MatrixXf& output, const long i_start,
                                   const long i_end) const
{
  const long length = i_end - i_start;
  const long out_channels = this->_weight.rows();

  // Resize output to (out_channels x length)
  output.resize(out_channels, length);

  // Extract input slice: (in_channels x length)
  Eigen::MatrixXf input_slice = input.middleCols(i_start, length);

  // Compute output = weight * input_slice: (out_channels x in_channels) * (in_channels x length) = (out_channels x
  // length)
  output.noalias() = this->_weight * input_slice;

  // Add bias to each column: output.colwise() += bias
  // output is (out_channels x length), bias is (out_channels x 1), so colwise() += works
  output.colwise() += this->_bias;
}

nam::convnet::ConvNet::ConvNet(const int in_channels, const int out_channels, const int channels,
                               const std::vector<int>& dilations, const bool batchnorm,
                               const activations::ActivationConfig& activation_config, std::vector<float>& weights,
                               const double expected_sample_rate, const int groups)
: Buffer(in_channels, out_channels,
         _verify_weights(in_channels, out_channels, channels, dilations, batchnorm, groups, weights.size()),
         expected_sample_rate)
{
  this->mEstimatedOperationsPerSample =
    estimate_convnet_operations_per_sample(in_channels, out_channels, channels, dilations.size());
  this->_blocks.resize(dilations.size());
  std::vector<float>::iterator it = weights.begin();
  // First block takes in_channels input, subsequent blocks take channels input
  for (size_t i = 0; i < dilations.size(); i++)
    this->_blocks[i].set_weights_(
      i == 0 ? in_channels : channels, channels, dilations[i], batchnorm, activation_config, groups, it);
  // Only need _block_vals for the head (one entry)
  // Conv1D layers manage their own buffers now
  this->_block_vals.resize(1);
  this->_block_vals[0].setZero();

  // Create single head that outputs all channels
  this->_head = _Head(channels, out_channels, it);

  if (it != weights.end())
    throw std::runtime_error("Didn't touch all the weights when initializing ConvNet");

  mPrewarmSamples = 1;
  for (size_t i = 0; i < dilations.size(); i++)
    mPrewarmSamples += dilations[i];
}


void nam::convnet::ConvNet::process(NAM_SAMPLE** input, NAM_SAMPLE** output, const int num_frames)

{
  this->_update_buffers_(input, num_frames);
  const int in_channels = NumInputChannels();
  const int out_channels = NumOutputChannels();

  // For multi-channel, we process each input channel independently through the network
  // and sum outputs to each output channel (simple implementation)
  // This can be extended later for more sophisticated cross-channel processing

  // Convert input buffers to matrix for first layer (stack input channels)
  Eigen::MatrixXf input_matrix(in_channels, num_frames);
  const long i_start = this->_input_buffer_offset;
  for (int ch = 0; ch < in_channels; ch++)
  {
    for (int i = 0; i < num_frames; i++)
      input_matrix(ch, i) = this->_input_buffers[ch][i_start + i];
  }

  // Process through ConvNetBlock layers
  // Each block now uses Conv1D's internal buffers via Process() and GetOutput()
  for (size_t i = 0; i < this->_blocks.size(); i++)
  {
    // Get input for this block
    Eigen::MatrixXf block_input;
    if (i == 0)
    {
      // First block uses the input matrix
      block_input = input_matrix;
    }
    else
    {
      // Subsequent blocks use output from previous block
      auto prev_output = this->_blocks[i - 1].GetOutput(num_frames);
      block_input = prev_output; // Copy to matrix
    }

    // Process block (handles Conv1D, batchnorm, and activation internally)
    this->_blocks[i].Process(block_input, num_frames);
  }

  // Process head for all output channels at once
  // We need _block_vals[0] for the head interface
  const long buffer_size = (long)this->_input_buffers[0].size();
  if (this->_block_vals[0].rows() != this->_blocks.back().get_out_channels()
      || this->_block_vals[0].cols() != buffer_size)
  {
    this->_block_vals[0].resize(this->_blocks.back().get_out_channels(), buffer_size);
  }

  // Copy last block output to _block_vals for head
  auto last_output = this->_blocks.back().GetOutput(num_frames);
  const long buffer_offset = this->_input_buffer_offset;
  const long buffer_i_end = buffer_offset + num_frames;
  // last_output is (channels x num_frames), _block_vals[0] is (channels x buffer_size)
  // Copy to the correct location in _block_vals
  this->_block_vals[0].block(0, buffer_offset, last_output.rows(), num_frames) = last_output;

  // Process head - outputs all channels at once
  // Head will resize _head_output internally
  this->_head.process_(this->_block_vals[0], this->_head_output, buffer_offset, buffer_i_end);

  // Copy to output arrays for each channel
  for (int ch = 0; ch < out_channels; ch++)
  {
    for (int s = 0; s < num_frames; s++)
      output[ch][s] = this->_head_output(ch, s);
  }

  // Prepare for next call:
  nam::Buffer::_advance_input_buffer_(num_frames);
}

int nam::convnet::ConvNet::_verify_weights(const int in_channels, const int out_channels, const int channels,
                                           const std::vector<int>& dilations, const bool batchnorm, const int groups,
                                           const size_t actual_weights)
{
  using model_validation::checked_add;
  using model_validation::checked_multiply;
  using model_validation::positive_dimension;

  const std::size_t input_channels = positive_dimension(in_channels, "ConvNet in_channels");
  const std::size_t output_channels = positive_dimension(out_channels, "ConvNet out_channels");
  const std::size_t hidden_channels = positive_dimension(channels, "ConvNet channels");
  const std::size_t group_count = positive_dimension(groups, "ConvNet groups");
  model_validation::require_grouped_channels(in_channels, channels, groups, "ConvNet first layer");

  if (dilations.empty())
    throw std::invalid_argument("ConvNet dilations must not be empty");

  std::size_t max_dilation = 0;
  std::size_t prewarm_samples = 1;
  for (const int dilation : dilations)
  {
    const std::size_t value = positive_dimension(dilation, "ConvNet dilation");
    max_dilation = std::max(max_dilation, value);
    prewarm_samples = checked_add(prewarm_samples, value, "ConvNet prewarm");
  }

  // Buffer currently reserves 32 times the receptive field using int
  // arithmetic. Reject a model that would overflow that calculation before
  // entering Buffer's constructor.
  const std::size_t max_safe_dilation =
    static_cast<std::size_t>(std::numeric_limits<int>::max()) / 32U;
  if (max_dilation > max_safe_dilation)
    throw std::invalid_argument("ConvNet dilation is too large");
  if (prewarm_samples > static_cast<std::size_t>(std::numeric_limits<int>::max()))
    throw std::invalid_argument("ConvNet prewarm size overflows int");
  const std::size_t input_buffer_values = checked_multiply(
    input_channels, checked_multiply(32U, max_dilation, "ConvNet input buffer"), "ConvNet input buffer");
  if (prewarm_samples > kMaxConvNetPrewarmSamples || input_buffer_values > kMaxConvNetBufferValues)
    throw std::invalid_argument("ConvNet buffer exceeds the supported resource limit");
  const std::size_t prewarm_work =
    checked_multiply(dilations.size(), prewarm_samples, "ConvNet prewarm work");
  if (prewarm_work > kMaxConvNetPrewarmWork)
    throw std::invalid_argument("ConvNet prewarm work exceeds the supported resource limit");

  const std::size_t dense_values =
    estimate_convnet_operations_per_sample(in_channels, out_channels, channels, dilations.size());
  if (dense_values > kMaxConvNetDenseValues)
    throw std::invalid_argument("ConvNet dense storage exceeds the supported resource limit");
  const std::size_t prewarm_compute =
    checked_multiply(dense_values, prewarm_samples, "ConvNet prewarm compute");
  if (prewarm_compute > model_validation::kMaxPrewarmComputeOperations)
    throw std::invalid_argument("ConvNet prewarm compute exceeds the supported resource limit");

  const std::size_t base_buffer_columns = checked_multiply(
    2U, checked_add(max_dilation, 32U * kValidationBufferFrames, "ConvNet runtime storage"),
    "ConvNet runtime storage");
  std::size_t runtime_values = checked_multiply(
    checked_add(input_channels, hidden_channels, "ConvNet runtime storage"), base_buffer_columns,
    "ConvNet runtime storage");
  for (std::size_t layer = 0; layer < dilations.size(); ++layer)
  {
    const std::size_t layer_inputs = layer == 0U ? input_channels : hidden_channels;
    const std::size_t history_values = checked_multiply(
      2U, checked_multiply(layer_inputs, static_cast<std::size_t>(dilations[layer]), "ConvNet runtime storage"),
      "ConvNet runtime storage");
    const std::size_t frame_rows = checked_add(
      layer_inputs, checked_multiply(3U, hidden_channels, "ConvNet runtime storage"),
      "ConvNet runtime storage");
    runtime_values = checked_add(runtime_values, history_values, "ConvNet runtime storage");
    runtime_values = checked_add(
      runtime_values, checked_multiply(frame_rows, kValidationBufferFrames, "ConvNet runtime storage"),
      "ConvNet runtime storage");
  }
  runtime_values = checked_add(
    runtime_values,
    checked_multiply(
      checked_add(input_channels, output_channels, "ConvNet runtime storage"), kValidationBufferFrames,
      "ConvNet runtime storage"),
    "ConvNet runtime storage");
  if (runtime_values > kMaxConvNetRuntimeValues)
    throw std::invalid_argument("ConvNet runtime storage exceeds the supported resource limit");

  const std::size_t first_group_inputs = input_channels / group_count;
  const std::size_t hidden_group_inputs = hidden_channels / group_count;
  const std::size_t first_block_weights = checked_multiply(
    2U, checked_multiply(hidden_channels, first_group_inputs, "ConvNet first layer"), "ConvNet first layer");
  const std::size_t later_block_weights = checked_multiply(
    2U, checked_multiply(hidden_channels, hidden_group_inputs, "ConvNet hidden layer"),
    "ConvNet hidden layer");

  std::size_t expected_weights = first_block_weights;
  if (dilations.size() > 1U)
  {
    expected_weights = checked_add(
      expected_weights,
      checked_multiply(dilations.size() - 1U, later_block_weights, "ConvNet hidden layers"),
      "ConvNet hidden layers");
  }

  const std::size_t per_block_tail = batchnorm
    ? checked_add(checked_multiply(4U, hidden_channels, "ConvNet batch normalization"), 1U,
                  "ConvNet batch normalization")
    : hidden_channels;
  expected_weights = checked_add(
    expected_weights, checked_multiply(dilations.size(), per_block_tail, "ConvNet block parameters"),
    "ConvNet block parameters");

  const std::size_t head_weights = checked_multiply(
    output_channels, checked_add(hidden_channels, 1U, "ConvNet head"), "ConvNet head");
  expected_weights = checked_add(expected_weights, head_weights, "ConvNet model");
  model_validation::require_exact_weight_count("ConvNet", expected_weights, actual_weights);

  return static_cast<int>(max_dilation);
}

void nam::convnet::ConvNet::SetMaxBufferSize(const int maxBufferSize)
{
  nam::Buffer::SetMaxBufferSize(maxBufferSize);

  // Reset all ConvNetBlock instances with the new buffer size
  for (auto& block : _blocks)
  {
    block.SetMaxBufferSize(maxBufferSize);
  }
}

void nam::convnet::ConvNet::_update_buffers_(NAM_SAMPLE** input, const int num_frames)
{
  this->Buffer::_update_buffers_(input, num_frames);

  // All channels use the same buffer size
  const long buffer_size = (long)this->_input_buffers[0].size();

  // Only need _block_vals[0] for the head
  // Conv1D layers manage their own buffers now
  if (this->_block_vals[0].rows() != this->_blocks.back().get_out_channels()
      || this->_block_vals[0].cols() != buffer_size)
  {
    this->_block_vals[0].resize(this->_blocks.back().get_out_channels(), buffer_size);
    this->_block_vals[0].setZero();
  }
}

void nam::convnet::ConvNet::_rewind_buffers_()
{
  // Conv1D instances now manage their own ring buffers and handle rewinding internally
  // So we don't need to rewind _block_vals for Conv1D layers
  // We only need _block_vals for the head, and it doesn't need rewinding since it's only used
  // for the current frame range

  // Just rewind the input buffer (for Buffer base class)
  this->Buffer::_rewind_buffers_();
}

// Factory
std::unique_ptr<nam::DSP> nam::convnet::Factory(const nlohmann::json& config, std::vector<float>& weights,
                                                const double expectedSampleRate)
{
  const int channels = model_validation::json_integer_at(config, "channels");
  const std::vector<int> dilations = model_validation::json_integer_vector_at(config, "dilations");
  const bool batchnorm = config.at("batchnorm");
  // Parse JSON into typed ActivationConfig at model loading boundary
  const activations::ActivationConfig activation_config =
    activations::ActivationConfig::from_json(config.at("activation"));
  const int groups = model_validation::json_integer_value(config, "groups", 1); // defaults to 1
  // Default to 1 channel in/out for backward compatibility
  const int in_channels = model_validation::json_integer_value(config, "in_channels", 1);
  const int out_channels = model_validation::json_integer_value(config, "out_channels", 1);
  return std::make_unique<nam::convnet::ConvNet>(
    in_channels, out_channels, channels, dilations, batchnorm, activation_config, weights, expectedSampleRate, groups);
}

namespace
{
static nam::factory::Helper _register_ConvNet("ConvNet", nam::convnet::Factory);
}
