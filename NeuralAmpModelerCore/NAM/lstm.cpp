#include <algorithm>
#include <cmath>
#include <limits>
#include <string>
#include <vector>
#include <memory>

#include "model_validation.h"
#include "registry.h"
#include "lstm.h"

namespace
{
constexpr std::size_t kMaxLSTMWorkingValues = 1U << 20;
constexpr std::size_t kMaxLSTMPrewarmSamples = 1U << 20;
constexpr std::size_t kMaxLSTMLayers = 16U;

int validate_lstm_model(const int in_channels, const int out_channels, const int num_layers, const int input_size,
                        const int hidden_size, const std::size_t actual_weights, const double expected_sample_rate)
{
  using nam::model_validation::checked_add;
  using nam::model_validation::checked_multiply;
  using nam::model_validation::nonnegative_dimension;
  using nam::model_validation::positive_dimension;

  const std::size_t input_channels = positive_dimension(in_channels, "LSTM in_channels");
  const std::size_t output_channels = positive_dimension(out_channels, "LSTM out_channels");
  const std::size_t layer_count = nonnegative_dimension(num_layers, "LSTM num_layers");
  const std::size_t cell_input_size = positive_dimension(input_size, "LSTM input_size");
  const std::size_t cell_hidden_size = positive_dimension(hidden_size, "LSTM hidden_size");

  if (layer_count > kMaxLSTMLayers)
    throw std::invalid_argument("LSTM num_layers exceeds the supported architecture limit");

  if (cell_input_size < input_channels)
    throw std::invalid_argument("LSTM input_size must be at least in_channels");

  // LSTMCell passes these expressions to Eigen as int dimensions. Check them
  // explicitly so malformed metadata cannot overflow before allocation.
  const std::size_t four_hidden = checked_multiply(4U, cell_hidden_size, "LSTM gate rows");
  const std::size_t input_and_hidden = checked_add(cell_input_size, cell_hidden_size, "LSTM cell columns");
  if (four_hidden > static_cast<std::size_t>(std::numeric_limits<int>::max())
      || input_and_hidden > static_cast<std::size_t>(std::numeric_limits<int>::max()))
  {
    throw std::invalid_argument("LSTM dimensions exceed int range");
  }

  std::size_t working_values = checked_add(cell_input_size, output_channels, "LSTM working storage");
  if (layer_count > 0U)
  {
    working_values = checked_add(
      working_values,
      checked_add(cell_input_size, checked_multiply(10U, cell_hidden_size, "LSTM working storage"),
                  "LSTM working storage"),
      "LSTM working storage");
    if (layer_count > 1U)
    {
      working_values = checked_add(
        working_values,
        checked_multiply(
          layer_count - 1U, checked_multiply(11U, cell_hidden_size, "LSTM working storage"),
          "LSTM working storage"),
        "LSTM working storage");
    }
  }
  if (working_values > kMaxLSTMWorkingValues)
    throw std::invalid_argument("LSTM working storage exceeds the supported resource limit");

  const std::size_t state_weights = checked_multiply(6U, cell_hidden_size, "LSTM cell state");
  const std::size_t first_cell_weights = checked_add(
    checked_multiply(four_hidden, input_and_hidden, "LSTM first cell"), state_weights, "LSTM first cell");
  const std::size_t later_cell_weights = checked_add(
    checked_multiply(8U, checked_multiply(cell_hidden_size, cell_hidden_size, "LSTM hidden cell"),
                     "LSTM hidden cell"),
    state_weights, "LSTM hidden cell");

  std::size_t expected_weights = checked_multiply(
    output_channels, checked_add(cell_hidden_size, 1U, "LSTM head"), "LSTM head");
  if (layer_count > 0U)
  {
    expected_weights = checked_add(expected_weights, first_cell_weights, "LSTM model");
    if (layer_count > 1U)
    {
      expected_weights = checked_add(
        expected_weights, checked_multiply(layer_count - 1U, later_cell_weights, "LSTM hidden layers"),
        "LSTM model");
    }
  }

  nam::model_validation::require_exact_weight_count("LSTM", expected_weights, actual_weights);
  std::size_t prewarm_samples = 1U;
  if (expected_sample_rate != NAM_UNKNOWN_EXPECTED_SAMPLE_RATE)
  {
    const double requested_samples = 0.5 * expected_sample_rate;
    if (!std::isfinite(requested_samples) || requested_samples <= 0.0
        || requested_samples > static_cast<double>(kMaxLSTMPrewarmSamples))
    {
      throw std::invalid_argument("LSTM prewarm exceeds the supported resource limit");
    }
    prewarm_samples = static_cast<std::size_t>(requested_samples);
  }
  const std::size_t prewarm_compute =
    checked_multiply(expected_weights, prewarm_samples, "LSTM prewarm compute");
  if (prewarm_compute > nam::model_validation::kMaxPrewarmComputeOperations)
    throw std::invalid_argument("LSTM prewarm compute exceeds the supported resource limit");
  return in_channels;
}
} // namespace

nam::lstm::LSTMCell::LSTMCell(const int input_size, const int hidden_size, std::vector<float>::iterator& weights)
{
  // Resize arrays
  this->_w.resize(4 * hidden_size, input_size + hidden_size);
  this->_b.resize(4 * hidden_size);
  this->_xh.resize(input_size + hidden_size);
  this->_ifgo.resize(4 * hidden_size);
  this->_c.resize(hidden_size);

  // Assign in row-major because that's how PyTorch goes.
  for (int i = 0; i < this->_w.rows(); i++)
    for (int j = 0; j < this->_w.cols(); j++)
      this->_w(i, j) = *(weights++);
  for (int i = 0; i < this->_b.size(); i++)
    this->_b[i] = *(weights++);
  const int h_offset = input_size;
  for (int i = 0; i < hidden_size; i++)
    this->_xh[i + h_offset] = *(weights++);
  for (int i = 0; i < hidden_size; i++)
    this->_c[i] = *(weights++);
}

void nam::lstm::LSTMCell::process_(const Eigen::VectorXf& x)
{
  const long hidden_size = this->_get_hidden_size();
  const long input_size = this->_get_input_size();
  // Assign inputs
  this->_xh(Eigen::seq(0, input_size - 1)) = x;
  // The matmul
  this->_ifgo = this->_w * this->_xh + this->_b;
  // Elementwise updates (apply nonlinearities here)
  const long i_offset = 0;
  const long f_offset = hidden_size;
  const long g_offset = 2 * hidden_size;
  const long o_offset = 3 * hidden_size;
  const long h_offset = input_size;

  if (activations::Activation::using_fast_tanh)
  {
    for (auto i = 0; i < hidden_size; i++)
      this->_c[i] =
        activations::fast_sigmoid(this->_ifgo[i + f_offset]) * this->_c[i]
        + activations::fast_sigmoid(this->_ifgo[i + i_offset]) * activations::fast_tanh(this->_ifgo[i + g_offset]);

    for (int i = 0; i < hidden_size; i++)
      this->_xh[i + h_offset] =
        activations::fast_sigmoid(this->_ifgo[i + o_offset]) * activations::fast_tanh(this->_c[i]);
  }
  else
  {
    for (auto i = 0; i < hidden_size; i++)
      this->_c[i] = activations::sigmoid(this->_ifgo[i + f_offset]) * this->_c[i]
                    + activations::sigmoid(this->_ifgo[i + i_offset]) * tanhf(this->_ifgo[i + g_offset]);

    for (int i = 0; i < hidden_size; i++)
      this->_xh[i + h_offset] = activations::sigmoid(this->_ifgo[i + o_offset]) * tanhf(this->_c[i]);
  }
}

nam::lstm::LSTM::LSTM(const int in_channels, const int out_channels, const int num_layers, const int input_size,
                      const int hidden_size, std::vector<float>& weights, const double expected_sample_rate)
: DSP(validate_lstm_model(
        in_channels, out_channels, num_layers, input_size, hidden_size, weights.size(), expected_sample_rate),
      out_channels, expected_sample_rate)
{
  this->mEstimatedOperationsPerSample = weights.size();
  // Allocate input and output vectors
  this->_input.resize(input_size);
  this->_input.setZero();
  this->_output.resize(out_channels);

  std::vector<float>::iterator it = weights.begin();
  for (int i = 0; i < num_layers; i++)
    this->_layers.push_back(LSTMCell(i == 0 ? input_size : hidden_size, hidden_size, it));

  // Load head weight as matrix (out_channels x hidden_size)
  // Weights are stored row-major: first row (output 0), then row 1 (output 1), etc.
  this->_head_weight.resize(out_channels, hidden_size);
  for (int out_ch = 0; out_ch < out_channels; out_ch++)
  {
    for (int h = 0; h < hidden_size; h++)
    {
      this->_head_weight(out_ch, h) = *(it++);
    }
  }

  // Load head bias as vector (out_channels)
  this->_head_bias.resize(out_channels);
  for (int out_ch = 0; out_ch < out_channels; out_ch++)
  {
    this->_head_bias(out_ch) = *(it++);
  }

  if (it != weights.end())
    throw std::runtime_error("LSTM internal weight-count validation failed");
}

void nam::lstm::LSTM::process(NAM_SAMPLE** input, NAM_SAMPLE** output, const int num_frames)
{
  const int in_channels = NumInputChannels();
  const int out_channels = NumOutputChannels();

  for (int i = 0; i < num_frames; i++)
  {
    // Copy multi-channel input to _input vector
    for (int ch = 0; ch < in_channels; ch++)
    {
      this->_input(ch) = input[ch][i];
    }

    // Process sample (stores result in _output)
    this->_process_sample();

    // Copy multi-channel output from _output to output arrays
    for (int ch = 0; ch < out_channels; ch++)
    {
      output[ch][i] = this->_output(ch);
    }
  }
}

int nam::lstm::LSTM::PrewarmSamples()
{
  // If the expected sample rate wasn't provided, it'll be -1.
  // Make sure something still happens.
  if (mExpectedSampleRate <= 0.0)
    return 1;
  const double requested_samples = 0.5 * mExpectedSampleRate;
  if (!std::isfinite(requested_samples)
      || requested_samples > static_cast<double>(kMaxLSTMPrewarmSamples))
  {
    throw std::invalid_argument("LSTM prewarm exceeds the supported resource limit");
  }
  return static_cast<int>(requested_samples);
}

void nam::lstm::LSTM::_process_sample()
{
  const int in_channels = NumInputChannels();
  const int out_channels = NumOutputChannels();

  if (this->_layers.size() == 0)
  {
    // No layers - pass input through to output (using first in_channels of output)
    const int channels_to_copy = std::min(in_channels, out_channels);
    for (int ch = 0; ch < channels_to_copy; ch++)
      this->_output(ch) = this->_input(ch);
    // Zero-fill remaining output channels if in_channels < out_channels
    for (int ch = channels_to_copy; ch < out_channels; ch++)
      this->_output(ch) = 0.0f;
    return;
  }

  this->_layers[0].process_(this->_input);
  for (size_t i = 1; i < this->_layers.size(); i++)
    this->_layers[i].process_(this->_layers[i - 1].get_hidden_state());

  // Compute output using head weight matrix and bias vector
  // _output = _head_weight * hidden_state + _head_bias
  const Eigen::VectorXf& hidden_state = this->_layers[this->_layers.size() - 1].get_hidden_state();

  // Compute matrix-vector product: (out_channels x hidden_size) * (hidden_size) = (out_channels)
  // Store directly in _output (which is already sized correctly in constructor)
  this->_output.noalias() = this->_head_weight * hidden_state;

  // Add bias: (out_channels) += (out_channels)
  this->_output.noalias() += this->_head_bias;
}

// Factory to instantiate from nlohmann json
std::unique_ptr<nam::DSP> nam::lstm::Factory(const nlohmann::json& config, std::vector<float>& weights,
                                             const double expectedSampleRate)
{
  const int num_layers = model_validation::json_integer_at(config, "num_layers");
  const int input_size = model_validation::json_integer_at(config, "input_size");
  const int hidden_size = model_validation::json_integer_at(config, "hidden_size");
  // Default to 1 channel in/out for backward compatibility
  const int in_channels = model_validation::json_integer_value(config, "in_channels", 1);
  const int out_channels = model_validation::json_integer_value(config, "out_channels", 1);
  return std::make_unique<nam::lstm::LSTM>(
    in_channels, out_channels, num_layers, input_size, hidden_size, weights, expectedSampleRate);
}

// Register the factory
namespace
{
static nam::factory::Helper _register_LSTM("LSTM", nam::lstm::Factory);
}
