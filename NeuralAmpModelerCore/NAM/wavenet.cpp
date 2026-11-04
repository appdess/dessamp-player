#include <cassert>
#include <cmath>
#include <limits>
#include <sstream>
#include <utility>

#include <Eigen/Dense>

#include "get_dsp.h"
#include "a2_fast.h"
#include "model_validation.h"
#include "registry.h"
#include "wavenet.h"

namespace
{
constexpr int kMaxGlobalConditionSize = 64;
constexpr std::size_t kMaxWaveNetReceptiveFieldSamples = 1U << 20;
constexpr std::size_t kMaxWaveNetHistoryValues = 16U << 20;
constexpr std::size_t kMaxWaveNetDenseValues = 32U << 20;
constexpr std::size_t kMaxWaveNetRuntimeValues = 64U << 20;
constexpr std::size_t kMaxWaveNetPrewarmWork = 64U << 20;
constexpr std::size_t kValidationBufferFrames = 4096;

struct WaveNetValidation
{
  std::size_t weight_count;
  int prewarm_samples;
  std::size_t operations_per_sample;
};

int validated_global_condition_size(const int global_condition_size)
{
  if (global_condition_size < 0 || global_condition_size > kMaxGlobalConditionSize)
    throw std::invalid_argument("global_condition_size is outside the supported range");
  return global_condition_size;
}

std::size_t count_grouped_linear_weights(const int in_channels, const int out_channels, const bool bias,
                                         const int groups, const char* context)
{
  nam::model_validation::require_grouped_channels(in_channels, out_channels, groups, context);
  const auto inputs_per_group = static_cast<std::size_t>(in_channels / groups);
  const auto output_size = static_cast<std::size_t>(out_channels);
  auto result = nam::model_validation::checked_multiply(output_size, inputs_per_group, context);
  if (bias)
    result = nam::model_validation::checked_add(result, output_size, context);
  return result;
}

std::size_t count_grouped_conv_weights(const int in_channels, const int out_channels, const int kernel_size,
                                       const bool bias, const int groups, const char* context)
{
  nam::model_validation::positive_dimension(kernel_size, "kernel_size");
  auto result = count_grouped_linear_weights(in_channels, out_channels, false, groups, context);
  result = nam::model_validation::checked_multiply(result, static_cast<std::size_t>(kernel_size), context);
  if (bias)
    result = nam::model_validation::checked_add(result, static_cast<std::size_t>(out_channels), context);
  return result;
}

int checked_film_output_size(const int input_size, const nam::wavenet::_FiLMParams& params)
{
  const auto multiplier = static_cast<std::size_t>(params.shift ? 2 : 1);
  const auto output_size = nam::model_validation::checked_multiply(
    nam::model_validation::positive_dimension(input_size, "FiLM input_size"), multiplier, "WaveNet FiLM");
  if (output_size > static_cast<std::size_t>(std::numeric_limits<int>::max()))
    throw std::invalid_argument("WaveNet FiLM output size is outside the supported range");
  return static_cast<int>(output_size);
}

std::size_t count_film_weights(const int condition_size, const int input_size,
                               const nam::wavenet::_FiLMParams& params)
{
  if (!params.active)
    return 0;
  return count_grouped_linear_weights(
    condition_size, checked_film_output_size(input_size, params), true, 1, "WaveNet FiLM");
}

std::size_t count_film_runtime_rows(const int input_size, const nam::wavenet::_FiLMParams& params)
{
  if (!params.active)
    return 0;
  return nam::model_validation::checked_add(
    static_cast<std::size_t>(checked_film_output_size(input_size, params)),
    static_cast<std::size_t>(input_size), "WaveNet FiLM runtime storage");
}

WaveNetValidation validate_and_count_wavenet(
  const int in_channels, const std::vector<nam::wavenet::LayerArrayParams>& layer_array_params,
  const bool with_head, const nam::DSP* condition_dsp, const int condition_prewarm,
  const int global_condition_size)
{
  nam::model_validation::positive_dimension(in_channels, "WaveNet in_channels");
  if (layer_array_params.empty())
    throw std::runtime_error("WaveNet requires at least one layer array");
  if (with_head)
    throw std::runtime_error("Head not implemented!");
  if (global_condition_size < 0 || global_condition_size > kMaxGlobalConditionSize)
    throw std::invalid_argument("global_condition_size is outside the supported range");

  int base_condition_size = in_channels;
  if (condition_dsp != nullptr)
  {
    if (condition_dsp->NumInputChannels() != in_channels)
      throw std::runtime_error("WaveNet input channels do not match condition DSP input channels");
    base_condition_size = condition_dsp->NumOutputChannels();
  }
  nam::model_validation::positive_dimension(base_condition_size, "WaveNet condition source size");
  const auto expected_condition_size_value = nam::model_validation::checked_add(
    static_cast<std::size_t>(base_condition_size), static_cast<std::size_t>(global_condition_size),
    "WaveNet condition size");
  if (expected_condition_size_value > static_cast<std::size_t>(std::numeric_limits<int>::max()))
    throw std::invalid_argument("WaveNet condition size is outside the supported range");
  const int expected_condition_size = static_cast<int>(expected_condition_size_value);

  // The final serialized float is the head scale. The condition DSP, if any,
  // owns and validates its separate weight vector before reaching this model.
  std::size_t total_weights = 1;
  std::size_t total_dense_values = 0;
  std::size_t total_runtime_rows = nam::model_validation::checked_add(
    static_cast<std::size_t>(in_channels), static_cast<std::size_t>(expected_condition_size),
    "WaveNet runtime storage");
  std::size_t total_receptive_field = 0;
  std::size_t total_history_values = 0;
  std::size_t total_layer_count = 0;
  int previous_channels = 0;
  int previous_head_size = 0;

  for (std::size_t array_index = 0; array_index < layer_array_params.size(); ++array_index)
  {
    const auto& params = layer_array_params[array_index];
    nam::model_validation::positive_dimension(params.input_size, "WaveNet input_size");
    nam::model_validation::positive_dimension(params.condition_size, "WaveNet condition_size");
    nam::model_validation::positive_dimension(params.head_size, "WaveNet head_size");
    nam::model_validation::positive_dimension(params.channels, "WaveNet channels");
    nam::model_validation::positive_dimension(params.bottleneck, "WaveNet bottleneck");
    nam::model_validation::positive_dimension(params.kernel_size, "WaveNet kernel_size");
    if (params.dilations.empty())
      throw std::invalid_argument("WaveNet layer arrays require at least one dilation");
    total_layer_count = nam::model_validation::checked_add(
      total_layer_count, params.dilations.size(), "WaveNet layer count");
    const auto kernel_lookback =
      nam::model_validation::positive_dimension(params.kernel_size, "WaveNet kernel_size") - 1U;
    for (const int dilation : params.dilations)
    {
      const auto dilation_value = nam::model_validation::positive_dimension(dilation, "WaveNet dilation");
      const auto layer_receptive_field = nam::model_validation::checked_multiply(
        kernel_lookback, dilation_value, "WaveNet receptive field");
      total_receptive_field = nam::model_validation::checked_add(
        total_receptive_field, layer_receptive_field, "WaveNet receptive field");
      total_history_values = nam::model_validation::checked_add(
        total_history_values,
        nam::model_validation::checked_multiply(
          static_cast<std::size_t>(params.channels), layer_receptive_field, "WaveNet history storage"),
        "WaveNet history storage");
    }
    if (total_receptive_field > kMaxWaveNetReceptiveFieldSamples
        || total_history_values > kMaxWaveNetHistoryValues)
    {
      throw std::invalid_argument("WaveNet receptive field exceeds the supported resource limit");
    }

    bool doubled_bottleneck = false;
    switch (params.gating_mode)
    {
      case nam::wavenet::GatingMode::NONE:
        break;
      case nam::wavenet::GatingMode::GATED:
      case nam::wavenet::GatingMode::BLENDED:
        doubled_bottleneck = true;
        break;
      default:
        throw std::invalid_argument("WaveNet gating mode is invalid");
    }

    const auto convolution_output_size = nam::model_validation::checked_multiply(
      static_cast<std::size_t>(params.bottleneck), static_cast<std::size_t>(doubled_bottleneck ? 2 : 1),
      "WaveNet convolution output");
    if (convolution_output_size > static_cast<std::size_t>(std::numeric_limits<int>::max()))
      throw std::invalid_argument("WaveNet convolution output size is outside the supported range");
    const int convolution_outputs = static_cast<int>(convolution_output_size);

    if (params.condition_size != expected_condition_size)
      throw std::runtime_error(
        "WaveNet layer condition_size does not match the condition source plus global condition");
    const int expected_input_size = array_index == 0 ? in_channels : previous_channels;
    if (params.input_size != expected_input_size)
      throw std::runtime_error("WaveNet layer input_size does not match its input source");

    nam::model_validation::require_grouped_channels(
      params.channels, convolution_outputs, params.groups_input, "WaveNet input convolution");
    nam::model_validation::require_grouped_channels(
      params.condition_size, convolution_outputs, params.groups_input_mixin, "WaveNet input mixin");
    nam::model_validation::require_grouped_channels(
      params.bottleneck, params.channels, params.groups_1x1, "WaveNet residual 1x1");

    int layer_head_size = params.bottleneck;
    if (params.head1x1_params.active)
    {
      nam::model_validation::require_grouped_channels(
        params.bottleneck, params.head1x1_params.out_channels, params.head1x1_params.groups,
        "WaveNet head 1x1");
      layer_head_size = params.head1x1_params.out_channels;
    }
    else if (params.head1x1_post_film_params.active)
    {
      throw std::invalid_argument("WaveNet post-head 1x1 FiLM requires an active head 1x1");
    }

    if (array_index > 0 && layer_head_size != previous_head_size)
      throw std::runtime_error("WaveNet accumulated head size does not match the preceding layer array");

    total_dense_values = nam::model_validation::checked_add(
      total_dense_values,
      count_grouped_linear_weights(params.input_size, params.channels, false, 1, "WaveNet dense rechannel"),
      "WaveNet dense storage");
    total_dense_values = nam::model_validation::checked_add(
      total_dense_values,
      count_grouped_linear_weights(layer_head_size, params.head_size, false, 1, "WaveNet dense head"),
      "WaveNet dense storage");
    total_runtime_rows = nam::model_validation::checked_add(
      total_runtime_rows,
      nam::model_validation::checked_add(
        nam::model_validation::checked_multiply(
          2U, static_cast<std::size_t>(params.channels), "WaveNet runtime storage"),
        nam::model_validation::checked_add(
          static_cast<std::size_t>(params.head_size), static_cast<std::size_t>(layer_head_size),
          "WaveNet runtime storage"),
        "WaveNet runtime storage"),
      "WaveNet runtime storage");

    total_weights = nam::model_validation::checked_add(
      total_weights,
      count_grouped_linear_weights(params.input_size, params.channels, false, 1, "WaveNet rechannel"),
      "WaveNet weights");

    std::size_t weights_per_layer = count_grouped_conv_weights(
      params.channels, convolution_outputs, params.kernel_size, true, params.groups_input,
      "WaveNet input convolution");
    weights_per_layer = nam::model_validation::checked_add(
      weights_per_layer,
      count_grouped_linear_weights(
        params.condition_size, convolution_outputs, false, params.groups_input_mixin, "WaveNet input mixin"),
      "WaveNet layer weights");
    weights_per_layer = nam::model_validation::checked_add(
      weights_per_layer,
      count_grouped_linear_weights(
        params.bottleneck, params.channels, true, params.groups_1x1, "WaveNet residual 1x1"),
      "WaveNet layer weights");
    if (params.head1x1_params.active)
    {
      weights_per_layer = nam::model_validation::checked_add(
        weights_per_layer,
        count_grouped_linear_weights(
          params.bottleneck, params.head1x1_params.out_channels, true, params.head1x1_params.groups,
          "WaveNet head 1x1"),
        "WaveNet layer weights");
    }

    std::size_t dense_values_per_layer = count_grouped_conv_weights(
      params.channels, convolution_outputs, params.kernel_size, true, 1, "WaveNet dense input convolution");
    dense_values_per_layer = nam::model_validation::checked_add(
      dense_values_per_layer,
      count_grouped_linear_weights(
        params.condition_size, convolution_outputs, false, 1, "WaveNet dense input mixin"),
      "WaveNet dense storage");
    dense_values_per_layer = nam::model_validation::checked_add(
      dense_values_per_layer,
      count_grouped_linear_weights(params.bottleneck, params.channels, true, 1, "WaveNet dense residual"),
      "WaveNet dense storage");
    if (params.head1x1_params.active)
    {
      dense_values_per_layer = nam::model_validation::checked_add(
        dense_values_per_layer,
        count_grouped_linear_weights(
          params.bottleneck, params.head1x1_params.out_channels, true, 1, "WaveNet dense head 1x1"),
        "WaveNet dense storage");
    }

    weights_per_layer = nam::model_validation::checked_add(
      weights_per_layer, count_film_weights(params.condition_size, params.channels, params.conv_pre_film_params),
      "WaveNet layer weights");
    weights_per_layer = nam::model_validation::checked_add(
      weights_per_layer,
      count_film_weights(params.condition_size, convolution_outputs, params.conv_post_film_params),
      "WaveNet layer weights");
    weights_per_layer = nam::model_validation::checked_add(
      weights_per_layer,
      count_film_weights(params.condition_size, params.condition_size, params.input_mixin_pre_film_params),
      "WaveNet layer weights");
    weights_per_layer = nam::model_validation::checked_add(
      weights_per_layer,
      count_film_weights(params.condition_size, convolution_outputs, params.input_mixin_post_film_params),
      "WaveNet layer weights");
    weights_per_layer = nam::model_validation::checked_add(
      weights_per_layer,
      count_film_weights(params.condition_size, convolution_outputs, params.activation_pre_film_params),
      "WaveNet layer weights");
    weights_per_layer = nam::model_validation::checked_add(
      weights_per_layer,
      count_film_weights(params.condition_size, params.bottleneck, params.activation_post_film_params),
      "WaveNet layer weights");
    weights_per_layer = nam::model_validation::checked_add(
      weights_per_layer,
      count_film_weights(params.condition_size, params.channels, params._1x1_post_film_params),
      "WaveNet layer weights");
    if (params.head1x1_post_film_params.active)
    {
      weights_per_layer = nam::model_validation::checked_add(
        weights_per_layer,
        count_film_weights(
          params.condition_size, params.head1x1_params.out_channels, params.head1x1_post_film_params),
        "WaveNet layer weights");
    }

    dense_values_per_layer = nam::model_validation::checked_add(
      dense_values_per_layer, count_film_weights(params.condition_size, params.channels, params.conv_pre_film_params),
      "WaveNet dense storage");
    dense_values_per_layer = nam::model_validation::checked_add(
      dense_values_per_layer,
      count_film_weights(params.condition_size, convolution_outputs, params.conv_post_film_params),
      "WaveNet dense storage");
    dense_values_per_layer = nam::model_validation::checked_add(
      dense_values_per_layer,
      count_film_weights(params.condition_size, params.condition_size, params.input_mixin_pre_film_params),
      "WaveNet dense storage");
    dense_values_per_layer = nam::model_validation::checked_add(
      dense_values_per_layer,
      count_film_weights(params.condition_size, convolution_outputs, params.input_mixin_post_film_params),
      "WaveNet dense storage");
    dense_values_per_layer = nam::model_validation::checked_add(
      dense_values_per_layer,
      count_film_weights(params.condition_size, convolution_outputs, params.activation_pre_film_params),
      "WaveNet dense storage");
    dense_values_per_layer = nam::model_validation::checked_add(
      dense_values_per_layer,
      count_film_weights(params.condition_size, params.bottleneck, params.activation_post_film_params),
      "WaveNet dense storage");
    dense_values_per_layer = nam::model_validation::checked_add(
      dense_values_per_layer,
      count_film_weights(params.condition_size, params.channels, params._1x1_post_film_params),
      "WaveNet dense storage");
    if (params.head1x1_post_film_params.active)
    {
      dense_values_per_layer = nam::model_validation::checked_add(
        dense_values_per_layer,
        count_film_weights(
          params.condition_size, params.head1x1_params.out_channels, params.head1x1_post_film_params),
        "WaveNet dense storage");
    }

    std::size_t runtime_rows_per_layer = nam::model_validation::checked_add(
      nam::model_validation::checked_multiply(
        3U, static_cast<std::size_t>(params.channels), "WaveNet runtime storage"),
      nam::model_validation::checked_add(
        nam::model_validation::checked_multiply(
          3U, static_cast<std::size_t>(convolution_outputs), "WaveNet runtime storage"),
        static_cast<std::size_t>(layer_head_size), "WaveNet runtime storage"),
      "WaveNet runtime storage");
    runtime_rows_per_layer = nam::model_validation::checked_add(
      runtime_rows_per_layer, count_film_runtime_rows(params.channels, params.conv_pre_film_params),
      "WaveNet runtime storage");
    runtime_rows_per_layer = nam::model_validation::checked_add(
      runtime_rows_per_layer, count_film_runtime_rows(convolution_outputs, params.conv_post_film_params),
      "WaveNet runtime storage");
    runtime_rows_per_layer = nam::model_validation::checked_add(
      runtime_rows_per_layer, count_film_runtime_rows(params.condition_size, params.input_mixin_pre_film_params),
      "WaveNet runtime storage");
    runtime_rows_per_layer = nam::model_validation::checked_add(
      runtime_rows_per_layer, count_film_runtime_rows(convolution_outputs, params.input_mixin_post_film_params),
      "WaveNet runtime storage");
    runtime_rows_per_layer = nam::model_validation::checked_add(
      runtime_rows_per_layer, count_film_runtime_rows(convolution_outputs, params.activation_pre_film_params),
      "WaveNet runtime storage");
    runtime_rows_per_layer = nam::model_validation::checked_add(
      runtime_rows_per_layer, count_film_runtime_rows(params.bottleneck, params.activation_post_film_params),
      "WaveNet runtime storage");
    runtime_rows_per_layer = nam::model_validation::checked_add(
      runtime_rows_per_layer, count_film_runtime_rows(params.channels, params._1x1_post_film_params),
      "WaveNet runtime storage");
    runtime_rows_per_layer = nam::model_validation::checked_add(
      runtime_rows_per_layer,
      count_film_runtime_rows(params.head1x1_params.out_channels, params.head1x1_post_film_params),
      "WaveNet runtime storage");

    total_dense_values = nam::model_validation::checked_add(
      total_dense_values,
      nam::model_validation::checked_multiply(
        dense_values_per_layer, params.dilations.size(), "WaveNet dense storage"),
      "WaveNet dense storage");
    total_runtime_rows = nam::model_validation::checked_add(
      total_runtime_rows,
      nam::model_validation::checked_multiply(
        runtime_rows_per_layer, params.dilations.size(), "WaveNet runtime storage"),
      "WaveNet runtime storage");

    total_weights = nam::model_validation::checked_add(
      total_weights,
      nam::model_validation::checked_multiply(
        weights_per_layer, params.dilations.size(), "WaveNet layer-array weights"),
      "WaveNet weights");
    total_weights = nam::model_validation::checked_add(
      total_weights,
      count_grouped_linear_weights(layer_head_size, params.head_size, params.head_bias, 1, "WaveNet head rechannel"),
      "WaveNet weights");

    previous_channels = params.channels;
    previous_head_size = params.head_size;
  }
  if (total_dense_values > kMaxWaveNetDenseValues)
    throw std::invalid_argument("WaveNet dense storage exceeds the supported resource limit");
  const auto total_runtime_values = nam::model_validation::checked_add(
    nam::model_validation::checked_multiply(
      total_runtime_rows, kValidationBufferFrames, "WaveNet runtime storage"),
    nam::model_validation::checked_multiply(2U, total_history_values, "WaveNet runtime storage"),
    "WaveNet runtime storage");
  if (total_runtime_values > kMaxWaveNetRuntimeValues)
    throw std::invalid_argument("WaveNet runtime storage exceeds the supported resource limit");
  if (condition_prewarm < 0)
    throw std::invalid_argument("WaveNet condition prewarm must not be negative");
  const auto total_prewarm = nam::model_validation::checked_add(
    static_cast<std::size_t>(condition_prewarm), total_receptive_field, "WaveNet prewarm");
  if (total_prewarm > kMaxWaveNetReceptiveFieldSamples
      || total_prewarm > static_cast<std::size_t>(std::numeric_limits<int>::max()))
  {
    throw std::invalid_argument("WaveNet prewarm exceeds the supported resource limit");
  }
  const auto prewarm_work =
    nam::model_validation::checked_multiply(total_layer_count, total_prewarm, "WaveNet prewarm work");
  if (prewarm_work > kMaxWaveNetPrewarmWork)
    throw std::invalid_argument("WaveNet prewarm work exceeds the supported resource limit");
  const std::size_t condition_operations =
    condition_dsp == nullptr ? 0U : condition_dsp->EstimatedOperationsPerSample();
  const std::size_t operations_per_sample = nam::model_validation::checked_add(
    total_dense_values, condition_operations, "WaveNet operations per sample");
  const std::size_t prewarm_compute = nam::model_validation::checked_multiply(
    operations_per_sample, total_prewarm, "WaveNet prewarm compute");
  if (prewarm_compute > nam::model_validation::kMaxPrewarmComputeOperations)
    throw std::invalid_argument("WaveNet prewarm compute exceeds the supported resource limit");
  return {total_weights, static_cast<int>(total_prewarm), operations_per_sample};
}
}

// Layer ======================================================================

void nam::wavenet::_Layer::SetMaxBufferSize(const int maxBufferSize)
{
  _conv.SetMaxBufferSize(maxBufferSize);
  _input_mixin.SetMaxBufferSize(maxBufferSize);
  const long z_channels = this->_conv.get_out_channels(); // This is 2*bottleneck when gated, bottleneck when not
  _z.resize(z_channels, maxBufferSize);
  _1x1.SetMaxBufferSize(maxBufferSize);
  // Pre-allocate output buffers
  const long channels = this->get_channels();
  this->_output_next_layer.resize(channels, maxBufferSize);
  // _output_head stores the activated portion: bottleneck rows when no head1x1, or head1x1 out_channels when head1x1 is
  // active
  if (_head1x1)
  {
    this->_output_head.resize(_head1x1->get_out_channels(), maxBufferSize);
    this->_output_head.setZero(); // Ensure consistent initialization across platforms
    _head1x1->SetMaxBufferSize(maxBufferSize);
  }
  else
  {
    this->_output_head.resize(this->_bottleneck, maxBufferSize);
    this->_output_head.setZero(); // Ensure consistent initialization across platforms
  }
  // Set max buffer size for FiLM objects
  if (this->_conv_pre_film)
    this->_conv_pre_film->SetMaxBufferSize(maxBufferSize);
  if (this->_conv_post_film)
    this->_conv_post_film->SetMaxBufferSize(maxBufferSize);
  if (this->_input_mixin_pre_film)
    this->_input_mixin_pre_film->SetMaxBufferSize(maxBufferSize);
  if (this->_input_mixin_post_film)
    this->_input_mixin_post_film->SetMaxBufferSize(maxBufferSize);
  if (this->_activation_pre_film)
    this->_activation_pre_film->SetMaxBufferSize(maxBufferSize);
  if (this->_activation_post_film)
    this->_activation_post_film->SetMaxBufferSize(maxBufferSize);
  if (this->_1x1_post_film)
    this->_1x1_post_film->SetMaxBufferSize(maxBufferSize);
  if (this->_head1x1_post_film)
    this->_head1x1_post_film->SetMaxBufferSize(maxBufferSize);
}

void nam::wavenet::_Layer::set_weights_(std::vector<float>::iterator& weights)
{
  this->_conv.set_weights_(weights);
  this->_input_mixin.set_weights_(weights);
  this->_1x1.set_weights_(weights);
  if (this->_head1x1)
  {
    this->_head1x1->set_weights_(weights);
  }
  // Set weights for FiLM objects
  if (this->_conv_pre_film)
    this->_conv_pre_film->set_weights_(weights);
  if (this->_conv_post_film)
    this->_conv_post_film->set_weights_(weights);
  if (this->_input_mixin_pre_film)
    this->_input_mixin_pre_film->set_weights_(weights);
  if (this->_input_mixin_post_film)
    this->_input_mixin_post_film->set_weights_(weights);
  if (this->_activation_pre_film)
    this->_activation_pre_film->set_weights_(weights);
  if (this->_activation_post_film)
    this->_activation_post_film->set_weights_(weights);
  if (this->_1x1_post_film)
    this->_1x1_post_film->set_weights_(weights);
  if (this->_head1x1_post_film)
    this->_head1x1_post_film->set_weights_(weights);
}

void nam::wavenet::_Layer::Process(const Eigen::MatrixXf& input, const Eigen::MatrixXf& condition, const int num_frames)
{
  const long bottleneck = this->_bottleneck; // Use the actual bottleneck value, not the doubled output channels

  // Step 1: input convolutions
  if (this->_conv_pre_film)
  {
    // Use Process() instead of Process_() since input is const
    this->_conv_pre_film->Process(input, condition, num_frames);
    this->_conv.Process(this->_conv_pre_film->GetOutput(), num_frames);
  }
  else
  {
    this->_conv.Process(input, num_frames);
  }
  if (this->_conv_post_film)
  {
    Eigen::MatrixXf& conv_output = this->_conv.GetOutput();
    this->_conv_post_film->Process_(conv_output, condition, num_frames);
  }

  if (this->_input_mixin_pre_film)
  {
    // Use Process() instead of Process_() since condition is const
    this->_input_mixin_pre_film->Process(condition, condition, num_frames);
    this->_input_mixin.process_(this->_input_mixin_pre_film->GetOutput(), num_frames);
  }
  else
  {
    this->_input_mixin.process_(condition, num_frames);
  }
  if (this->_input_mixin_post_film)
  {
    Eigen::MatrixXf& input_mixin_output = this->_input_mixin.GetOutput();
    this->_input_mixin_post_film->Process_(input_mixin_output, condition, num_frames);
  }
  this->_z.leftCols(num_frames).noalias() =
    _conv.GetOutput().leftCols(num_frames) + _input_mixin.GetOutput().leftCols(num_frames);
  if (this->_activation_pre_film)
  {
    this->_activation_pre_film->Process_(this->_z, condition, num_frames);
  }

  // Step 2 & 3: activation and 1x1
  //
  // A note about the gating/blending activations:
  // They take 2x dimension as input.
  // The top channels are for the "primary" activation and will be in-place modified for the final result.
  // The bottom channels are for the "secondary" activation and should not be used post-activation.
  if (this->_gating_mode == GatingMode::NONE)
  {
    this->_activation->apply(this->_z.leftCols(num_frames));
    if (this->_activation_post_film)
    {
      this->_activation_post_film->Process_(this->_z, condition, num_frames);
    }
    _1x1.process_(_z, num_frames);
  }
  else if (this->_gating_mode == GatingMode::GATED)
  {
    // Use the GatingActivation class
    // Extract the blocks first to avoid temporary reference issues
    auto input_block = this->_z.leftCols(num_frames);
    auto output_block = this->_z.topRows(bottleneck).leftCols(num_frames);
    this->_gating_activation->apply(input_block, output_block);
    if (this->_activation_post_film)
    {
      // Use Process() for blocks and copy result back
      this->_activation_post_film->Process(this->_z.topRows(bottleneck), condition, num_frames);
      this->_z.topRows(bottleneck).leftCols(num_frames).noalias() =
        this->_activation_post_film->GetOutput().leftCols(num_frames);
    }
    _1x1.process_(this->_z.topRows(bottleneck), num_frames);
  }
  else if (this->_gating_mode == GatingMode::BLENDED)
  {
    // Use the BlendingActivation class
    // Extract the blocks first to avoid temporary reference issues
    auto input_block = this->_z.leftCols(num_frames);
    auto output_block = this->_z.topRows(bottleneck).leftCols(num_frames);
    this->_blending_activation->apply(input_block, output_block);
    if (this->_activation_post_film)
    {
      // Use Process() for blocks and copy result back
      this->_activation_post_film->Process(this->_z.topRows(bottleneck), condition, num_frames);
      this->_z.topRows(bottleneck).leftCols(num_frames).noalias() =
        this->_activation_post_film->GetOutput().leftCols(num_frames);
    }
    _1x1.process_(this->_z.topRows(bottleneck), num_frames);
    if (this->_1x1_post_film)
    {
      Eigen::MatrixXf& _1x1_output = this->_1x1.GetOutput();
      this->_1x1_post_film->Process_(_1x1_output, condition, num_frames);
    }
  }

  if (this->_head1x1)
  {
    if (this->_gating_mode == GatingMode::NONE)
    {
      this->_head1x1->process_(this->_z.leftCols(num_frames), num_frames);
    }
    else
    {
      this->_head1x1->process_(this->_z.topRows(bottleneck).leftCols(num_frames), num_frames);
    }
    this->_head1x1->process(this->_z.topRows(bottleneck).leftCols(num_frames), num_frames);
    if (this->_head1x1_post_film)
    {
      Eigen::MatrixXf& head1x1_output = this->_head1x1->GetOutput();
      this->_head1x1_post_film->Process_(head1x1_output, condition, num_frames);
    }
    this->_output_head.leftCols(num_frames).noalias() = this->_head1x1->GetOutput().leftCols(num_frames);
  }
  else // No head 1x1
  {
    // (No FiLM)
    // Store output to head (skip connection: activated conv output)
    if (this->_gating_mode == GatingMode::NONE)
      this->_output_head.leftCols(num_frames).noalias() = this->_z.leftCols(num_frames);
    else
      this->_output_head.leftCols(num_frames).noalias() = this->_z.topRows(bottleneck).leftCols(num_frames);
  }

  // Store output to next layer (residual connection: input + _1x1 output)
  this->_output_next_layer.leftCols(num_frames).noalias() =
    input.leftCols(num_frames) + _1x1.GetOutput().leftCols(num_frames);
}

// LayerArray =================================================================

nam::wavenet::_LayerArray::_LayerArray(
  const int input_size, const int condition_size, const int head_size, const int channels, const int bottleneck,
  const int kernel_size, const std::vector<int>& dilations, const activations::ActivationConfig& activation_config,
  const GatingMode gating_mode, const bool head_bias, const int groups_input, const int groups_input_mixin,
  const int groups_1x1, const Head1x1Params& head1x1_params,
  const activations::ActivationConfig& secondary_activation_config, const _FiLMParams& conv_pre_film_params,
  const _FiLMParams& conv_post_film_params, const _FiLMParams& input_mixin_pre_film_params,
  const _FiLMParams& input_mixin_post_film_params, const _FiLMParams& activation_pre_film_params,
  const _FiLMParams& activation_post_film_params, const _FiLMParams& _1x1_post_film_params,
  const _FiLMParams& head1x1_post_film_params)
: _rechannel(input_size, channels, false)
, _head_rechannel(head1x1_params.active ? head1x1_params.out_channels : bottleneck, head_size, head_bias)
, _head_output_size(head1x1_params.active ? head1x1_params.out_channels : bottleneck)
{
  for (size_t i = 0; i < dilations.size(); i++)
    this->_layers.push_back(
      _Layer(condition_size, channels, bottleneck, kernel_size, dilations[i], activation_config, gating_mode,
             groups_input, groups_input_mixin, groups_1x1, head1x1_params, secondary_activation_config,
             conv_pre_film_params, conv_post_film_params, input_mixin_pre_film_params, input_mixin_post_film_params,
             activation_pre_film_params, activation_post_film_params, _1x1_post_film_params, head1x1_post_film_params));
}

void nam::wavenet::_LayerArray::SetMaxBufferSize(const int maxBufferSize)
{
  _rechannel.SetMaxBufferSize(maxBufferSize);
  _head_rechannel.SetMaxBufferSize(maxBufferSize);
  for (auto it = _layers.begin(); it != _layers.end(); ++it)
  {
    it->SetMaxBufferSize(maxBufferSize);
  }
  // Pre-allocate output buffers
  const long channels = this->_get_channels();
  this->_layer_outputs.resize(channels, maxBufferSize);
  // _head_inputs size matches actual head output: head1x1.out_channels if active, else bottleneck
  this->_head_inputs.resize(this->_head_output_size, maxBufferSize);
}


long nam::wavenet::_LayerArray::get_receptive_field() const
{
  long result = 0;
  for (size_t i = 0; i < this->_layers.size(); i++)
    result += this->_layers[i].get_dilation() * (this->_layers[i].get_kernel_size() - 1);
  return result;
}


void nam::wavenet::_LayerArray::Process(const Eigen::MatrixXf& layer_inputs, const Eigen::MatrixXf& condition,
                                        const int num_frames)
{
  // Zero head inputs accumulator (first layer array)
  this->_head_inputs.setZero();
  ProcessInner(layer_inputs, condition, num_frames);
}

void nam::wavenet::_LayerArray::Process(const Eigen::MatrixXf& layer_inputs, const Eigen::MatrixXf& condition,
                                        const Eigen::MatrixXf& head_inputs, const int num_frames)
{
  // Copy head inputs from previous layer array
  this->_head_inputs.leftCols(num_frames).noalias() = head_inputs.leftCols(num_frames);
  ProcessInner(layer_inputs, condition, num_frames);
}

void nam::wavenet::_LayerArray::ProcessInner(const Eigen::MatrixXf& layer_inputs, const Eigen::MatrixXf& condition,
                                             const int num_frames)
{
  // Process rechannel and get output
  this->_rechannel.process_(layer_inputs, num_frames);
  Eigen::MatrixXf& rechannel_output = _rechannel.GetOutput();

  // Process layers
  for (size_t i = 0; i < this->_layers.size(); i++)
  {
    // Process first layer with rechannel output, subsequent layers with previous layer output
    // Use separate branches to avoid ternary operator creating temporaries
    if (i == 0)
    {
      // First layer consumes the rechannel output buffer
      this->_layers[i].Process(rechannel_output, condition, num_frames);
    }
    else
    {
      // Subsequent layers consume the full output buffer of the previous layer
      Eigen::MatrixXf& prev_output = this->_layers[i - 1].GetOutputNextLayer();
      this->_layers[i].Process(prev_output, condition, num_frames);
    }

    // Accumulate head output from this layer
    this->_head_inputs.leftCols(num_frames).noalias() += this->_layers[i].GetOutputHead().leftCols(num_frames);
  }

  // Store output from last layer
  const size_t last_layer = this->_layers.size() - 1;
  this->_layer_outputs.leftCols(num_frames).noalias() =
    this->_layers[last_layer].GetOutputNextLayer().leftCols(num_frames);

  // Process head rechannel
  _head_rechannel.process_(this->_head_inputs, num_frames);
}


Eigen::MatrixXf& nam::wavenet::_LayerArray::GetHeadOutputs()
{
  return this->_head_rechannel.GetOutput();
}

const Eigen::MatrixXf& nam::wavenet::_LayerArray::GetHeadOutputs() const
{
  return this->_head_rechannel.GetOutput();
}


void nam::wavenet::_LayerArray::set_weights_(std::vector<float>::iterator& weights)
{
  this->_rechannel.set_weights_(weights);
  for (size_t i = 0; i < this->_layers.size(); i++)
    this->_layers[i].set_weights_(weights);
  this->_head_rechannel.set_weights_(weights);
}

long nam::wavenet::_LayerArray::_get_channels() const
{
  return this->_layers.size() > 0 ? this->_layers[0].get_channels() : 0;
}

// WaveNet ====================================================================

nam::wavenet::WaveNet::WaveNet(const int in_channels,
                               const std::vector<nam::wavenet::LayerArrayParams>& layer_array_params,
                               const float head_scale, const bool with_head, std::vector<float> weights,
                               std::unique_ptr<DSP> condition_dsp, const double expected_sample_rate,
                               const int global_condition_size)
: DSP(in_channels,
      layer_array_params.empty() ? throw std::runtime_error("WaveNet requires at least one layer array")
                                 : layer_array_params.back().head_size,
      expected_sample_rate)
, _condition_dsp(std::move(condition_dsp))
, _head_scale(head_scale)
, _global_condition_size(validated_global_condition_size(global_condition_size))
, _default_global_condition(static_cast<size_t>(_global_condition_size), 0.5)
{
  // Validate the complete outer model before constructing any nested arrays or
  // handing their unchecked set_weights_ methods an iterator.
  const int condition_prewarm = this->_condition_dsp != nullptr ? this->_condition_dsp->PrewarmSamples() : 1;
  const auto validation = validate_and_count_wavenet(
    in_channels, layer_array_params, with_head, this->_condition_dsp.get(), condition_prewarm,
    this->_global_condition_size);
  nam::model_validation::require_exact_weight_count("WaveNet", validation.weight_count, weights.size());
  this->mEstimatedOperationsPerSample = validation.operations_per_sample;

  for (size_t i = 0; i < layer_array_params.size(); i++)
  {
    this->_layer_arrays.push_back(nam::wavenet::_LayerArray(
      layer_array_params[i].input_size, layer_array_params[i].condition_size, layer_array_params[i].head_size,
      layer_array_params[i].channels, layer_array_params[i].bottleneck, layer_array_params[i].kernel_size,
      layer_array_params[i].dilations, layer_array_params[i].activation_config, layer_array_params[i].gating_mode,
      layer_array_params[i].head_bias, layer_array_params[i].groups_input, layer_array_params[i].groups_input_mixin,
      layer_array_params[i].groups_1x1, layer_array_params[i].head1x1_params,
      layer_array_params[i].secondary_activation_config, layer_array_params[i].conv_pre_film_params,
      layer_array_params[i].conv_post_film_params, layer_array_params[i].input_mixin_pre_film_params,
      layer_array_params[i].input_mixin_post_film_params, layer_array_params[i].activation_pre_film_params,
      layer_array_params[i].activation_post_film_params, layer_array_params[i]._1x1_post_film_params,
      layer_array_params[i].head1x1_post_film_params));
  }
  this->set_weights_(weights);

  // Finally, figure out how much pre-warming is needed for this model.
  mPrewarmSamples = validation.prewarm_samples;
}

void nam::wavenet::WaveNet::set_weights_(std::vector<float>& weights)
{
  std::vector<float>::iterator it = weights.begin();
  // Note: condition_dsp already has its own weights from construction,
  // so we don't need to set its weights here.
  for (size_t i = 0; i < this->_layer_arrays.size(); i++)
    this->_layer_arrays[i].set_weights_(it);
  this->_head_scale = *(it++); // TODO `LayerArray.absorb_head_scale()`
  // The outer constructor has already required the exact checked count before
  // this unchecked iterator consumer runs.
  assert(it == weights.end());
}

void nam::wavenet::WaveNet::SetMaxBufferSize(const int maxBufferSize)
{
  DSP::SetMaxBufferSize(maxBufferSize);
  this->_condition_input.resize(this->_get_condition_dim(), maxBufferSize);
  const int condition_output_channels = this->_condition_dsp == nullptr
                                          ? this->_get_condition_dim()
                                          : this->_condition_dsp->NumOutputChannels();
  // Resize condition output
  if (this->_condition_dsp == nullptr)
  {
    this->_condition_output.resize(condition_output_channels + this->_global_condition_size, maxBufferSize);
  }
  else
  {
    this->_condition_dsp->SetMaxBufferSize(maxBufferSize);
    this->_condition_output.resize(condition_output_channels + this->_global_condition_size, maxBufferSize);

    // Resize temporary buffers for condition DSP processing
    const int condition_dim = this->_get_condition_dim();
    this->_condition_dsp_input_buffers.resize(condition_dim);
    this->_condition_dsp_output_buffers.resize(condition_output_channels);
    this->_condition_dsp_input_ptrs.resize(condition_dim);
    this->_condition_dsp_output_ptrs.resize(condition_output_channels);

    for (int ch = 0; ch < condition_dim; ch++)
    {
      this->_condition_dsp_input_buffers[ch].resize(maxBufferSize);
      this->_condition_dsp_input_ptrs[ch] = this->_condition_dsp_input_buffers[ch].data();
    }

    for (int ch = 0; ch < condition_output_channels; ch++)
    {
      this->_condition_dsp_output_buffers[ch].resize(maxBufferSize);
      this->_condition_dsp_output_ptrs[ch] = this->_condition_dsp_output_buffers[ch].data();
    }
  }

  for (size_t i = 0; i < this->_layer_arrays.size(); i++)
    this->_layer_arrays[i].SetMaxBufferSize(maxBufferSize);
}

void nam::wavenet::WaveNet::_process_condition(const int num_frames)
{
  if (this->_condition_dsp == nullptr)
  {
    this->_condition_output.topRows(this->_get_condition_dim()).leftCols(num_frames) =
      this->_condition_input.leftCols(num_frames);
  }
  else
  {
    // Copy input data from Eigen matrix to pre-allocated contiguous buffers
    // Since Eigen matrices are column-major, rows are not contiguous
    // TODO maybe use row-major here?
    const int condition_dim = this->_get_condition_dim();
    for (int ch = 0; ch < condition_dim; ch++)
    {
      for (int j = 0; j < num_frames; j++)
        this->_condition_dsp_input_buffers[ch][j] = (NAM_SAMPLE)this->_condition_input(ch, j);
    }

    // Process through condition DSP using pre-allocated buffers
    this->_condition_dsp->process(
      this->_condition_dsp_input_ptrs.data(), this->_condition_dsp_output_ptrs.data(), num_frames);

    // Copy output data back to Eigen matrix
    const int condition_output_channels = this->_condition_dsp->NumOutputChannels();
    for (int ch = 0; ch < condition_output_channels; ch++)
    {
      for (int j = 0; j < num_frames; j++)
        this->_condition_output(ch, j) = (float)this->_condition_dsp_output_buffers[ch][j];
    }
  }
}

void nam::wavenet::WaveNet::_set_condition_array(NAM_SAMPLE** input, const int num_frames)
{
  const int in_channels = NumInputChannels();
  // Fill condition array with input channels
  for (int ch = 0; ch < in_channels; ch++)
  {
    for (int j = 0; j < num_frames; j++)
    {
      this->_condition_input(ch, j) = input[ch][j];
    }
  }
}

void nam::wavenet::WaveNet::process(NAM_SAMPLE** input, NAM_SAMPLE** output, const int num_frames)
{
  if (this->_global_condition_size == 0)
    return this->_process(input, output, num_frames, nullptr, 0);

  // A stable midpoint is safer than feeding an uninitialised condition when a
  // generic NAM host invokes the legacy, non-parametric interface.
  this->_process(input, output, num_frames, this->_default_global_condition.data(), this->_global_condition_size);
}

void nam::wavenet::WaveNet::process(NAM_SAMPLE** input, NAM_SAMPLE** output, const int num_frames,
                                    const double* params, const int num_params)
{
  this->_process(input, output, num_frames, params, num_params);
}

void nam::wavenet::WaveNet::SetDefaultParams(const double* params, const int num_params)
{
  if (num_params != this->_global_condition_size || (num_params > 0 && params == nullptr))
    throw std::invalid_argument("Default global conditioning vector has the wrong size");
  for (int parameter = 0; parameter < num_params; parameter++)
  {
    if (!std::isfinite(params[parameter]))
      throw std::invalid_argument("Default global conditioning values must be finite");
    this->_default_global_condition[static_cast<size_t>(parameter)] = params[parameter];
  }
}

void nam::wavenet::WaveNet::_process(NAM_SAMPLE** input, NAM_SAMPLE** output, const int num_frames,
                                     const double* params, const int num_params)
{
  assert(num_frames <= mMaxBufferSize);
  if (num_frames < 0 || num_frames > mMaxBufferSize)
    throw std::invalid_argument("Invalid WaveNet frame count");
  if (num_params != this->_global_condition_size || (num_params > 0 && params == nullptr))
    throw std::invalid_argument("Global conditioning vector has the wrong size");
  const int out_channels = NumOutputChannels();

  this->_set_condition_array(input, num_frames);
  this->_process_condition(num_frames);
  const int base_condition_rows = static_cast<int>(this->_condition_output.rows()) - this->_global_condition_size;
  for (int parameter = 0; parameter < this->_global_condition_size; parameter++)
  {
    if (!std::isfinite(params[parameter]))
      throw std::invalid_argument("Global conditioning values must be finite");
    const float value = static_cast<float>(params[parameter]);
    this->_condition_output.row(base_condition_rows + parameter).head(num_frames).setConstant(value);
  }

  // Main layer arrays:
  // Layer-to-layer
  for (size_t i = 0; i < this->_layer_arrays.size(); i++)
  {
    if (i == 0)
    {
      // First layer array - no head input
      // layer_inputs should be the original input (before condition_dsp processing),
      // condition should be the processed condition output (after condition_dsp)
      this->_layer_arrays[i].Process(this->_condition_input, this->_condition_output, num_frames);
    }
    else
    {
      // Subsequent layer arrays - use outputs from previous layer array.
      // Pass full buffers and slice inside the callee to avoid passing Blocks
      // across API boundaries (which can cause Eigen to allocate temporaries).
      Eigen::MatrixXf& prev_layer_outputs = this->_layer_arrays[i - 1].GetLayerOutputs();
      Eigen::MatrixXf& prev_head_outputs = this->_layer_arrays[i - 1].GetHeadOutputs();
      this->_layer_arrays[i].Process(prev_layer_outputs, this->_condition_output, prev_head_outputs, num_frames);
    }
  }

  // (Head not implemented)

  auto& final_head_outputs = this->_layer_arrays.back().GetHeadOutputs();
  assert(final_head_outputs.rows() == out_channels);

  for (int ch = 0; ch < out_channels; ch++)
  {
    for (int s = 0; s < num_frames; s++)
    {
      const float out = this->_head_scale * final_head_outputs(ch, s);
      output[ch][s] = out;
    }
  }
}

// Factory to instantiate from nlohmann json
std::unique_ptr<nam::DSP> nam::wavenet::Factory(const nlohmann::json& config, std::vector<float>& weights,
                                                const double expectedSampleRate)
{
#if NAM_ENABLE_A2_FAST
  int a2_channels = 0;
  if (nam::wavenet::a2_fast::is_a2_shape(config, &a2_channels))
    return nam::wavenet::a2_fast::Factory(config, weights, expectedSampleRate);
#endif
  const int global_condition_size = model_validation::json_integer_value(config, "global_condition_size", 0);
  if (global_condition_size < 0 || global_condition_size > kMaxGlobalConditionSize)
    throw std::invalid_argument("global_condition_size is outside the supported range");
  std::unique_ptr<nam::DSP> condition_dsp = nullptr;
  if (config.find("condition_dsp") != config.end())
  {
    const nlohmann::json& condition_dsp_json = config.at("condition_dsp");
    condition_dsp = nam::get_dsp(condition_dsp_json);
    if (condition_dsp->GetExpectedSampleRate() != expectedSampleRate)
    {
      std::stringstream ss;
      ss << "Condition DSP expected sample rate (" << condition_dsp->GetExpectedSampleRate()
         << ") doesn't match WaveNet expected sample rate (" << expectedSampleRate << "!\n";
      throw std::runtime_error(ss.str().c_str());
    }
  }
  std::vector<nam::wavenet::LayerArrayParams> layer_array_params;
  const nlohmann::json& layers = config.at("layers");
  for (size_t i = 0; i < layers.size(); i++)
  {
    nlohmann::json layer_config = layers.at(i);

    const int groups = model_validation::json_integer_value(layer_config, "groups_input", 1); // defaults to 1
    const int groups_input_mixin =
      model_validation::json_integer_value(layer_config, "groups_input_mixin", 1); // defaults to 1
    const int groups_1x1 = model_validation::json_integer_value(layer_config, "groups_1x1", 1); // defaults to 1

    const int channels = model_validation::json_integer_at(layer_config, "channels");
    const int bottleneck =
      model_validation::json_integer_value(layer_config, "bottleneck", channels); // defaults to channels

    const int input_size = model_validation::json_integer_at(layer_config, "input_size");
    const int layer_global_condition_size =
      model_validation::json_integer_value(layer_config, "global_condition_size", global_condition_size);
    if (layer_global_condition_size != global_condition_size)
      throw std::invalid_argument("Inconsistent global_condition_size in WaveNet layer configuration");
    const int base_condition_size = model_validation::json_integer_at(layer_config, "condition_size");
    if (base_condition_size < 0 || base_condition_size > std::numeric_limits<int>::max() - global_condition_size)
      throw std::invalid_argument("WaveNet condition_size is outside the supported range");
    const int condition_size = base_condition_size + global_condition_size;
    const int head_size = model_validation::json_integer_at(layer_config, "head_size");
    const int kernel_size = model_validation::json_integer_at(layer_config, "kernel_size");
    std::vector<int> dilations = model_validation::json_integer_vector_at(layer_config, "dilations");
    // Parse JSON into typed ActivationConfig at model loading boundary
    const activations::ActivationConfig activation_config =
      activations::ActivationConfig::from_json(layer_config.at("activation"));
    // Parse gating mode - support both old "gated" boolean and new "gating_mode" string
    GatingMode gating_mode = GatingMode::NONE;
    activations::ActivationConfig secondary_activation_config;

    if (layer_config.find("gating_mode") != layer_config.end())
    {
      std::string gating_mode_str = layer_config.at("gating_mode").get<std::string>();
      if (gating_mode_str == "gated")
      {
        gating_mode = GatingMode::GATED;
        secondary_activation_config = activations::ActivationConfig::from_json(layer_config.at("secondary_activation"));
      }
      else if (gating_mode_str == "blended")
      {
        gating_mode = GatingMode::BLENDED;
        secondary_activation_config = activations::ActivationConfig::from_json(layer_config.at("secondary_activation"));
      }
      else if (gating_mode_str == "none")
      {
        gating_mode = GatingMode::NONE;
        // Leave secondary_activation_config with empty type
      }
      else
        throw std::runtime_error("Invalid gating_mode: " + gating_mode_str);
    }
    else if (layer_config.find("gated") != layer_config.end())
    {
      // Backward compatibility: convert old "gated" boolean to new enum
      bool gated = layer_config.at("gated");
      gating_mode = gated ? GatingMode::GATED : GatingMode::NONE;
      if (gated)
      {
        secondary_activation_config = activations::ActivationConfig::simple(activations::ActivationType::Sigmoid);
      }
      // else: leave secondary_activation_config uninitialized
    }
    else
    {
      throw std::invalid_argument("No information on gating mode found for layer array " + std::to_string(i));
    }

    const bool head_bias = layer_config.at("head_bias");

    // Parse head1x1 parameters
    bool head1x1_active = false;
    int head1x1_out_channels = channels;
    int head1x1_groups = 1;
    if (layer_config.find("head_1x1") != layer_config.end())
    {
      const auto& head1x1_config = layer_config.at("head_1x1");
      head1x1_active = head1x1_config.at("active");
      head1x1_out_channels = model_validation::json_integer_at(head1x1_config, "out_channels");
      head1x1_groups = model_validation::json_integer_at(head1x1_config, "groups");
    }
    nam::wavenet::Head1x1Params head1x1_params(head1x1_active, head1x1_out_channels, head1x1_groups);

    // Helper function to parse FiLM parameters
    auto parse_film_params = [&layer_config](const std::string& key) -> nam::wavenet::_FiLMParams {
      if (layer_config.find(key) == layer_config.end() || layer_config.at(key) == false)
      {
        return nam::wavenet::_FiLMParams(false, false);
      }
      const nlohmann::json& film_config = layer_config.at(key);
      bool active = film_config.value("active", true);
      bool shift = film_config.value("shift", true);
      return nam::wavenet::_FiLMParams(active, shift);
    };

    // Parse FiLM parameters
    nam::wavenet::_FiLMParams conv_pre_film_params = parse_film_params("conv_pre_film");
    nam::wavenet::_FiLMParams conv_post_film_params = parse_film_params("conv_post_film");
    nam::wavenet::_FiLMParams input_mixin_pre_film_params = parse_film_params("input_mixin_pre_film");
    nam::wavenet::_FiLMParams input_mixin_post_film_params = parse_film_params("input_mixin_post_film");
    nam::wavenet::_FiLMParams activation_pre_film_params = parse_film_params("activation_pre_film");
    nam::wavenet::_FiLMParams activation_post_film_params = parse_film_params("activation_post_film");
    nam::wavenet::_FiLMParams _1x1_post_film_params = parse_film_params("1x1_post_film");
    nam::wavenet::_FiLMParams head1x1_post_film_params = parse_film_params("head1x1_post_film");

    layer_array_params.push_back(nam::wavenet::LayerArrayParams(
      input_size, condition_size, head_size, channels, bottleneck, kernel_size, std::move(dilations), activation_config,
      gating_mode, head_bias, groups, groups_input_mixin, groups_1x1, head1x1_params, secondary_activation_config,
      conv_pre_film_params, conv_post_film_params, input_mixin_pre_film_params, input_mixin_post_film_params,
      activation_pre_film_params, activation_post_film_params, _1x1_post_film_params, head1x1_post_film_params));
  }
  const bool with_head = !config.at("head").is_null();
  const float head_scale = model_validation::json_float_at(config, "head_scale");

  if (layer_array_params.empty())
    throw std::runtime_error("WaveNet config requires at least one layer array");

  // Backward compatibility: assume 1 input channel
  const int in_channels = model_validation::json_integer_value(config, "in_channels", 1);

  // out_channels is determined from last layer array's head_size
  return std::make_unique<nam::wavenet::WaveNet>(
    in_channels, layer_array_params, head_scale, with_head, weights, std::move(condition_dsp), expectedSampleRate,
    global_condition_size);
}

// Register the factory
namespace
{
static nam::factory::Helper _register_WaveNet("WaveNet", nam::wavenet::Factory);
}
