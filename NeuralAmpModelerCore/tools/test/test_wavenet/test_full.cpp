// Tests for full WaveNet model

#include <Eigen/Dense>
#include <cassert>
#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <vector>

#include "NAM/wavenet.h"

namespace test_wavenet
{
namespace test_full
{
// Helper function to create default (inactive) FiLM parameters
static nam::wavenet::_FiLMParams make_default_film_params()
{
  return nam::wavenet::_FiLMParams(false, false);
}

// Helper function to create LayerArrayParams with default FiLM parameters
static nam::wavenet::LayerArrayParams make_layer_array_params(
  const int input_size, const int condition_size, const int head_size, const int channels, const int bottleneck,
  const int kernel_size, std::vector<int>&& dilations, const nam::activations::ActivationConfig& activation_config,
  const nam::wavenet::GatingMode gating_mode, const bool head_bias, const int groups_input,
  const int groups_input_mixin, const int groups_1x1, const nam::wavenet::Head1x1Params& head1x1_params,
  const nam::activations::ActivationConfig& secondary_activation_config)
{
  auto film_params = make_default_film_params();
  return nam::wavenet::LayerArrayParams(
    input_size, condition_size, head_size, channels, bottleneck, kernel_size, std::move(dilations), activation_config,
    gating_mode, head_bias, groups_input, groups_input_mixin, groups_1x1, head1x1_params, secondary_activation_config,
    film_params, film_params, film_params, film_params, film_params, film_params, film_params, film_params);
}

template <typename Callable> static void expect_model_rejected(Callable&& callable)
{
  bool threw = false;
  try
  {
    callable();
  }
  catch (const std::exception&)
  {
    threw = true;
  }
  assert(threw);
}

static void construct_wavenet(const int in_channels,
                              const std::vector<nam::wavenet::LayerArrayParams>& layer_array_params,
                              std::vector<float> weights, const int global_condition_size = 0)
{
  std::unique_ptr<nam::DSP> condition_dsp = nullptr;
  nam::wavenet::WaveNet model(
    in_channels, layer_array_params, 1.0f, false, std::move(weights), std::move(condition_dsp), 48000.0,
    global_condition_size);
}

static nam::wavenet::LayerArrayParams make_simple_validation_params(
  const int input_size = 1, const int condition_size = 1, const int head_size = 1, const int channels = 1,
  const int bottleneck = 1, std::vector<int>&& dilations = std::vector<int>{1}, const int groups_input = 1,
  const nam::wavenet::GatingMode gating_mode = nam::wavenet::GatingMode::NONE, const int kernel_size = 1)
{
  return make_layer_array_params(
    input_size, condition_size, head_size, channels, bottleneck, kernel_size, std::move(dilations),
    nam::activations::ActivationConfig::simple(nam::activations::ActivationType::ReLU), gating_mode, false,
    groups_input, 1, 1, nam::wavenet::Head1x1Params(false, channels, 1),
    nam::activations::ActivationConfig{});
}
// Test full WaveNet model
void test_wavenet_model()
{
  const int input_size = 1;
  const int condition_size = 1;
  const int head_size = 1;
  const int channels = 1;
  const int bottleneck = channels;
  const int kernel_size = 1;
  std::vector<int> dilations{1};
  const auto activation = nam::activations::ActivationConfig::simple(nam::activations::ActivationType::ReLU);
  const nam::wavenet::GatingMode gating_mode = nam::wavenet::GatingMode::NONE;
  const bool head_bias = false;
  const float head_scale = 1.0f;
  const bool with_head = false;
  const int groups = 1;
  const int groups_input_mixin = 1;
  const int groups_1x1 = 1;
  const bool head1x1_active = false;

  nam::wavenet::Head1x1Params head1x1_params(head1x1_active, channels, 1);
  nam::activations::ActivationConfig empty_config{};
  nam::wavenet::LayerArrayParams params = make_layer_array_params(
    input_size, condition_size, head_size, channels, bottleneck, kernel_size, std::move(dilations), activation,
    gating_mode, head_bias, groups, groups_input_mixin, groups_1x1, head1x1_params, empty_config);
  std::vector<nam::wavenet::LayerArrayParams> layer_array_params;
  layer_array_params.push_back(std::move(params));

  // Calculate weights needed
  // Layer array 0:
  //   Rechannel: (1,1) weight
  //   Layer 0: conv (1,1,1) + bias, input_mixin (1,1), 1x1 (1,1) + bias
  //   Head rechannel: (1,1) weight
  // Head scale: 1 float
  std::vector<float> weights;
  weights.push_back(1.0f); // Rechannel
  weights.insert(weights.end(), {1.0f, 0.0f, 1.0f, 1.0f, 0.0f}); // Layer 0
  weights.push_back(1.0f); // Head rechannel
  weights.push_back(head_scale); // Head scale

  std::unique_ptr<nam::wavenet::WaveNet> condition_dsp = nullptr;
  auto wavenet = std::make_unique<nam::wavenet::WaveNet>(
    input_size, layer_array_params, head_scale, with_head, weights, std::move(condition_dsp), 48000.0);

  const int numFrames = 4;
  const int maxBufferSize = 64;
  wavenet->Reset(48000.0, maxBufferSize);

  std::vector<NAM_SAMPLE> input(numFrames, 1.0f);
  std::vector<NAM_SAMPLE> output(numFrames, 0.0f);
  NAM_SAMPLE* inputPtrs[] = {input.data()};
  NAM_SAMPLE* outputPtrs[] = {output.data()};

  wavenet->process(inputPtrs, outputPtrs, numFrames);

  // Verify output dimensions
  assert(output.size() == numFrames);
  // Output should be non-zero
  for (int i = 0; i < numFrames; i++)
  {
    assert(std::isfinite(output[i]));
  }
}

// Test WaveNet with multiple layer arrays
void test_wavenet_multiple_arrays()
{
  const int input_size = 1;
  const int condition_size = 1;
  const int head_size = 1;
  const int channels = 1;
  const int kernel_size = 1;
  std::vector<int> dilations{1};
  const auto activation = nam::activations::ActivationConfig::simple(nam::activations::ActivationType::ReLU);
  const nam::wavenet::GatingMode gating_mode = nam::wavenet::GatingMode::NONE;
  const bool head_bias = false;
  const float head_scale = 0.5f;
  const bool with_head = false;
  const int groups = 1;
  const int groups_input_mixin = 1;

  std::vector<nam::wavenet::LayerArrayParams> layer_array_params;
  // First array
  std::vector<int> dilations1{1};
  const int bottleneck = channels;
  const int groups_1x1 = 1;
  const bool head1x1_active = false;

  nam::wavenet::Head1x1Params head1x1_params(head1x1_active, channels, 1);
  layer_array_params.push_back(make_layer_array_params(input_size, condition_size, head_size, channels, bottleneck,
                                                       kernel_size, std::move(dilations1), activation, gating_mode,
                                                       head_bias, groups, groups_input_mixin, groups_1x1,
                                                       head1x1_params, nam::activations::ActivationConfig{}));
  // Second array (head_size of first must match channels of second)
  std::vector<int> dilations2{1};
  layer_array_params.push_back(make_layer_array_params(head_size, condition_size, head_size, channels, bottleneck,
                                                       kernel_size, std::move(dilations2), activation, gating_mode,
                                                       head_bias, groups, groups_input_mixin, groups_1x1,
                                                       head1x1_params, nam::activations::ActivationConfig{}));

  std::vector<float> weights;
  // Array 0: rechannel, layer, head_rechannel
  weights.insert(weights.end(), {1.0f, 1.0f, 0.0f, 1.0f, 1.0f, 0.0f, 1.0f});
  // Array 1: rechannel, layer, head_rechannel
  weights.insert(weights.end(), {1.0f, 1.0f, 0.0f, 1.0f, 1.0f, 0.0f, 1.0f});
  weights.push_back(head_scale);

  std::unique_ptr<nam::wavenet::WaveNet> condition_dsp = nullptr;
  auto wavenet = std::make_unique<nam::wavenet::WaveNet>(
    input_size, layer_array_params, head_scale, with_head, weights, std::move(condition_dsp), 48000.0);

  const int numFrames = 4;
  const int maxBufferSize = 64;
  wavenet->Reset(48000.0, maxBufferSize);

  std::vector<NAM_SAMPLE> input(numFrames, 1.0f);
  std::vector<NAM_SAMPLE> output(numFrames, 0.0f);
  NAM_SAMPLE* inputPtrs[] = {input.data()};
  NAM_SAMPLE* outputPtrs[] = {output.data()};

  wavenet->process(inputPtrs, outputPtrs, numFrames);

  assert(output.size() == numFrames);
  for (int i = 0; i < numFrames; i++)
  {
    assert(std::isfinite(output[i]));
  }
}

// Test WaveNet with zero input
void test_wavenet_zero_input()
{
  const int input_size = 1;
  const int condition_size = 1;
  const int head_size = 1;
  const int channels = 1;
  const int bottleneck = channels;
  const int kernel_size = 1;
  std::vector<int> dilations{1};
  const auto activation = nam::activations::ActivationConfig::simple(nam::activations::ActivationType::ReLU);
  const nam::wavenet::GatingMode gating_mode = nam::wavenet::GatingMode::NONE;
  const bool head_bias = false;
  const float head_scale = 1.0f;
  const bool with_head = false;
  const int groups = 1;
  const int groups_input_mixin = 1;
  const int groups_1x1 = 1;
  const bool head1x1_active = false;
  nam::wavenet::Head1x1Params head1x1_params(head1x1_active, channels, 1);

  nam::wavenet::LayerArrayParams params =
    make_layer_array_params(input_size, condition_size, head_size, channels, bottleneck, kernel_size,
                            std::move(dilations), activation, gating_mode, head_bias, groups, groups_input_mixin,
                            groups_1x1, head1x1_params, nam::activations::ActivationConfig{});
  std::vector<nam::wavenet::LayerArrayParams> layer_array_params;
  layer_array_params.push_back(std::move(params));

  std::vector<float> weights{1.0f, 1.0f, 0.0f, 1.0f, 1.0f, 0.0f, 1.0f, head_scale};

  std::unique_ptr<nam::wavenet::WaveNet> condition_dsp = nullptr;
  auto wavenet = std::make_unique<nam::wavenet::WaveNet>(
    input_size, layer_array_params, head_scale, with_head, weights, std::move(condition_dsp), 48000.0);

  const int numFrames = 4;
  wavenet->Reset(48000.0, numFrames);

  std::vector<NAM_SAMPLE> input(numFrames, 0.0f);
  std::vector<NAM_SAMPLE> output(numFrames, 0.0f);
  NAM_SAMPLE* inputPtrs[] = {input.data()};
  NAM_SAMPLE* outputPtrs[] = {output.data()};

  wavenet->process(inputPtrs, outputPtrs, numFrames);

  // With zero input, output should be finite (may be zero or non-zero depending on bias)
  for (int i = 0; i < numFrames; i++)
  {
    assert(std::isfinite(output[i]));
  }
}

// Test WaveNet with different buffer sizes
void test_wavenet_different_buffer_sizes()
{
  const int input_size = 1;
  const int condition_size = 1;
  const int head_size = 1;
  const int channels = 1;
  const int bottleneck = channels;
  const int kernel_size = 1;
  std::vector<int> dilations{1};
  const auto activation = nam::activations::ActivationConfig::simple(nam::activations::ActivationType::ReLU);
  const nam::wavenet::GatingMode gating_mode = nam::wavenet::GatingMode::NONE;
  const bool head_bias = false;
  const float head_scale = 1.0f;
  const bool with_head = false;
  const int groups = 1;
  const int groups_input_mixin = 1;
  const int groups_1x1 = 1;
  const bool head1x1_active = false;
  nam::wavenet::Head1x1Params head1x1_params(head1x1_active, channels, 1);

  nam::wavenet::LayerArrayParams params =
    make_layer_array_params(input_size, condition_size, head_size, channels, bottleneck, kernel_size,
                            std::move(dilations), activation, gating_mode, head_bias, groups, groups_input_mixin,
                            groups_1x1, head1x1_params, nam::activations::ActivationConfig{});
  std::vector<nam::wavenet::LayerArrayParams> layer_array_params;
  layer_array_params.push_back(std::move(params));

  std::vector<float> weights{1.0f, 1.0f, 0.0f, 1.0f, 1.0f, 0.0f, 1.0f, head_scale};

  std::unique_ptr<nam::wavenet::WaveNet> condition_dsp = nullptr;
  auto wavenet = std::make_unique<nam::wavenet::WaveNet>(
    input_size, layer_array_params, head_scale, with_head, weights, std::move(condition_dsp), 48000.0);

  // Test with different buffer sizes
  wavenet->Reset(48000.0, 64);
  std::vector<NAM_SAMPLE> input1(32, 1.0f);
  std::vector<NAM_SAMPLE> output1(32, 0.0f);
  NAM_SAMPLE* inputPtrs1[] = {input1.data()};
  NAM_SAMPLE* outputPtrs1[] = {output1.data()};
  wavenet->process(inputPtrs1, outputPtrs1, 32);

  wavenet->Reset(48000.0, 128);
  std::vector<NAM_SAMPLE> input2(64, 1.0f);
  std::vector<NAM_SAMPLE> output2(64, 0.0f);
  NAM_SAMPLE* inputPtrs2[] = {input2.data()};
  NAM_SAMPLE* outputPtrs2[] = {output2.data()};
  wavenet->process(inputPtrs2, outputPtrs2, 64);

  // Both should work without errors
  assert(output1.size() == 32);
  assert(output2.size() == 64);
}

// Test WaveNet prewarm functionality
void test_wavenet_prewarm()
{
  const int input_size = 1;
  const int condition_size = 1;
  const int head_size = 1;
  const int channels = 1;
  const int bottleneck = channels;
  const int kernel_size = 3;
  std::vector<int> dilations{1, 2, 4};
  const auto activation = nam::activations::ActivationConfig::simple(nam::activations::ActivationType::ReLU);
  const nam::wavenet::GatingMode gating_mode = nam::wavenet::GatingMode::NONE;
  const bool head_bias = false;
  const float head_scale = 1.0f;
  const bool with_head = false;
  const int groups = 1;
  const int groups_input_mixin = 1;
  const int groups_1x1 = 1;
  const bool head1x1_active = false;

  nam::wavenet::Head1x1Params head1x1_params(head1x1_active, channels, 1);

  nam::wavenet::LayerArrayParams params =
    make_layer_array_params(input_size, condition_size, head_size, channels, bottleneck, kernel_size,
                            std::move(dilations), activation, gating_mode, head_bias, groups, groups_input_mixin,
                            groups_1x1, head1x1_params, nam::activations::ActivationConfig{});
  std::vector<nam::wavenet::LayerArrayParams> layer_array_params;
  layer_array_params.push_back(std::move(params));

  std::vector<float> weights;
  // Rechannel: (1,1) weight, no bias
  weights.push_back(1.0f);
  // 3 layers: each needs:
  //   Conv: kernel_size=3, in_channels=1, out_channels=1, bias=true -> 3*1*1 + 1 = 4 weights
  //   Input mixin: condition_size=1, out_channels=1, no bias -> 1 weight
  //   1x1: in_channels=1, out_channels=1, bias=true -> 1*1 + 1 = 2 weights
  //   Total per layer: 7 weights
  for (int i = 0; i < 3; i++)
  {
    // Conv weights: 3 weights (kernel_size * in_channels * out_channels) + 1 bias
    weights.insert(weights.end(), {1.0f, 1.0f, 1.0f, 0.0f});
    // Input mixin: 1 weight
    weights.push_back(1.0f);
    // 1x1: 1 weight + 1 bias
    weights.insert(weights.end(), {1.0f, 0.0f});
  }
  // Head rechannel: (1,1) weight, no bias
  weights.push_back(1.0f);
  weights.push_back(head_scale);

  std::unique_ptr<nam::wavenet::WaveNet> condition_dsp = nullptr;
  auto wavenet = std::make_unique<nam::wavenet::WaveNet>(
    input_size, layer_array_params, head_scale, with_head, weights, std::move(condition_dsp), 48000.0);

  // Test that prewarm can be called without errors
  wavenet->Reset(48000.0, 64);
  wavenet->prewarm();

  // After prewarm, processing should work
  const int numFrames = 4;
  std::vector<NAM_SAMPLE> input(numFrames, 1.0f);
  std::vector<NAM_SAMPLE> output(numFrames, 0.0f);
  NAM_SAMPLE* inputPtrs[] = {input.data()};
  NAM_SAMPLE* outputPtrs[] = {output.data()};
  wavenet->process(inputPtrs, outputPtrs, numFrames);

  // Output should be finite
  for (int i = 0; i < numFrames; i++)
  {
    assert(std::isfinite(output[i]));
  }
}

void test_wavenet_exact_weight_count_includes_all_films()
{
  const auto activation = nam::activations::ActivationConfig::simple(nam::activations::ActivationType::Tanh);
  const auto secondary_activation =
    nam::activations::ActivationConfig::simple(nam::activations::ActivationType::Sigmoid);
  const nam::wavenet::_FiLMParams shift_film(true, true);
  const nam::wavenet::_FiLMParams scale_film(true, false);
  std::vector<nam::wavenet::LayerArrayParams> params;
  params.emplace_back(
    2, 2, 3, 4, 2, 3, std::vector<int>{1, 2}, activation, nam::wavenet::GatingMode::GATED, true,
    2, 2, 2, nam::wavenet::Head1x1Params(true, 4, 2), secondary_activation,
    shift_film, scale_film, shift_film, scale_film, shift_film, scale_film, shift_film, scale_film);

  // rechannel 8 + two layers * 174 + biased head rechannel 15 + serialized head scale 1.
  std::vector<float> exact_weights(372, 0.0f);
  exact_weights.back() = 1.0f;
  construct_wavenet(2, params, exact_weights);

  auto short_weights = exact_weights;
  short_weights.pop_back();
  expect_model_rejected([&]() { construct_wavenet(2, params, short_weights); });

  auto surplus_weights = exact_weights;
  surplus_weights.push_back(0.0f);
  expect_model_rejected([&]() { construct_wavenet(2, params, surplus_weights); });
}

void test_wavenet_rejects_short_and_surplus_weights()
{
  const std::vector<nam::wavenet::LayerArrayParams> params{make_simple_validation_params()};
  // Simple model: rechannel 1 + layer 5 + head rechannel 1 + head scale 1.
  construct_wavenet(1, params, std::vector<float>(8, 0.0f));
  expect_model_rejected([&]() { construct_wavenet(1, params, std::vector<float>(7, 0.0f)); });
  expect_model_rejected([&]() { construct_wavenet(1, params, std::vector<float>(9, 0.0f)); });
}

void test_wavenet_rejects_invalid_groups_dilations_and_gating()
{
  const std::vector<float> placeholder_weights(8, 0.0f);
  expect_model_rejected([&]() {
    const std::vector<nam::wavenet::LayerArrayParams> params{
      make_simple_validation_params(1, 1, 1, 1, 1, std::vector<int>{1}, 0)};
    construct_wavenet(1, params, placeholder_weights);
  });
  expect_model_rejected([&]() {
    const std::vector<nam::wavenet::LayerArrayParams> params{
      make_simple_validation_params(1, 1, 1, 1, 1, std::vector<int>{})};
    construct_wavenet(1, params, placeholder_weights);
  });
  expect_model_rejected([&]() {
    const std::vector<nam::wavenet::LayerArrayParams> params{
      make_simple_validation_params(1, 1, 1, 1, 1, std::vector<int>{0})};
    construct_wavenet(1, params, placeholder_weights);
  });
  expect_model_rejected([&]() {
    const auto invalid_gating_mode = static_cast<nam::wavenet::GatingMode>(99);
    const std::vector<nam::wavenet::LayerArrayParams> params{
      make_simple_validation_params(1, 1, 1, 1, 1, std::vector<int>{1}, 1, invalid_gating_mode)};
    construct_wavenet(1, params, placeholder_weights);
  });
  expect_model_rejected([&]() {
    const std::vector<nam::wavenet::LayerArrayParams> params{
      make_simple_validation_params(
        1, 1, 1, 1, 1, std::vector<int>{std::numeric_limits<int>::max()}, 1,
        nam::wavenet::GatingMode::NONE, 2)};
    construct_wavenet(1, params, placeholder_weights);
  });
  expect_model_rejected([&]() {
    const std::vector<nam::wavenet::LayerArrayParams> params{
      make_simple_validation_params(5000, 5000, 1, 5000, 5000, std::vector<int>{1}, 5000)};
    construct_wavenet(5000, params, placeholder_weights);
  });
  expect_model_rejected([&]() {
    const std::vector<nam::wavenet::LayerArrayParams> params{
      make_simple_validation_params(1, 1, 1, 16, 16, std::vector<int>(600, 1))};
    construct_wavenet(1, params, placeholder_weights);
  });
}

void test_wavenet_rejects_input_condition_and_head_chain_mismatches()
{
  const std::vector<float> placeholder_weights(64, 0.0f);
  expect_model_rejected([&]() {
    const std::vector<nam::wavenet::LayerArrayParams> params{
      make_simple_validation_params(2, 1, 1, 1, 1)};
    construct_wavenet(1, params, placeholder_weights);
  });
  expect_model_rejected([&]() {
    const std::vector<nam::wavenet::LayerArrayParams> params{
      make_simple_validation_params(1, 1, 1, 1, 1)};
    construct_wavenet(1, params, placeholder_weights, 1);
  });
  expect_model_rejected([&]() {
    std::vector<nam::wavenet::LayerArrayParams> params;
    params.push_back(make_simple_validation_params(1, 1, 2, 3, 3));
    params.push_back(make_simple_validation_params(4, 1, 1, 2, 2));
    construct_wavenet(1, params, placeholder_weights);
  });
  expect_model_rejected([&]() {
    std::vector<nam::wavenet::LayerArrayParams> params;
    params.push_back(make_simple_validation_params(1, 1, 2, 3, 3));
    params.push_back(make_simple_validation_params(3, 1, 1, 3, 3));
    construct_wavenet(1, params, placeholder_weights);
  });
}

void test_wavenet_rejects_head_film_without_head_projection()
{
  const auto inactive_film = nam::wavenet::_FiLMParams(false, false);
  const auto active_film = nam::wavenet::_FiLMParams(true, false);
  std::vector<nam::wavenet::LayerArrayParams> params;
  params.emplace_back(
    1, 1, 1, 1, 1, 1, std::vector<int>{1},
    nam::activations::ActivationConfig::simple(nam::activations::ActivationType::ReLU),
    nam::wavenet::GatingMode::NONE, false, 1, 1, 1, nam::wavenet::Head1x1Params(false, 1, 1),
    nam::activations::ActivationConfig{}, inactive_film, inactive_film, inactive_film, inactive_film,
    inactive_film, inactive_film, inactive_film, active_film);
  expect_model_rejected([&]() { construct_wavenet(1, params, std::vector<float>(8, 0.0f)); });
}

void test_wavenet_counts_nested_linear_prewarm_compute()
{
  expect_model_rejected([]() {
    const std::vector<nam::wavenet::LayerArrayParams> params{
      make_simple_validation_params(1, 1, 1, 1, 1, std::vector<int>{100000}, 1,
                                    nam::wavenet::GatingMode::NONE, 2)};
    std::vector<float> linear_weights(50000, 0.0f);
    std::unique_ptr<nam::DSP> condition_dsp =
      std::make_unique<nam::Linear>(1, 1, 50000, false, linear_weights, 48000.0);
    nam::wavenet::WaveNet model(
      1, params, 1.0f, false, std::vector<float>{}, std::move(condition_dsp), 48000.0);
  });
}
}; // namespace test_full

} // namespace test_wavenet
