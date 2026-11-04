// Tests for ConvNet

#include <Eigen/Dense>
#include <cassert>
#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include "NAM/convnet.h"

namespace test_convnet
{
template <typename Function>
void expect_convnet_failure(Function&& function, const std::string& expected_message)
{
  bool threw = false;
  try
  {
    function();
  }
  catch (const std::exception& error)
  {
    threw = true;
    assert(std::string(error.what()).find(expected_message) != std::string::npos);
  }
  assert(threw);
}

// Test basic ConvNet construction and processing
void test_convnet_basic()
{
  const int in_channels = 1;
  const int out_channels = 1;
  const int channels = 2;
  const std::vector<int> dilations{1, 2};
  const bool batchnorm = false;
  const auto activation = nam::activations::ActivationConfig::simple(nam::activations::ActivationType::ReLU);
  const double expected_sample_rate = 48000.0;

  // Calculate weights needed:
  // Block 0: Conv1D (1, 2, 2, !batchnorm=true, 1) -> 2*1*2 = 4 weights + 2 bias = 6 total
  // Block 1: Conv1D (2, 2, 2, !batchnorm=true, 2) -> 2*2*2 = 8 weights + 2 bias = 10 total
  // Head: (2, 1) weight + 1 bias = 3 weights
  // Total: 6 + 10 + 3 = 19 weights
  std::vector<float> weights;
  // Block 0 weights (4 weights: kernel[0] and kernel[1], each 2x1) + 2 bias
  weights.insert(weights.end(), {1.0f, 1.0f, 1.0f, 1.0f, 0.0f, 0.0f});
  // Block 1 weights (8 weights: kernel[0] and kernel[1], each 2x2) + 2 bias
  weights.insert(weights.end(), {1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 0.0f, 0.0f});
  // Head weights (2 weights + 1 bias)
  weights.insert(weights.end(), {1.0f, 1.0f, 0.0f});

  nam::convnet::ConvNet convnet(
    in_channels, out_channels, channels, dilations, batchnorm, activation, weights, expected_sample_rate);

  const int numFrames = 4;
  const int maxBufferSize = 64;
  convnet.Reset(expected_sample_rate, maxBufferSize);

  std::vector<NAM_SAMPLE> input(numFrames, 1.0f);
  std::vector<NAM_SAMPLE> output(numFrames, 0.0f);
  NAM_SAMPLE* inputPtrs[] = {input.data()};
  NAM_SAMPLE* outputPtrs[] = {output.data()};

  convnet.process(inputPtrs, outputPtrs, numFrames);

  // Verify output dimensions
  assert(output.size() == numFrames);
  // Output should be non-zero and finite
  for (int i = 0; i < numFrames; i++)
  {
    assert(std::isfinite(output[i]));
  }
}

// Test ConvNet with batchnorm
void test_convnet_batchnorm()
{
  const int in_channels = 1;
  const int out_channels = 1;
  const int channels = 1;
  const std::vector<int> dilations{1};
  const bool batchnorm = true;
  const auto activation = nam::activations::ActivationConfig::simple(nam::activations::ActivationType::ReLU);
  const double expected_sample_rate = 48000.0;

  // Calculate weights needed:
  // Block 0: Conv1D (1, 1, 2, !batchnorm=false, 1) -> 2*1*1 = 2 weights (no bias when batchnorm=true)
  // BatchNorm: running_mean(1) + running_var(1) + weight(1) + bias(1) + eps(1) = 5 weights
  // Head: (1, 1) weight + 1 bias = 2 weights
  // Total: 2 + 5 + 2 = 9 weights
  std::vector<float> weights;
  // Block 0 weights (2 weights: kernel[0], kernel[1], no bias)
  weights.insert(weights.end(), {1.0f, 1.0f});
  // BatchNorm weights (5: mean, var, weight, bias, eps)
  weights.insert(weights.end(), {0.0f, 1.0f, 1.0f, 0.0f, 1e-5f});
  // Head weights (1 weight + 1 bias)
  weights.insert(weights.end(), {1.0f, 0.0f});

  nam::convnet::ConvNet convnet(
    in_channels, out_channels, channels, dilations, batchnorm, activation, weights, expected_sample_rate);

  const int numFrames = 4;
  const int maxBufferSize = 64;
  convnet.Reset(expected_sample_rate, maxBufferSize);

  std::vector<NAM_SAMPLE> input(numFrames, 1.0f);
  std::vector<NAM_SAMPLE> output(numFrames, 0.0f);
  NAM_SAMPLE* inputPtrs[] = {input.data()};
  NAM_SAMPLE* outputPtrs[] = {output.data()};

  convnet.process(inputPtrs, outputPtrs, numFrames);

  assert(output.size() == numFrames);
  for (int i = 0; i < numFrames; i++)
  {
    assert(std::isfinite(output[i]));
  }
}

// Test ConvNet with multiple blocks
void test_convnet_multiple_blocks()
{
  const int in_channels = 1;
  const int out_channels = 1;
  const int channels = 2;
  const std::vector<int> dilations{1, 2, 4};
  const bool batchnorm = false;
  const auto activation = nam::activations::ActivationConfig::simple(nam::activations::ActivationType::Tanh);
  const double expected_sample_rate = 48000.0;

  // Calculate weights needed:
  // Block 0: Conv1D (1, 2, 2, !batchnorm=true, 1) -> 2*1*2 = 4 weights + 2 bias = 6 total
  // Block 1: Conv1D (2, 2, 2, !batchnorm=true, 2) -> 2*2*2 = 8 weights + 2 bias = 10 total
  // Block 2: Conv1D (2, 2, 2, !batchnorm=true, 4) -> 2*2*2 = 8 weights + 2 bias = 10 total
  // Head: (2, 1) weight + 1 bias = 3 weights
  // Total: 6 + 10 + 10 + 3 = 29 weights
  std::vector<float> weights;
  // Block 0 weights (4 weights + 2 bias)
  weights.insert(weights.end(), {1.0f, 1.0f, 1.0f, 1.0f, 0.0f, 0.0f});
  // Block 1 weights (8 weights + 2 bias)
  weights.insert(weights.end(), {1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 0.0f, 0.0f});
  // Block 2 weights (8 weights + 2 bias)
  weights.insert(weights.end(), {1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 0.0f, 0.0f});
  // Head weights
  weights.insert(weights.end(), {1.0f, 1.0f, 0.0f});

  nam::convnet::ConvNet convnet(
    in_channels, out_channels, channels, dilations, batchnorm, activation, weights, expected_sample_rate);

  const int numFrames = 8;
  const int maxBufferSize = 64;
  convnet.Reset(expected_sample_rate, maxBufferSize);

  std::vector<NAM_SAMPLE> input(numFrames, 0.5f);
  std::vector<NAM_SAMPLE> output(numFrames, 0.0f);
  NAM_SAMPLE* inputPtrs[] = {input.data()};
  NAM_SAMPLE* outputPtrs[] = {output.data()};

  convnet.process(inputPtrs, outputPtrs, numFrames);

  assert(output.size() == numFrames);
  for (int i = 0; i < numFrames; i++)
  {
    assert(std::isfinite(output[i]));
  }
}

// Test ConvNet with zero input
void test_convnet_zero_input()
{
  const int in_channels = 1;
  const int out_channels = 1;
  const int channels = 1;
  const std::vector<int> dilations{1};
  const bool batchnorm = false;
  const auto activation = nam::activations::ActivationConfig::simple(nam::activations::ActivationType::ReLU);
  const double expected_sample_rate = 48000.0;

  std::vector<float> weights;
  // Block 0 weights (2 weights: kernel[0], kernel[1] + 1 bias, since batchnorm=false)
  weights.insert(weights.end(), {1.0f, 1.0f, 0.0f});
  // Head weights (1 weight + 1 bias)
  weights.insert(weights.end(), {1.0f, 0.0f});

  nam::convnet::ConvNet convnet(
    in_channels, out_channels, channels, dilations, batchnorm, activation, weights, expected_sample_rate);

  const int numFrames = 4;
  convnet.Reset(expected_sample_rate, numFrames);

  std::vector<NAM_SAMPLE> input(numFrames, 0.0f);
  std::vector<NAM_SAMPLE> output(numFrames, 0.0f);
  NAM_SAMPLE* inputPtrs[] = {input.data()};
  NAM_SAMPLE* outputPtrs[] = {output.data()};

  convnet.process(inputPtrs, outputPtrs, numFrames);

  // With zero input, output should be finite (may be zero or non-zero depending on bias)
  for (int i = 0; i < numFrames; i++)
  {
    assert(std::isfinite(output[i]));
  }
}

// Test ConvNet with different buffer sizes
void test_convnet_different_buffer_sizes()
{
  const int in_channels = 1;
  const int out_channels = 1;
  const int channels = 1;
  const std::vector<int> dilations{1};
  const bool batchnorm = false;
  const auto activation = nam::activations::ActivationConfig::simple(nam::activations::ActivationType::ReLU);
  const double expected_sample_rate = 48000.0;

  std::vector<float> weights;
  // Block 0 weights (2 weights: kernel[0], kernel[1] + 1 bias, since batchnorm=false)
  weights.insert(weights.end(), {1.0f, 1.0f, 0.0f});
  // Head weights (1 weight + 1 bias)
  weights.insert(weights.end(), {1.0f, 0.0f});

  nam::convnet::ConvNet convnet(
    in_channels, out_channels, channels, dilations, batchnorm, activation, weights, expected_sample_rate);

  // Test with different buffer sizes
  convnet.Reset(expected_sample_rate, 64);
  std::vector<NAM_SAMPLE> input1(32, 1.0f);
  std::vector<NAM_SAMPLE> output1(32, 0.0f);
  NAM_SAMPLE* inputPtrs1[] = {input1.data()};
  NAM_SAMPLE* outputPtrs1[] = {output1.data()};
  convnet.process(inputPtrs1, outputPtrs1, 32);

  convnet.Reset(expected_sample_rate, 128);
  std::vector<NAM_SAMPLE> input2(64, 1.0f);
  std::vector<NAM_SAMPLE> output2(64, 0.0f);
  NAM_SAMPLE* inputPtrs2[] = {input2.data()};
  NAM_SAMPLE* outputPtrs2[] = {output2.data()};
  convnet.process(inputPtrs2, outputPtrs2, 64);

  // Both should work without errors
  assert(output1.size() == 32);
  assert(output2.size() == 64);
}

// Test ConvNet prewarm functionality
void test_convnet_prewarm()
{
  const int in_channels = 1;
  const int out_channels = 1;
  const int channels = 2;
  const std::vector<int> dilations{1, 2, 4};
  const bool batchnorm = false;
  const auto activation = nam::activations::ActivationConfig::simple(nam::activations::ActivationType::ReLU);
  const double expected_sample_rate = 48000.0;

  std::vector<float> weights;
  // Block 0 weights (4 weights + 2 bias, since batchnorm=false)
  weights.insert(weights.end(), {1.0f, 1.0f, 1.0f, 1.0f, 0.0f, 0.0f});
  // Block 1 weights (8 weights + 2 bias, since batchnorm=false)
  weights.insert(weights.end(), {1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 0.0f, 0.0f});
  // Block 2 weights (8 weights + 2 bias, since batchnorm=false)
  weights.insert(weights.end(), {1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 0.0f, 0.0f});
  // Head weights (2 weights + 1 bias)
  weights.insert(weights.end(), {1.0f, 1.0f, 0.0f});

  nam::convnet::ConvNet convnet(
    in_channels, out_channels, channels, dilations, batchnorm, activation, weights, expected_sample_rate);

  // Test that prewarm can be called without errors
  convnet.Reset(expected_sample_rate, 64);
  convnet.prewarm();

  // After prewarm, processing should work
  const int numFrames = 4;
  std::vector<NAM_SAMPLE> input(numFrames, 1.0f);
  std::vector<NAM_SAMPLE> output(numFrames, 0.0f);
  NAM_SAMPLE* inputPtrs[] = {input.data()};
  NAM_SAMPLE* outputPtrs[] = {output.data()};
  convnet.process(inputPtrs, outputPtrs, numFrames);

  // Output should be finite
  for (int i = 0; i < numFrames; i++)
  {
    assert(std::isfinite(output[i]));
  }
}

// Test multiple process() calls (ring buffer functionality)
void test_convnet_multiple_calls()
{
  const int in_channels = 1;
  const int out_channels = 1;
  const int channels = 1;
  const std::vector<int> dilations{1};
  const bool batchnorm = false;
  const auto activation = nam::activations::ActivationConfig::simple(nam::activations::ActivationType::ReLU);
  const double expected_sample_rate = 48000.0;

  std::vector<float> weights;
  // Block 0 weights (2 weights: kernel[0], kernel[1] + 1 bias, since batchnorm=false)
  weights.insert(weights.end(), {1.0f, 1.0f, 0.0f});
  // Head weights (1 weight + 1 bias)
  weights.insert(weights.end(), {1.0f, 0.0f});

  nam::convnet::ConvNet convnet(
    in_channels, out_channels, channels, dilations, batchnorm, activation, weights, expected_sample_rate);

  const int numFrames = 2;
  convnet.Reset(expected_sample_rate, numFrames);

  // Multiple calls should work correctly with ring buffer
  for (int i = 0; i < 5; i++)
  {
    std::vector<NAM_SAMPLE> input(numFrames, 1.0f);
    std::vector<NAM_SAMPLE> output(numFrames, 0.0f);
    NAM_SAMPLE* inputPtrs[] = {input.data()};
    NAM_SAMPLE* outputPtrs[] = {output.data()};
    convnet.process(inputPtrs, outputPtrs, numFrames);

    // Output should be finite
    for (int j = 0; j < numFrames; j++)
    {
      assert(std::isfinite(output[j]));
    }
  }
}

void test_convnet_grouped_weight_count()
{
  const auto activation = nam::activations::ActivationConfig::simple(nam::activations::ActivationType::ReLU);
  // Conv: 2 * 2 * (2 / 2) = 4; bias = 2; head = 1 * (2 + 1) = 3.
  std::vector<float> weights(9, 0.0f);
  nam::convnet::ConvNet convnet(2, 1, 2, {1}, false, activation, weights, 48000.0, 2);
  assert(convnet.NumInputChannels() == 2);
  assert(convnet.NumOutputChannels() == 1);
}

void test_convnet_rejects_short_and_surplus_weights()
{
  const auto activation = nam::activations::ActivationConfig::simple(nam::activations::ActivationType::ReLU);
  // Conv: 2 weights + 1 bias; head: 1 weight + 1 bias.
  std::vector<float> short_weights(4, 0.0f);
  expect_convnet_failure(
    [&]() { nam::convnet::ConvNet model(1, 1, 1, {1}, false, activation, short_weights); },
    "ConvNet weight count mismatch: expected 5, received 4");

  std::vector<float> surplus_weights(6, 0.0f);
  expect_convnet_failure(
    [&]() { nam::convnet::ConvNet model(1, 1, 1, {1}, false, activation, surplus_weights); },
    "ConvNet weight count mismatch: expected 5, received 6");
}

void test_convnet_rejects_malformed_dimensions()
{
  const auto activation = nam::activations::ActivationConfig::simple(nam::activations::ActivationType::ReLU);
  std::vector<float> weights(5, 0.0f);

  expect_convnet_failure(
    [&]() { nam::convnet::ConvNet model(1, 1, 1, {}, false, activation, weights); },
    "dilations must not be empty");
  expect_convnet_failure(
    [&]() { nam::convnet::ConvNet model(1, 1, 1, {0}, false, activation, weights); },
    "dilation must be positive");
  expect_convnet_failure(
    [&]() { nam::convnet::ConvNet model(1, 1, 1, {-1}, false, activation, weights); },
    "dilation must be positive");
  expect_convnet_failure(
    [&]() { nam::convnet::ConvNet model(1, 1, 0, {1}, false, activation, weights); },
    "channels must be positive");
  expect_convnet_failure(
    [&]() { nam::convnet::ConvNet model(1, 1, 1, {1}, false, activation, weights, 48000.0, 0); },
    "groups must be positive");
  expect_convnet_failure(
    [&]() { nam::convnet::ConvNet model(3, 1, 4, {1}, false, activation, weights, 48000.0, 2); },
    "channels must be divisible by groups");
  expect_convnet_failure(
    [&]() {
      nam::convnet::ConvNet model(
        1, 1, 1, {std::numeric_limits<int>::max()}, false, activation, weights);
    },
    "dilation is too large");
  expect_convnet_failure(
    [&]() { nam::convnet::ConvNet model(1, 1, 1, {1 << 20}, false, activation, weights); },
    "buffer exceeds the supported resource limit");
  expect_convnet_failure(
    [&]() { nam::convnet::ConvNet model(5000, 1, 5000, {1}, false, activation, weights, 48000.0, 5000); },
    "dense storage exceeds the supported resource limit");
}
}; // namespace test_convnet
