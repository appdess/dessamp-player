#include <cassert>
#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>
#include <stdexcept>
#include <vector>

#include "NAM/a2_fast.h"
#include "NAM/get_dsp.h"
#include "NAM/slimmable.h"
#include "json.hpp"

namespace test_a2_container
{
namespace
{
nlohmann::json MakeLstm(double output_bias)
{
  std::vector<float> weights(16U, 0.0f);
  weights.back() = static_cast<float>(output_bias);
  return {
    {"version", "0.7.0"},
    {"architecture", "LSTM"},
    {"sample_rate", 48000.0},
    {"metadata", nullptr},
    {"config", {{"input_size", 1}, {"hidden_size", 1}, {"num_layers", 1}}},
    {"weights", weights},
  };
}

nlohmann::json MakeContainer()
{
  return {
    {"version", "0.7.0"},
    {"architecture", "SlimmableContainer"},
    {"sample_rate", 48000.0},
    {"metadata", nullptr},
    {"config",
     {{"submodels",
       nlohmann::json::array(
         {{{"max_value", 0.5}, {"model", MakeLstm(0.1)}}, {{"max_value", 1.0}, {"model", MakeLstm(0.2)}}})}}},
    {"weights", nlohmann::json::array()},
  };
}

nlohmann::json MakeA2Shape()
{
  const std::vector<int> kernels = {6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6,
                                    6, 6, 15, 15, 6, 6, 6, 6, 6, 6, 6};
  const std::vector<int> dilations = {1, 3, 7, 17, 41, 101, 239, 1, 3, 7, 17, 41,
                                      101, 239, 1, 13, 1, 3, 7, 17, 41, 101, 239};
  nlohmann::json activations = nlohmann::json::array();
  nlohmann::json gating = nlohmann::json::array();
  nlohmann::json secondary = nlohmann::json::array();
  for (std::size_t index = 0; index < kernels.size(); ++index)
  {
    activations.push_back({{"type", "LeakyReLU"}, {"negative_slope", 0.01}});
    gating.push_back("none");
    secondary.push_back(nullptr);
  }
  nlohmann::json layer = {
    {"input_size", 1},
    {"condition_size", 1},
    {"head", {{"out_channels", 1}, {"kernel_size", 16}, {"bias", true}}},
    {"channels", 3},
    {"bottleneck", 3},
    {"kernel_sizes", kernels},
    {"dilations", dilations},
    {"activation", activations},
    {"gating_mode", gating},
    {"secondary_activation", secondary},
    {"head1x1", {{"active", false}}},
    {"layer1x1", {{"active", true}, {"groups", 1}}},
    {"groups_input", 1},
    {"groups_input_mixin", 1},
    {"slimmable", nullptr},
  };
  return {{"layers", nlohmann::json::array({layer})}, {"head", nullptr}, {"head_scale", 1.0}};
}

std::vector<float> MakeConditionedA2Weights(int channels, int global_condition_size)
{
  std::vector<float> weights;
  auto append_zeros = [&weights](std::size_t count) { weights.insert(weights.end(), count, 0.0f); };

  // Input rechannel.
  append_zeros(static_cast<std::size_t>(channels));
  for (int layer = 0; layer < nam::wavenet::a2_fast::kNumLayers; ++layer)
  {
    const int kernel = nam::wavenet::a2_fast::kKernelSizes[static_cast<std::size_t>(layer)];
    append_zeros(static_cast<std::size_t>(kernel * channels * channels));
    append_zeros(static_cast<std::size_t>(channels)); // convolution bias

    // Conv1x1 export order is output-major, then input. Give only the first
    // output channel's first global row a unit weight in layer zero.
    for (int output = 0; output < channels; ++output)
    {
      for (int condition = 0; condition < 1 + global_condition_size; ++condition)
      {
        const bool selected = layer == 0 && output == 0 && condition == 1;
        weights.push_back(selected ? 1.0f : 0.0f);
      }
    }

    append_zeros(static_cast<std::size_t>(channels * channels)); // residual 1x1
    append_zeros(static_cast<std::size_t>(channels)); // residual bias
  }

  // Head Conv1d export order is input-major, then tap. Route channel zero's
  // current sample (last tap) to output.
  for (int input = 0; input < channels; ++input)
  {
    for (int tap = 0; tap < nam::wavenet::a2_fast::kHeadKernelSize; ++tap)
      weights.push_back(input == 0 && tap == nam::wavenet::a2_fast::kHeadKernelSize - 1 ? 1.0f : 0.0f);
  }
  weights.push_back(0.0f); // head bias
  weights.push_back(1.0f); // head scale
  return weights;
}

nlohmann::json MakeConditionedA2Model(int channels, int global_condition_size)
{
  auto config = MakeA2Shape();
  config["global_condition_size"] = global_condition_size;
  config["layers"][0]["global_condition_size"] = global_condition_size;
  config["layers"][0]["channels"] = channels;
  config["layers"][0]["bottleneck"] = channels;
  return {
    {"version", "0.7.0"},
    {"architecture", "WaveNet"},
    {"sample_rate", 48000.0},
    {"metadata", nullptr},
    {"config", std::move(config)},
    {"weights", MakeConditionedA2Weights(channels, global_condition_size)},
  };
}
} // namespace

void test_container_defaults_to_full_and_switches()
{
  std::unique_ptr<nam::DSP> model = nam::get_dsp(MakeContainer());
  auto* slimmable = dynamic_cast<nam::SlimmableModel*>(model.get());
  assert(slimmable != nullptr);
  assert(model->NumGlobalParams() == 0);
  const auto breakpoints = slimmable->GetSlimmableSizeBreakpoints();
  assert(breakpoints.size() == 1U);
  assert(std::abs(breakpoints.front() - 0.5) < 1.0e-12);

  model->Reset(48000.0, 4);
  NAM_SAMPLE input_values[4] = {0.0, 0.1, -0.1, 0.0};
  NAM_SAMPLE output_values[4] = {};
  NAM_SAMPLE* inputs[] = {input_values};
  NAM_SAMPLE* outputs[] = {output_values};
  model->process(inputs, outputs, 4);
  for (const auto value : output_values)
    assert(std::abs(value - 0.2) < 1.0e-6);

  slimmable->SetSlimmableSize(0.0);
  model->process(inputs, outputs, 4);
  for (const auto value : output_values)
    assert(std::abs(value - 0.1) < 1.0e-6);

  // Container thresholds are exclusive. Exactly 0.5 selects Full, not Lite.
  slimmable->SetSlimmableSize(0.5);
  model->process(inputs, outputs, 4);
  for (const auto value : output_values)
    assert(std::abs(value - 0.2) < 1.0e-6);
}

void test_a2_detector_accepts_bounded_global_conditioning()
{
  nlohmann::json config = MakeA2Shape();
  int channels = 0;
  assert(nam::wavenet::a2_fast::is_a2_shape(config, &channels));
  assert(channels == 3);
  config["global_condition_size"] = 7;
  config["layers"][0]["global_condition_size"] = 7;
  assert(nam::wavenet::a2_fast::is_a2_shape(config, &channels));
  assert(channels == 3);

  config["layers"][0]["global_condition_size"] = 6;
  assert(!nam::wavenet::a2_fast::is_a2_shape(config, nullptr));

  config = MakeA2Shape();
  config["global_condition_size"] = 65;
  assert(!nam::wavenet::a2_fast::is_a2_shape(config, nullptr));

  config = MakeA2Shape();
  config["global_condition_size"] = 7.0;
  assert(!nam::wavenet::a2_fast::is_a2_shape(config, nullptr));
}

void test_a2_detector_rejects_integer_wraparound()
{
  const auto wrapped = [](const std::uint64_t valid_low_bits) {
    return (std::uint64_t{1} << 32U) + valid_low_bits;
  };

  nlohmann::json config = MakeA2Shape();
  config["global_condition_size"] = wrapped(7U);
  assert(!nam::wavenet::a2_fast::is_a2_shape(config, nullptr));

  config = MakeA2Shape();
  config["in_channels"] = wrapped(1U);
  assert(!nam::wavenet::a2_fast::is_a2_shape(config, nullptr));

  config = MakeA2Shape();
  config["layers"][0]["global_condition_size"] = wrapped(0U);
  assert(!nam::wavenet::a2_fast::is_a2_shape(config, nullptr));

  config = MakeA2Shape();
  config["layers"][0]["channels"] = wrapped(3U);
  assert(!nam::wavenet::a2_fast::is_a2_shape(config, nullptr));

  config = MakeA2Shape();
  config["layers"][0]["kernel_sizes"][0] = wrapped(6U);
  assert(!nam::wavenet::a2_fast::is_a2_shape(config, nullptr));

  config = MakeA2Shape();
  config["layers"][0]["dilations"][0] = wrapped(1U);
  assert(!nam::wavenet::a2_fast::is_a2_shape(config, nullptr));

  config = MakeA2Shape();
  config["layers"][0]["layer1x1"]["groups"] = wrapped(1U);
  assert(!nam::wavenet::a2_fast::is_a2_shape(config, nullptr));

  config = MakeA2Shape();
  config["layers"][0]["head"]["kernel_size"] = wrapped(16U);
  assert(!nam::wavenet::a2_fast::is_a2_shape(config, nullptr));

  config = MakeA2Shape();
  config["layers"][0]["groups_input"] = wrapped(1U);
  assert(!nam::wavenet::a2_fast::is_a2_shape(config, nullptr));
}

void test_conditioned_a2_fast_kernel_uses_controls_and_defaults()
{
  for (const int channels : {3, 8})
  {
    std::unique_ptr<nam::DSP> model = nam::get_dsp(MakeConditionedA2Model(channels, 7));
    assert(model->NumGlobalParams() == 7);
    model->Reset(48000.0, 4);

    NAM_SAMPLE input_values[4] = {};
    NAM_SAMPLE output_values[4] = {};
    NAM_SAMPLE* inputs[] = {input_values};
    NAM_SAMPLE* outputs[] = {output_values};

    // Legacy hosts receive the stable midpoint vector.
    model->process(inputs, outputs, 4);
    for (const auto value : output_values)
      assert(std::abs(value - 0.5) < 1.0e-6);

    const double controls[7] = {0.25, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    model->Reset(48000.0, 4);
    model->process(inputs, outputs, 4, controls, 7);
    for (const auto value : output_values)
      assert(std::abs(value - 0.25) < 1.0e-6);

    const double defaults[7] = {0.75, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    model->SetDefaultParams(defaults, 7);
    model->Reset(48000.0, 4);
    model->process(inputs, outputs, 4);
    for (const auto value : output_values)
      assert(std::abs(value - 0.75) < 1.0e-6);

    bool rejected = false;
    try
    {
      model->process(inputs, outputs, 4, controls, 6);
    }
    catch (const std::invalid_argument&)
    {
      rejected = true;
    }
    assert(rejected);

    double nonfinite[7] = {};
    nonfinite[0] = std::numeric_limits<double>::infinity();
    rejected = false;
    try
    {
      model->process(inputs, outputs, 4, nonfinite, 7);
    }
    catch (const std::invalid_argument&)
    {
      rejected = true;
    }
    assert(rejected);
  }
}

void test_a2_detector_rejects_unsupported_variants()
{
  auto config = MakeA2Shape();
  config["layers"][0]["channels"] = 4;
  config["layers"][0]["bottleneck"] = 4;
  assert(!nam::wavenet::a2_fast::is_a2_shape(config, nullptr));

  config = MakeA2Shape();
  config["layers"][0]["kernel_sizes"][0] = 7;
  assert(!nam::wavenet::a2_fast::is_a2_shape(config, nullptr));

  config = MakeA2Shape();
  config["layers"][0]["activation"][0] = {{"type", "Tanh"}};
  assert(!nam::wavenet::a2_fast::is_a2_shape(config, nullptr));

  config = MakeA2Shape();
  config["layers"][0]["gating_mode"][0] = "gated";
  assert(!nam::wavenet::a2_fast::is_a2_shape(config, nullptr));

  config = MakeA2Shape();
  config["condition_dsp"] = {{"version", "0.5.0"},
                             {"architecture", "Linear"},
                             {"config", nlohmann::json::object()},
                             {"weights", nlohmann::json::array()}};
  assert(!nam::wavenet::a2_fast::is_a2_shape(config, nullptr));

  config = MakeA2Shape();
  config["layers"][0]["gated"] = true;
  assert(!nam::wavenet::a2_fast::is_a2_shape(config, nullptr));
}

} // namespace test_a2_container
