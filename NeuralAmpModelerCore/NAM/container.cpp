#include "container.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>

#include "get_dsp.h"
#include "model_validation.h"
#include "registry.h"

namespace
{
constexpr std::size_t kMaxContainerSubmodels = 16U;

const nam::DSP& FirstModel(const std::vector<nam::container::Submodel>& submodels)
{
  if (submodels.empty() || submodels.front().model == nullptr)
    throw std::invalid_argument("SlimmableContainer requires at least one valid submodel");
  return *submodels.front().model;
}
} // namespace

nam::container::ContainerModel::ContainerModel(std::vector<Submodel> submodels, double expected_sample_rate)
: DSP(FirstModel(submodels).NumInputChannels(), FirstModel(submodels).NumOutputChannels(), expected_sample_rate)
, mSubmodels(std::move(submodels))
{
  if (mSubmodels.size() > kMaxContainerSubmodels)
    throw std::invalid_argument("SlimmableContainer has too many submodels");

  const int input_channels = mSubmodels.front().model->NumInputChannels();
  const int output_channels = mSubmodels.front().model->NumOutputChannels();
  const int global_parameters = mSubmodels.front().model->NumGlobalParams();
  double previous_threshold = 0.0;
  for (const auto& submodel : mSubmodels)
  {
    if (submodel.model == nullptr)
      throw std::invalid_argument("SlimmableContainer contains a null submodel");
    if (!std::isfinite(submodel.max_value) || submodel.max_value <= previous_threshold)
      throw std::invalid_argument("SlimmableContainer thresholds must be finite, positive, and strictly increasing");
    if (submodel.model->NumInputChannels() != input_channels
        || submodel.model->NumOutputChannels() != output_channels)
      throw std::invalid_argument("SlimmableContainer submodels must have matching channel counts");
    if (submodel.model->NumGlobalParams() != global_parameters)
      throw std::invalid_argument("SlimmableContainer submodels must have matching global parameter counts");

    const double submodel_sample_rate = submodel.model->GetExpectedSampleRate();
    if (submodel_sample_rate != expected_sample_rate
        && submodel_sample_rate != NAM_UNKNOWN_EXPECTED_SAMPLE_RATE
        && expected_sample_rate != NAM_UNKNOWN_EXPECTED_SAMPLE_RATE)
      throw std::invalid_argument("SlimmableContainer submodels must have matching sample rates");
    previous_threshold = submodel.max_value;
  }
  if (mSubmodels.back().max_value < 1.0)
    throw std::invalid_argument("SlimmableContainer final threshold must cover 1.0");

  // A2 defaults to its highest-fidelity child.
  mActiveIndex.store(mSubmodels.size() - 1U, std::memory_order_release);
}

void nam::container::ContainerModel::process(NAM_SAMPLE** input, NAM_SAMPLE** output, int num_frames)
{
  const std::size_t index = mActiveIndex.load(std::memory_order_acquire);
  mSubmodels[index].model->process(input, output, num_frames);
}

void nam::container::ContainerModel::process(NAM_SAMPLE** input, NAM_SAMPLE** output, int num_frames,
                                              const double* params, int num_params)
{
  const std::size_t index = mActiveIndex.load(std::memory_order_acquire);
  mSubmodels[index].model->process(input, output, num_frames, params, num_params);
}

void nam::container::ContainerModel::prewarm()
{
  const std::size_t index = mActiveIndex.load(std::memory_order_acquire);
  mSubmodels[index].model->prewarm();
}

void nam::container::ContainerModel::Reset(double sample_rate, int max_buffer_size)
{
  std::lock_guard<std::mutex> lock(mSwitchMutex);
  mExternalSampleRate = sample_rate;
  mHaveExternalSampleRate = true;
  SetMaxBufferSize(max_buffer_size);
  const std::size_t index = mActiveIndex.load(std::memory_order_acquire);
  mSubmodels[index].model->Reset(sample_rate, max_buffer_size);
}

void nam::container::ContainerModel::SetDefaultParams(const double* params, int num_params)
{
  // Keep all children coherent before a later quality switch.
  for (auto& submodel : mSubmodels)
    submodel.model->SetDefaultParams(params, num_params);
}

int nam::container::ContainerModel::NumGlobalParams() const
{
  return mSubmodels.front().model->NumGlobalParams();
}

std::size_t nam::container::ContainerModel::IndexForSize(double value) const
{
  if (!std::isfinite(value) || value < 0.0 || value > 1.0)
    throw std::invalid_argument("Slimmable size must be finite and in [0, 1]");
  for (std::size_t index = 0; index + 1U < mSubmodels.size(); ++index)
  {
    if (value < mSubmodels[index].max_value)
      return index;
  }
  return mSubmodels.size() - 1U;
}

void nam::container::ContainerModel::SetSlimmableSize(double value)
{
  const std::size_t requested = IndexForSize(value);
  if (requested == mActiveIndex.load(std::memory_order_acquire))
    return;

  std::lock_guard<std::mutex> lock(mSwitchMutex);
  if (requested == mActiveIndex.load(std::memory_order_acquire))
    return;

  const int buffer_size = GetMaxBufferSize() > 0 ? GetMaxBufferSize() : 4096;
  const double sample_rate = mHaveExternalSampleRate ? mExternalSampleRate : GetExpectedSampleRate();
  mSubmodels[requested].model->Reset(sample_rate, buffer_size);
  mActiveIndex.store(requested, std::memory_order_release);
}

std::vector<double> nam::container::ContainerModel::GetSlimmableSizeBreakpoints() const
{
  std::vector<double> result;
  result.reserve(mSubmodels.size() - 1U);
  for (std::size_t index = 0; index + 1U < mSubmodels.size(); ++index)
    result.push_back(mSubmodels[index].max_value);
  return result;
}

std::size_t nam::container::ContainerModel::EstimatedOperationsPerSample() const
{
  const std::size_t index = mActiveIndex.load(std::memory_order_acquire);
  return mSubmodels[index].model->EstimatedOperationsPerSample();
}

std::unique_ptr<nam::DSP> nam::container::Factory(const nlohmann::json& config, std::vector<float>& weights,
                                                  double expected_sample_rate)
{
  if (!weights.empty())
    throw std::invalid_argument("SlimmableContainer top-level weights must be empty");
  const auto submodels_it = config.find("submodels");
  if (submodels_it == config.end() || !submodels_it->is_array() || submodels_it->empty())
    throw std::invalid_argument("SlimmableContainer submodels must be a non-empty array");
  if (submodels_it->size() > kMaxContainerSubmodels)
    throw std::invalid_argument("SlimmableContainer has too many submodels");

  std::vector<Submodel> submodels;
  submodels.reserve(submodels_it->size());
  for (const auto& entry : *submodels_it)
  {
    if (!entry.is_object())
      throw std::invalid_argument("SlimmableContainer submodel entries must be objects");
    const double max_value = model_validation::json_float_at(entry, "max_value");
    const auto model_it = entry.find("model");
    if (model_it == entry.end() || !model_it->is_object())
      throw std::invalid_argument("SlimmableContainer submodel is missing its model object");
    // Child loading also primes the A2 fast path's reusable prewarm cache.
    submodels.push_back({max_value, nam::get_dsp(*model_it)});
  }
  return std::make_unique<ContainerModel>(std::move(submodels), expected_sample_rate);
}

namespace
{
static nam::factory::Helper register_slimmable_container("SlimmableContainer", nam::container::Factory);
}
