#pragma once

#include <atomic>
#include <memory>
#include <mutex>
#include <vector>

#include "dsp.h"
#include "json.hpp"
#include "slimmable.h"

namespace nam
{
namespace container
{

struct Submodel
{
  double max_value;
  std::unique_ptr<DSP> model;
};

/// Owns the Lite/Full children stored in a NAM SlimmableContainer file.
class ContainerModel : public DSP, public SlimmableModel
{
public:
  ContainerModel(std::vector<Submodel> submodels, double expected_sample_rate);

  void process(NAM_SAMPLE** input, NAM_SAMPLE** output, int num_frames) override;
  void process(NAM_SAMPLE** input, NAM_SAMPLE** output, int num_frames, const double* params,
               int num_params) override;
  void prewarm() override;
  void Reset(double sample_rate, int max_buffer_size) override;
  void SetDefaultParams(const double* params, int num_params) override;
  int NumGlobalParams() const override;
  void SetSlimmableSize(double value) override;
  std::vector<double> GetSlimmableSizeBreakpoints() const override;
  std::size_t EstimatedOperationsPerSample() const override;

private:
  std::size_t IndexForSize(double value) const;

  std::vector<Submodel> mSubmodels;
  std::atomic<std::size_t> mActiveIndex{0};
  std::mutex mSwitchMutex;
};

std::unique_ptr<DSP> Factory(const nlohmann::json& config, std::vector<float>& weights,
                             double expected_sample_rate);

} // namespace container
} // namespace nam
