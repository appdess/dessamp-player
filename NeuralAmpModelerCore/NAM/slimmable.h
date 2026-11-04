#pragma once

#include <vector>

namespace nam
{

/// Interface implemented by models that can trade fidelity for CPU use.
class SlimmableModel
{
public:
  virtual ~SlimmableModel() = default;

  /// Select a normalized model size. Thread-safe, but not real-time safe.
  virtual void SetSlimmableSize(double value) = 0;

  /// Internal thresholds; 0.0 and 1.0 are implicit outer bounds.
  virtual std::vector<double> GetSlimmableSizeBreakpoints() const = 0;
};

} // namespace nam
