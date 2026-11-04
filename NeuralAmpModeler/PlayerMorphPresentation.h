#pragma once

#include <string_view>

#include "PlayerShellIdentity.h"

namespace dessamp::player
{
enum class MorphPresentationSource
{
  None,
  LearnedCondition,
  DualModel,
};

struct MorphPresentation
{
  MorphPresentationSource source = MorphPresentationSource::None;
  int conditionIndex = -1;
  std::string_view fromLabel;
  std::string_view toLabel;

  bool IsEnabled() const noexcept { return source != MorphPresentationSource::None; }
};

template <typename Controls>
inline MorphPresentation ResolveMorphPresentation(const HeadShellIdentity shellIdentity,
                                                   const bool dualModelMorphEnabled,
                                                   const int globalConditionSize,
                                                   const std::string_view dualModelFromLabel,
                                                   const std::string_view dualModelToLabel,
                                                   const Controls& controls) noexcept
{
  // The historical dual-model runtime owns one appended host slot. Preserve
  // its package-provided labels and index exactly as before.
  if (dualModelMorphEnabled)
  {
    return {
      MorphPresentationSource::DualModel, globalConditionSize, dualModelFromLabel, dualModelToLabel};
  }

  // DESS REDLINE's trained Morph is already one of the primary conditioned
  // model inputs. Bind the presentation to that existing slot; never append a
  // second UI, host, state, or DSP parameter for it.
  if (shellIdentity != HeadShellIdentity::Redline)
    return {};
  for (const auto& control : controls)
  {
    if (control.id == "morph" && control.model_parameter_index >= 0
        && control.model_parameter_index < globalConditionSize && !control.IsSwitch()
        && control.minimum == 0.0 && control.maximum == 1.0)
    {
      return {MorphPresentationSource::LearnedCondition, control.model_parameter_index, "US", "BRIT"};
    }
  }
  return {};
}
} // namespace dessamp::player
