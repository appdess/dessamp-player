#include "../PlayerMorphPresentation.h"

#include <array>
#include <cassert>
#include <string_view>

namespace
{
struct TestControl
{
  std::string_view id;
  int model_parameter_index = -1;
  double minimum = 0.0;
  double maximum = 1.0;
  bool isSwitch = false;

  bool IsSwitch() const noexcept { return isSwitch; }
};
} // namespace

int main()
{
  using dessamp::player::HeadShellIdentity;
  using dessamp::player::MorphPresentationSource;
  using dessamp::player::ResolveMorphPresentation;

  const std::array<TestControl, 2> learnedControls{{
    {"gain", 0, 0.0, 1.0, false},
    {"morph", 1, 0.0, 1.0, false},
  }};
  const auto learned = ResolveMorphPresentation(
    HeadShellIdentity::Redline, false, 2, "GER", "BRIT", learnedControls);
  assert(learned.source == MorphPresentationSource::LearnedCondition);
  assert(learned.conditionIndex == 1);
  assert(learned.fromLabel == "US");
  assert(learned.toLabel == "BRIT");
  // Presentation reuses condition slot 1 from the two-element trained vector;
  // it does not append a third parameter.
  assert(learned.conditionIndex < static_cast<int>(learnedControls.size()));

  const auto genericLearned = ResolveMorphPresentation(
    HeadShellIdentity::Generic, false, 2, "GER", "BRIT", learnedControls);
  assert(!genericLearned.IsEnabled());

  const std::array<TestControl, 1> primaryControls{{{"gain", 0, 0.0, 1.0, false}}};
  const auto dualModel = ResolveMorphPresentation(
    HeadShellIdentity::Forge, true, 1, "GER", "BRIT", primaryControls);
  assert(dualModel.source == MorphPresentationSource::DualModel);
  assert(dualModel.conditionIndex == 1);
  assert(dualModel.fromLabel == "GER");
  assert(dualModel.toLabel == "BRIT");

  const std::array<TestControl, 2> invalidLearnedControls{{
    {"gain", 0, 0.0, 1.0, false},
    {"morph", 1, 0.1, 0.9, false},
  }};
  const auto invalidLearned = ResolveMorphPresentation(
    HeadShellIdentity::Redline, false, 2, "GER", "BRIT", invalidLearnedControls);
  assert(!invalidLearned.IsEnabled());

  return 0;
}
