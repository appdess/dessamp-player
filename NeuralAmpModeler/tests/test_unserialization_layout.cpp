#include "../Unserialization.h"

#include <array>
#include <cassert>
#include <cstddef>
#include <iostream>
#include <string_view>

using dessmetal::state::DetectParameterLayout01;
using dessmetal::state::ParameterBytes;
using dessmetal::state::ParameterLayout01;

namespace
{
template <std::size_t N>
void ExpectNames(const std::array<std::string_view, N>& actual,
                 const std::array<std::string_view, N>& expected)
{
  assert(actual == expected);
}
} // namespace

int main()
{
  constexpr std::array<std::string_view, 14> expectedLegacyNames{
    "Input",          "Gain",        "Threshold",            "Bass",       "Middle",
    "Treble",         "Output",      "NoiseGateActive",      "ToneStack",  "IRToggle",
    "CalibrateInput", "InputCalibrationLevel", "OutputMode", "Amp Model",
  };
  constexpr std::array<std::string_view, 20> expectedCurrentNames{
    "Input",          "Gain",         "Threshold",       "Bass",          "Middle",
    "Treble",         "Output",       "Boost Level",     "Boost Tone",    "NoiseGateActive",
    "ToneStack",      "Boost",        "BoostOutput",     "IRToggle",      "Boost Model",
    "CalibrateInput", "InputCalibrationLevel", "OutputMode", "Amp Model", "Amp Enabled",
  };
  constexpr std::array<std::string_view, 29> expectedPlayerNames{
    "Input",          "Gain",         "Threshold",       "Bass",          "Middle",
    "Treble",         "Output",       "Boost Level",     "Boost Tone",    "NoiseGateActive",
    "ToneStack",      "Boost",        "BoostOutput",     "IRToggle",      "Boost Model",
    "CalibrateInput", "InputCalibrationLevel", "OutputMode", "Amp Model", "Amp Enabled",
    "Player Mode",    "Player Condition 1", "Player Condition 2", "Player Condition 3",
    "Player Condition 4", "Player Condition 5", "Player Condition 6", "Player Condition 7",
    "Player Condition 8",
  };

  ExpectNames(dessmetal::state::kLegacy010ParameterNames, expectedLegacyNames);
  ExpectNames(dessmetal::state::kCurrent01ParameterNames, expectedCurrentNames);
  ExpectNames(dessmetal::state::kCurrent02ParameterNames, expectedPlayerNames);

  constexpr auto legacyBytes = ParameterBytes(expectedLegacyNames.size());
  constexpr auto currentBytes = ParameterBytes(expectedCurrentNames.size());
  constexpr auto playerBytes = ParameterBytes(expectedPlayerNames.size());
  static_assert(legacyBytes == 112);
  static_assert(currentBytes == 160);
  static_assert(playerBytes == 232);

  // Exact historical 0.1.0 and current 0.1.x payload shapes are accepted.
  assert(DetectParameterLayout01(legacyBytes, true, false) == ParameterLayout01::Legacy010_14);
  assert(DetectParameterLayout01(legacyBytes, true, true) == ParameterLayout01::Legacy010_14);
  assert(DetectParameterLayout01(legacyBytes + dessmetal::state::kVST3BypassSuffixBytes, true, true)
         == ParameterLayout01::Legacy010_14);
  assert(DetectParameterLayout01(legacyBytes + dessmetal::state::kVST3BypassSuffixBytes, true, false)
         == ParameterLayout01::Invalid);
  assert(DetectParameterLayout01(currentBytes, true, false) == ParameterLayout01::Current20);
  assert(DetectParameterLayout01(currentBytes, true, true) == ParameterLayout01::Current20);
  assert(DetectParameterLayout01(currentBytes + dessmetal::state::kVST3BypassSuffixBytes, true, true)
         == ParameterLayout01::Current20);
  assert(DetectParameterLayout01(currentBytes + dessmetal::state::kVST3BypassSuffixBytes, true, false)
         == ParameterLayout01::Invalid);
  assert(DetectParameterLayout01(currentBytes, false, false) == ParameterLayout01::Current20);

  // A 0.1.1+ state may not claim the old layout, and malformed or ambiguous
  // lengths must never be partially reinterpreted.
  assert(DetectParameterLayout01(legacyBytes, false, false) == ParameterLayout01::Invalid);
  assert(DetectParameterLayout01(legacyBytes + dessmetal::state::kVST3BypassSuffixBytes, false, true)
         == ParameterLayout01::Invalid);
  assert(DetectParameterLayout01(legacyBytes - 1, true, true) == ParameterLayout01::Invalid);
  assert(DetectParameterLayout01(legacyBytes + 1, true, true) == ParameterLayout01::Invalid);
  assert(DetectParameterLayout01(legacyBytes + dessmetal::state::kVST3BypassSuffixBytes + 1, true, true)
         == ParameterLayout01::Invalid);
  assert(DetectParameterLayout01(ParameterBytes(15), true, true) == ParameterLayout01::Invalid);
  assert(DetectParameterLayout01(ParameterBytes(18), true, false) == ParameterLayout01::UnsupportedWip18);
  assert(DetectParameterLayout01(ParameterBytes(18) + dessmetal::state::kVST3BypassSuffixBytes, true, true)
         == ParameterLayout01::UnsupportedWip18);
  assert(DetectParameterLayout01(ParameterBytes(18) + dessmetal::state::kVST3BypassSuffixBytes, true, false)
         == ParameterLayout01::Invalid);
  assert(DetectParameterLayout01(ParameterBytes(19), true, false) == ParameterLayout01::UnsupportedWip19);
  assert(DetectParameterLayout01(ParameterBytes(19) + dessmetal::state::kVST3BypassSuffixBytes, true, true)
         == ParameterLayout01::UnsupportedWip19);
  assert(DetectParameterLayout01(ParameterBytes(19) + dessmetal::state::kVST3BypassSuffixBytes, true, false)
         == ParameterLayout01::Invalid);
  assert(DetectParameterLayout01(ParameterBytes(18), false, true) == ParameterLayout01::Invalid);
  assert(DetectParameterLayout01(ParameterBytes(19), false, true) == ParameterLayout01::Invalid);
  assert(DetectParameterLayout01(currentBytes - 1, true, true) == ParameterLayout01::Invalid);
  assert(DetectParameterLayout01(currentBytes + 1, true, true) == ParameterLayout01::Invalid);
  assert(DetectParameterLayout01(currentBytes + dessmetal::state::kVST3BypassSuffixBytes + 1, true, true)
         == ParameterLayout01::Invalid);

  constexpr std::array<dessmetal::state::NamedDefault, 6> expectedDefaults{
    dessmetal::state::NamedDefault{"Boost Level", 0.0},
    dessmetal::state::NamedDefault{"Boost Tone", 5.0},
    dessmetal::state::NamedDefault{"Boost", 0.0},
    dessmetal::state::NamedDefault{"BoostOutput", 5.0},
    dessmetal::state::NamedDefault{"Boost Model", 0.0},
    dessmetal::state::NamedDefault{"Amp Enabled", 1.0},
  };
  for (std::size_t i = 0; i < expectedDefaults.size(); ++i)
  {
    assert(dessmetal::state::kLegacy010AddedParameterDefaults[i].name == expectedDefaults[i].name);
    assert(dessmetal::state::kLegacy010AddedParameterDefaults[i].value == expectedDefaults[i].value);
  }

  assert(dessmetal::state::IsSupportedLegacy010AmpModel(0.0));
  assert(dessmetal::state::IsSupportedLegacy010AmpModel(1.0));
  assert(!dessmetal::state::IsSupportedLegacy010AmpModel(2.0));
  assert(!dessmetal::state::IsSupportedLegacy010AmpModel(3.0));
  assert(dessmetal::state::MigrateLegacy010AmpModel(0.0) == 0.0);
  assert(dessmetal::state::MigrateLegacy010AmpModel(1.0) == 2.0);

  std::cout << "DessMetal 0.1-to-0.2 state-layout migration tests passed\n";
  return 0;
}
