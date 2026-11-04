#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <string_view>

namespace dessmetal::state
{
// Commit f55d0201 serialized these values positionally as DessMetal 0.1.0.
// The last destination name is the current name of the historical "Amp Switch"
// parameter; parameter names themselves are not present in an IByteChunk.
inline constexpr std::array<std::string_view, 14> kLegacy010ParameterNames{
  "Input",
  "Gain",
  "Threshold",
  "Bass",
  "Middle",
  "Treble",
  "Output",
  "NoiseGateActive",
  "ToneStack",
  "IRToggle",
  "CalibrateInput",
  "InputCalibrationLevel",
  "OutputMode",
  "Amp Model",
};

// The current order must stay identical to EParams/SerializeParams.
inline constexpr std::array<std::string_view, 20> kCurrent01ParameterNames{
  "Input",
  "Gain",
  "Threshold",
  "Bass",
  "Middle",
  "Treble",
  "Output",
  "Boost Level",
  "Boost Tone",
  "NoiseGateActive",
  "ToneStack",
  "Boost",
  "BoostOutput",
  "IRToggle",
  "Boost Model",
  "CalibrateInput",
  "InputCalibrationLevel",
  "OutputMode",
  "Amp Model",
  "Amp Enabled",
};

// DessMetal 0.2 appends a portable conditioned-player mode. The released
// 0.1 identifiers remain the first 20 values in exactly the same order.
inline constexpr std::array<std::string_view, 29> kCurrent02ParameterNames{
  "Input",
  "Gain",
  "Threshold",
  "Bass",
  "Middle",
  "Treble",
  "Output",
  "Boost Level",
  "Boost Tone",
  "NoiseGateActive",
  "ToneStack",
  "Boost",
  "BoostOutput",
  "IRToggle",
  "Boost Model",
  "CalibrateInput",
  "InputCalibrationLevel",
  "OutputMode",
  "Amp Model",
  "Amp Enabled",
  "Player Mode",
  "Player Condition 1",
  "Player Condition 2",
  "Player Condition 3",
  "Player Condition 4",
  "Player Condition 5",
  "Player Condition 6",
  "Player Condition 7",
  "Player Condition 8",
};

struct NamedDefault
{
  std::string_view name;
  double value;
};

inline constexpr std::array<std::string_view, 4> kPedalLevelNames{
  "Oxide Level", "Amber Level", "Verdant Level", "Nocturne Level",
};

// A legacy project had no drive stage or amp-bypass parameter. Keep the newly
// introduced drive out of its signal path, keep the amp active, and initialize
// the remaining host-visible compatibility parameters deterministically.
inline constexpr std::array<NamedDefault, 6> kLegacy010AddedParameterDefaults{
  NamedDefault{"Boost Level", 0.0},
  NamedDefault{"Boost Tone", 5.0},
  NamedDefault{"Boost", 0.0},
  NamedDefault{"BoostOutput", 5.0},
  NamedDefault{"Boost Model", 0.0},
  NamedDefault{"Amp Enabled", 1.0},
};

enum class ParameterLayout01
{
  Invalid,
  Legacy010_14,
  UnsupportedWip18,
  UnsupportedWip19,
  Current20,
};

inline constexpr std::size_t ParameterBytes(const std::size_t parameterCount)
{
  return parameterCount * sizeof(double);
}

// iPlug2's VST3 wrapper appends its own int32 bypass value after the plug-in
// state and expects UnserializeState() to return the position just before it.
inline constexpr std::size_t kVST3BypassSuffixBytes = sizeof(std::int32_t);

inline constexpr bool MatchesParameterPayload(const std::size_t remainingBytes, const std::size_t parameterCount,
                                              const bool allowVST3BypassSuffix)
{
  const auto bytes = ParameterBytes(parameterCount);
  return remainingBytes == bytes
         || (allowVST3BypassSuffix && remainingBytes == bytes + kVST3BypassSuffixBytes);
}

inline constexpr ParameterLayout01 DetectParameterLayout01(const std::size_t remainingBytes,
                                                            const bool allowLegacy010,
                                                            const bool allowVST3BypassSuffix)
{
  if (MatchesParameterPayload(remainingBytes, kCurrent01ParameterNames.size(), allowVST3BypassSuffix))
    return ParameterLayout01::Current20;
  if (allowLegacy010
      && MatchesParameterPayload(remainingBytes, kLegacy010ParameterNames.size(), allowVST3BypassSuffix))
    return ParameterLayout01::Legacy010_14;
  if (allowLegacy010 && MatchesParameterPayload(remainingBytes, 18, allowVST3BypassSuffix))
    return ParameterLayout01::UnsupportedWip18;
  if (allowLegacy010 && MatchesParameterPayload(remainingBytes, 19, allowVST3BypassSuffix))
    return ParameterLayout01::UnsupportedWip19;
  return ParameterLayout01::Invalid;
}

// The original DessMetal enum was DessTortion=0, DessBlock=1. The current
// stable host enum keeps DessTortion-blue at 0 and DessBlock-green at 2.
inline constexpr bool IsSupportedLegacy010AmpModel(const double legacyValue)
{
  return legacyValue == 0.0 || legacyValue == 1.0;
}

inline constexpr double MigrateLegacy010AmpModel(const double legacyValue)
{
  return legacyValue == 1.0 ? 2.0 : legacyValue;
}
} // namespace dessmetal::state
