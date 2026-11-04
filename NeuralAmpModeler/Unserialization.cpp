// Unserialization
//
// This plugin is used in important places, so we need to be considerate when
// attempting to unserialize. If the project was last saved with a legacy
// version, then we need it to "update" to the current version is as
// reasonable a way as possible.
//
// In order to handle older versions, the pattern is:
// 1. Implement unserialization for every version into a version-specific
//    struct (Let's use our friend nlohmann::json. Why not?)
// 2. Implement an "update" from each struct to the next one.
// 3. Implement assigning the data contained in the current struct to the
//    current plugin configuration.
//
// This way, a constant amount of effort is required every time the
// serialization changes instead of having to implement a current
// unserialization for each past version.

// Add new unserialization versions to the top, then add logic to the class method at the bottom.

#include <cmath>
#include <unordered_map>

#include "Unserialization.h"

// Boilerplate

void NeuralAmpModeler::_UnserializeApplyConfig(nlohmann::json& config, bool ampPreset)
{
  // Legacy session/preset states only had a single Boost Level. Keep its
  // value and automation, and clear per-pedal trims from any previous state.
  for (const auto name : dessmetal::state::kPedalLevelNames)
    if (!config.contains(std::string(name)))
      config[std::string(name)] = 0.0;
  TRACE
  ENTER_PARAMS_MUTEX
  for (int index = 0; index < kNumParams; ++index)
  {
    auto* param = GetParam(index);
    if (ampPreset && _IsRigParam(index)) continue;
    const auto value = config.find(param->GetName());
    if (value != config.end())
    {
      param->Set(*value);
    }
  }
  OnParamReset(iplug::EParamSource::kPresetRecall);
  LEAVE_PARAMS_MUTEX

  // Fixed amps still load from bundle resources. DessMetal 0.2 reuses the
  // historical first path slot for a portable player-manifest path; older
  // NAMPath values are deliberately not reinterpreted as manifests.
  _SetNAMPath("");
  const std::string playerManifestPath =
    config.contains("PlayerManifestPath") && config["PlayerManifestPath"].is_string()
      ? static_cast<std::string>(config["PlayerManifestPath"])
      : std::string();
  _SetPlayerManifestPath(playerManifestPath.c_str());
  if (GetParam(kPlayerMode)->Bool() && !playerManifestPath.empty())
    _SetPlayerManifestPathAndRequest(playerManifestPath.c_str(), false);

  const std::string irPath = config.contains("IRPath") && config["IRPath"].is_string()
                               ? static_cast<std::string>(config["IRPath"])
                               : std::string();
  if (irPath.empty())
    _SetIRPath("");
  else
    _SetIRPathAndRequest(irPath.c_str());
}

int _UnserializePaths(const iplug::IByteChunk& chunk, int startPos, nlohmann::json& config)
{
  int pos = startPos;
  WDL_String path;
  pos = GetBoundedChunkString(chunk, path, pos, 4096);
  if (pos < 0)
    throw std::runtime_error("Preset is truncated before the model path");
  config["NAMPath"] = std::string(path.Get());
  pos = GetBoundedChunkString(chunk, path, pos, 4096);
  if (pos < 0)
    throw std::runtime_error("Preset is truncated before the IR path");
  config["IRPath"] = std::string(path.Get());

  return pos;
}

template <typename TNames>
int _UnserializeExpectedKeys(const iplug::IByteChunk& chunk, int startPos, nlohmann::json& config,
                             const TNames& paramNames)
{
  int pos = startPos;
  for (const auto& name : paramNames)
  {
    double v = 0.0;
    pos = chunk.Get(&v, pos);
    if (pos < 0 || !std::isfinite(v))
      throw std::runtime_error("Preset contains truncated or invalid parameter data");
    config[std::string(name)] = v;
  }
  return pos;
}

void _Add02PlayerDefaults(nlohmann::json& config)
{
  // A host can apply an old preset to an already-running 0.2 instance. Make
  // the append-only fields deterministic instead of inheriting Player Mode or
  // condition values from the previously loaded 0.2 state.
  config["Player Mode"] = 0.0;
  for (int index = 1; index <= kNumPlayerConditionParams; ++index)
    config["Player Condition " + std::to_string(index)] = 50.0;
}

// Unserialize NAM Path, IR path, then named keys.
int _UnserializePathsAndExpectedKeys(const iplug::IByteChunk& chunk, int startPos, nlohmann::json& config,
                                     const std::vector<std::string>& paramNames)
{
  const int pos = _UnserializePaths(chunk, startPos, config);
  return _UnserializeExpectedKeys(chunk, pos, config, paramNames);
}

// DessMetal 0.1.0 was built with both a 14-parameter and a later
// 20-parameter positional layout. Distinguish them by the exact bytes remaining
// after the two bounded path strings; never reinterpret a partial layout.
int _GetConfigFrom_0_1_x(const iplug::IByteChunk& chunk, int startPos, nlohmann::json& config,
                         const bool allowLegacy010)
{
#if defined(VST3_API) || defined(VST3C_API) || defined(VST3P_API)
  constexpr bool allowVST3BypassSuffix = true;
#else
  constexpr bool allowVST3BypassSuffix = false;
#endif

  int pos = _UnserializePaths(chunk, startPos, config);
  const auto remainingBytes = static_cast<std::size_t>(chunk.Size() - pos);
  const auto layout =
    dessmetal::state::DetectParameterLayout01(remainingBytes, allowLegacy010, allowVST3BypassSuffix);
  std::size_t parameterCount = 0;

  switch (layout)
  {
    case dessmetal::state::ParameterLayout01::Legacy010_14:
      pos = _UnserializeExpectedKeys(chunk, pos, config, dessmetal::state::kLegacy010ParameterNames);
      if (!dessmetal::state::IsSupportedLegacy010AmpModel(static_cast<double>(config["Amp Model"])))
        throw std::runtime_error(
          "Legacy DessMetal 0.1.0 custom amp state cannot be restored because path-based custom loading is unavailable");
      for (const auto& addedDefault : dessmetal::state::kLegacy010AddedParameterDefaults)
        config[std::string(addedDefault.name)] = addedDefault.value;
      config["Amp Model"] =
        dessmetal::state::MigrateLegacy010AmpModel(static_cast<double>(config["Amp Model"]));
      parameterCount = dessmetal::state::kLegacy010ParameterNames.size();
      break;
    case dessmetal::state::ParameterLayout01::UnsupportedWip18:
      throw std::runtime_error(
        "DessMetal 0.1.0 18-parameter WIP state is recognized but cannot be migrated safely because drive semantics changed");
    case dessmetal::state::ParameterLayout01::UnsupportedWip19:
      throw std::runtime_error(
        "DessMetal 0.1.0 19-parameter WIP state is recognized but cannot be migrated safely because drive semantics changed");
    case dessmetal::state::ParameterLayout01::Current20:
      pos = _UnserializeExpectedKeys(chunk, pos, config, dessmetal::state::kCurrent01ParameterNames);
      parameterCount = dessmetal::state::kCurrent01ParameterNames.size();
      break;
    case dessmetal::state::ParameterLayout01::Invalid:
      break;
  }

  if (parameterCount != 0)
  {
    const auto parameterBytes = dessmetal::state::ParameterBytes(parameterCount);
    if (remainingBytes == parameterBytes + dessmetal::state::kVST3BypassSuffixBytes)
    {
      std::int32_t bypass = 0;
      const int suffixEnd = chunk.Get(&bypass, pos);
      if (suffixEnd < 0 || (bypass != 0 && bypass != 1))
        throw std::runtime_error("Preset contains an invalid VST3 bypass suffix");
    }
    return pos;
  }

  std::ostringstream message;
  message << "Unsupported DessMetal 0.1.x parameter layout: " << remainingBytes << " bytes remain; expected "
          << dessmetal::state::ParameterBytes(dessmetal::state::kCurrent01ParameterNames.size());
  if (allowVST3BypassSuffix)
    message << " plus an optional " << dessmetal::state::kVST3BypassSuffixBytes << "-byte VST3 bypass value";
  if (allowLegacy010)
    message << " or " << dessmetal::state::ParameterBytes(dessmetal::state::kLegacy010ParameterNames.size())
            << "; known 18- and 19-parameter WIP layouts are rejected rather than restored with changed drive semantics";
  throw std::runtime_error(message.str());
}

int _GetConfigFrom_0_2_x(const iplug::IByteChunk& chunk, int startPos, nlohmann::json& config)
{
#if defined(VST3_API) || defined(VST3C_API) || defined(VST3P_API)
  constexpr bool allowVST3BypassSuffix = true;
#else
  constexpr bool allowVST3BypassSuffix = false;
#endif

  int pos = _UnserializePaths(chunk, startPos, config);
  const auto remainingBytes = static_cast<std::size_t>(chunk.Size() - pos);
  const auto fullCount = dessmetal::state::kCurrent02ParameterNames.size() + dessmetal::state::kPedalLevelNames.size();
  const bool hasPedalLevels = dessmetal::state::MatchesParameterPayload(remainingBytes, fullCount, allowVST3BypassSuffix);
  if (!hasPedalLevels && !dessmetal::state::MatchesParameterPayload(
        remainingBytes, dessmetal::state::kCurrent02ParameterNames.size(), allowVST3BypassSuffix))
    throw std::runtime_error("Unsupported DessMetal 0.2 parameter layout");

  pos = _UnserializeExpectedKeys(chunk, pos, config, dessmetal::state::kCurrent02ParameterNames);
  if (hasPedalLevels)
    pos = _UnserializeExpectedKeys(chunk, pos, config, dessmetal::state::kPedalLevelNames);
  config["PlayerManifestPath"] = config["NAMPath"];
  config.erase("NAMPath");

  const auto parameterBytes =
    dessmetal::state::ParameterBytes(hasPedalLevels ? fullCount : dessmetal::state::kCurrent02ParameterNames.size());
  if (remainingBytes == parameterBytes + dessmetal::state::kVST3BypassSuffixBytes)
  {
    std::int32_t bypass = 0;
    const int suffixEnd = chunk.Get(&bypass, pos);
    if (suffixEnd < 0 || (bypass != 0 && bypass != 1))
      throw std::runtime_error("Preset contains an invalid VST3 bypass suffix");
  }
  return pos;
}

void _RenameKeys(nlohmann::json& j, std::unordered_map<std::string, std::string> newNames)
{
  // Assumes no aliasing!
  for (auto it = newNames.begin(); it != newNames.end(); ++it)
  {
    j[it->second] = j[it->first];
    j.erase(it->first);
  }
}

// v0.7.12

void _UpdateConfigFrom_0_7_12(nlohmann::json& config)
{
  // Fill me in once something changes!
}

int _GetConfigFrom_0_7_12(const iplug::IByteChunk& chunk, int startPos, nlohmann::json& config)
{
  std::vector<std::string> paramNames{"Input",
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
                                      "OutputMode"};

  int pos = _UnserializePathsAndExpectedKeys(chunk, startPos, config, paramNames);
  // Then update:
  _UpdateConfigFrom_0_7_12(config);
  return pos;
}

// 0.7.10

void _UpdateConfigFrom_0_7_10(nlohmann::json& config)
{
  // Note: "OutNorm" is Bool-like in v0.7.10, but "OutputMode" is enum.
  // This works because 0 is "Raw" (cf OutNorm false) and 1 is "Calibrated" (cf OutNorm true).
  std::unordered_map<std::string, std::string> newNames{{"OutNorm", "OutputMode"}};
  _RenameKeys(config, newNames);
  // There are new parameters. If they're not included, then 0.7.12 is ok, but future ones might not be.
  config[kCalibrateInputParamName] = (double)kDefaultCalibrateInput;
  config[kInputCalibrationLevelParamName] = kDefaultInputCalibrationLevel;
  _UpdateConfigFrom_0_7_12(config);
}

int _GetConfigFrom_0_7_10(const iplug::IByteChunk& chunk, int startPos, nlohmann::json& config)
{
  std::vector<std::string> paramNames{
    "Input", "Threshold", "Bass", "Middle", "Treble", "Output", "NoiseGateActive", "ToneStack", "OutNorm", "IRToggle"};
  int pos = _UnserializePathsAndExpectedKeys(chunk, startPos, config, paramNames);
  // Then update:
  _UpdateConfigFrom_0_7_10(config);
  return pos;
}

// Earlier than 0.7.10 (Assumed to be 0.7.3-0.7.9)

void _UpdateConfigFrom_Earlier(nlohmann::json& config)
{
  std::unordered_map<std::string, std::string> newNames{{"Gate", "Threshold"}};
  _RenameKeys(config, newNames);
  _UpdateConfigFrom_0_7_10(config);
}

int _GetConfigFrom_Earlier(const iplug::IByteChunk& chunk, int startPos, nlohmann::json& config)
{
  std::vector<std::string> paramNames{
    "Input", "Gate", "Bass", "Middle", "Treble", "Output", "NoiseGateActive", "ToneStack", "OutNorm", "IRToggle"};

  int pos = _UnserializePathsAndExpectedKeys(chunk, startPos, config, paramNames);
  // Then update:
  _UpdateConfigFrom_Earlier(config);
  return pos;
}

//==============================================================================

class _Version
{
public:
  _Version(const int major, const int minor, const int patch)
  : mMajor(major)
  , mMinor(minor)
  , mPatch(patch) {};
  _Version(const std::string& versionStr)
  {
    std::istringstream stream(versionStr);
    std::string token;
    std::vector<int> parts;

    // Split the string by "."
    while (std::getline(stream, token, '.'))
    {
      parts.push_back(std::stoi(token)); // Convert to int and store
    }

    // Check if we have exactly 3 parts
    if (parts.size() != 3)
    {
      throw std::invalid_argument("Input string does not contain exactly 3 segments separated by '.'");
    }

    // Assign the parts to the provided int variables
    mMajor = parts[0];
    mMinor = parts[1];
    mPatch = parts[2];
  };

  bool operator>=(const _Version& other) const
  {
    // Compare on major version:
    if (GetMajor() > other.GetMajor())
    {
      return true;
    }
    if (GetMajor() < other.GetMajor())
    {
      return false;
    }
    // Compare on minor
    if (GetMinor() > other.GetMinor())
    {
      return true;
    }
    if (GetMinor() < other.GetMinor())
    {
      return false;
    }
    // Compare on patch
    return GetPatch() >= other.GetPatch();
  };

  int GetMajor() const { return mMajor; };
  int GetMinor() const { return mMinor; };
  int GetPatch() const { return mPatch; };

private:
  int mMajor;
  int mMinor;
  int mPatch;
};

int NeuralAmpModeler::_UnserializeStateWithKnownVersion(const iplug::IByteChunk& chunk, int startPos, bool ampPreset)
{
  // We already got through the header before calling this.
  int pos = startPos;

  // Get the version
  WDL_String wVersion;
  pos = GetBoundedChunkString(chunk, wVersion, pos, 64);
  if (pos < 0)
    throw std::runtime_error("Preset contains an invalid version string");
  std::string versionStr(wVersion.Get());
  _Version version(versionStr);
  // Act accordingly
  nlohmann::json config;
  if (version.GetMajor() == 0 && version.GetMinor() == 2)
  {
    pos = _GetConfigFrom_0_2_x(chunk, pos, config);
  }
  else if (version.GetMajor() == 0 && version.GetMinor() == 1)
  {
    // Only 0.1.0 emitted the historical 14-value layout. 0.1.1 establishes
    // the current 20-value layout while still accepting existing 20-value
    // states that were tagged 0.1.0.
    pos = _GetConfigFrom_0_1_x(chunk, pos, config, version.GetPatch() == 0);
  }
  else if (version >= _Version(0, 7, 12))
  {
    pos = _GetConfigFrom_0_7_12(chunk, pos, config);
  }
  else if (version >= _Version(0, 7, 10))
  {
    pos = _GetConfigFrom_0_7_10(chunk, pos, config);
  }
  else if (version >= _Version(0, 7, 9))
  {
    pos = _GetConfigFrom_Earlier(chunk, pos, config);
  }
  else
  {
    throw std::runtime_error("Unsupported preset version: " + versionStr);
  }
  if (!(version.GetMajor() == 0 && version.GetMinor() == 2))
    _Add02PlayerDefaults(config);
  _UnserializeApplyConfig(config, ampPreset);
  return pos;
}

int NeuralAmpModeler::_UnserializeStateWithUnknownVersion(const iplug::IByteChunk& chunk, int startPos, bool ampPreset)
{
  nlohmann::json config;
  int pos = _GetConfigFrom_Earlier(chunk, startPos, config);
  _Add02PlayerDefaults(config);
  _UnserializeApplyConfig(config, ampPreset);
  return pos;
}
