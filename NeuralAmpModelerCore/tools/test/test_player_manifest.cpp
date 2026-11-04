#include <array>
#include <cassert>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <memory>
#include <random>
#include <string>
#include <vector>

#include "NAM/get_dsp.h"
#include "NAM/player_manifest.h"
#include "allocation_tracking.h"
#include "json.hpp"

namespace test_player_manifest
{
namespace
{
class TemporaryDirectory
{
public:
  TemporaryDirectory()
  {
    std::random_device random;
    mPath = std::filesystem::temp_directory_path()
            / ("dessmetal-player-manifest-test-" + std::to_string(random()) + "-" + std::to_string(random()));
    std::filesystem::create_directories(mPath);
  }

  ~TemporaryDirectory()
  {
    std::error_code error;
    std::filesystem::remove_all(mPath, error);
  }

  const std::filesystem::path& Path() const { return mPath; }

private:
  std::filesystem::path mPath;
};

nlohmann::json MakeLstm()
{
  std::vector<float> weights(16U, 0.0f);
  weights.back() = 0.125f;
  return {
    {"version", "0.7.0"},
    {"architecture", "LSTM"},
    {"sample_rate", 48000.0},
    {"metadata", nullptr},
    {"config", {{"input_size", 1}, {"hidden_size", 1}, {"num_layers", 1}}},
    {"weights", weights},
  };
}

class GainDSP final : public nam::DSP
{
public:
  explicit GainDSP(const double gain)
  : nam::DSP(1, 1, 48000.0)
  , mGain(gain)
  {
  }

  void process(NAM_SAMPLE** input, NAM_SAMPLE** output, const int num_frames) override
  {
    for (int frame = 0; frame < num_frames; ++frame)
      output[0][frame] = static_cast<NAM_SAMPLE>(mGain * input[0][frame]);
  }

private:
  double mGain;
};

void AppendLittleEndian32(std::vector<std::uint8_t>& bytes, const std::uint32_t value)
{
  for (std::uint32_t index = 0U; index < 4U; ++index)
    bytes.push_back(static_cast<std::uint8_t>(value >> (index * 8U)));
}

void AppendLittleEndian64(std::vector<std::uint8_t>& bytes, const std::uint64_t value)
{
  for (std::uint32_t index = 0U; index < 8U; ++index)
    bytes.push_back(static_cast<std::uint8_t>(value >> (index * 8U)));
}

void AppendDigest(std::vector<std::uint8_t>& bytes, const std::array<std::uint8_t, 32>& digest)
{
  bytes.insert(bytes.end(), digest.begin(), digest.end());
}

std::vector<std::uint8_t> MakePackage()
{
  const std::string model = MakeLstm().dump();
  const auto model_digest =
    nam::player::detail::Sha256Bytes(reinterpret_cast<const std::uint8_t*>(model.data()), model.size());
  const nlohmann::json manifest = {
    {"schema_version", 1},
    {"artifact_id", "native-package-test"},
    {"name", "Native package test"},
    {"model",
     {{"path", "model.nam"},
      {"format", "nam"},
      {"sha256", nam::player::detail::DigestHex(model_digest)},
      {"format_version", "0.7.0"},
      {"architecture", "LSTM"},
      {"sample_rate", 48000.0},
      {"parameter_mode", "snapshot"},
      {"global_condition_size", 0},
      {"runtime_quality", {{"mode", "fixed"}, {"default_size", 1.0}, {"breakpoints", nlohmann::json::array()}}}}},
    {"controls", nlohmann::json::array()},
  };
  const std::string manifest_text = manifest.dump();
  const auto manifest_digest =
    nam::player::detail::Sha256Bytes(reinterpret_cast<const std::uint8_t*>(manifest_text.data()), manifest_text.size());

  std::vector<std::uint8_t> package{'D', 'E', 'S', 'S', 'A', 'M', 'P', 0U};
  AppendLittleEndian32(package, 1U);
  AppendLittleEndian32(package, 0U);
  AppendLittleEndian64(package, static_cast<std::uint64_t>(manifest_text.size()));
  AppendLittleEndian64(package, static_cast<std::uint64_t>(model.size()));
  AppendDigest(package, manifest_digest);
  AppendDigest(package, model_digest);
  package.insert(package.end(), manifest_text.begin(), manifest_text.end());
  package.insert(package.end(), model.begin(), model.end());
  return package;
}

std::vector<std::uint8_t> MakeConditionedPackage(const bool include_display_minimum, const bool include_display_maximum,
                                                 const std::string& taper = "linear",
                                                 const double default_value = 0.643164, const bool include_taper = true,
                                                 nlohmann::json fixed_controls = nlohmann::json::array(),
                                                 const std::string& head_shell_style = "",
                                                 const std::string& product_family = "",
                                                 const std::string& product_variant = "")
{
  const std::string model = MakeLstm().dump();
  const auto model_digest =
    nam::player::detail::Sha256Bytes(reinterpret_cast<const std::uint8_t*>(model.data()), model.size());
  nlohmann::json control = {
    {"id", "gain"},
    {"label", "Gain"},
    {"unit", "%"},
    {"minimum", 0.35},
    {"maximum", 0.80},
    {"default_value", default_value},
    {"model_parameter_index", 0},
  };
  if (include_taper)
    control["taper"] = taper;
  if (include_display_minimum)
    control["display_minimum"] = 35.0;
  if (include_display_maximum)
    control["display_maximum"] = 80.0;

  nlohmann::json manifest = {
    {"schema_version", 1},
    {"artifact_id", "condition-display-test"},
    {"name", "Condition display test"},
    {"model",
     {{"path", "model.nam"},
      {"format", "nam"},
      {"sha256", nam::player::detail::DigestHex(model_digest)},
      {"format_version", "0.7.0"},
      {"architecture", "LSTM"},
      {"sample_rate", 48000.0},
      {"parameter_mode", "global_condition"},
      {"global_condition_size", 1},
      {"runtime_quality", {{"mode", "fixed"}, {"default_size", 1.0}, {"breakpoints", nlohmann::json::array()}}}}},
    {"controls", nlohmann::json::array({control})},
    {"fixed_controls", std::move(fixed_controls)},
  };
  if (!head_shell_style.empty() || !product_family.empty() || !product_variant.empty())
  {
    manifest["presentation"] = nlohmann::json::object();
    if (!head_shell_style.empty())
      manifest["presentation"]["head_shell"] = head_shell_style;
    if (!product_family.empty())
      manifest["presentation"]["product_family"] = product_family;
    if (!product_variant.empty())
      manifest["presentation"]["product_variant"] = product_variant;
  }
  const std::string manifest_text = manifest.dump();
  const auto manifest_digest =
    nam::player::detail::Sha256Bytes(reinterpret_cast<const std::uint8_t*>(manifest_text.data()), manifest_text.size());

  std::vector<std::uint8_t> package{'D', 'E', 'S', 'S', 'A', 'M', 'P', 0U};
  AppendLittleEndian32(package, 1U);
  AppendLittleEndian32(package, 0U);
  AppendLittleEndian64(package, static_cast<std::uint64_t>(manifest_text.size()));
  AppendLittleEndian64(package, static_cast<std::uint64_t>(model.size()));
  AppendDigest(package, manifest_digest);
  AppendDigest(package, model_digest);
  package.insert(package.end(), manifest_text.begin(), manifest_text.end());
  package.insert(package.end(), model.begin(), model.end());
  return package;
}

std::vector<std::uint8_t> MakeMorphPackage(const bool include_learned_morph = false)
{
  const std::string primary_model = MakeLstm().dump();
  nlohmann::json secondary_json = MakeLstm();
  secondary_json["weights"][15] = 0.25f;
  const std::string secondary_model = secondary_json.dump();
  const auto primary_digest =
    nam::player::detail::Sha256Bytes(reinterpret_cast<const std::uint8_t*>(primary_model.data()), primary_model.size());
  const auto secondary_digest = nam::player::detail::Sha256Bytes(
    reinterpret_cast<const std::uint8_t*>(secondary_model.data()), secondary_model.size());
  const nlohmann::json runtime_quality = {
    {"mode", "fixed"}, {"default_size", 1.0}, {"breakpoints", nlohmann::json::array()}};
  nlohmann::json manifest = {
    {"schema_version", 1},
    {"artifact_id", "native-morph-package-test"},
    {"name", "DESS FORGE test"},
    {"model",
     {{"path", "model.nam"},
      {"format", "nam"},
      {"sha256", nam::player::detail::DigestHex(primary_digest)},
      {"format_version", "0.7.0"},
      {"architecture", "LSTM"},
      {"sample_rate", 48000.0},
      {"parameter_mode", "snapshot"},
      {"global_condition_size", 0},
      {"runtime_quality", runtime_quality}}},
    {"controls", nlohmann::json::array()},
    {"morph",
     {{"mode", "dual_model_linear_v1"},
      {"from_label", "GER"},
      {"to_label", "BRIT"},
      {"model",
       {{"path", "morph-model.nam"},
        {"format", "nam"},
        {"sha256", nam::player::detail::DigestHex(secondary_digest)},
        {"sample_rate", 48000.0},
        {"parameter_mode", "snapshot"},
        {"global_condition_size", 0},
        {"runtime_quality", runtime_quality}}},
      {"alignment",
       {{"delay_samples", 1},
        {"polarity", -1},
        {"level_scale", 0.6074090625600933},
        {"reference_sample_rate", 48000.0}}},
      {"blend", {{"law", "linear"}, {"center_gain", 1.1137418941545114}, {"smoothing_ms", 30.0}}},
      {"control",
       {{"id", "morph"},
        {"label", "Morph"},
        {"unit", "%"},
        {"taper", "linear"},
        {"minimum", 0.0},
        {"maximum", 1.0},
        {"default_value", 0.5},
        {"display_minimum", 0.0},
        {"display_maximum", 100.0}}}}},
  };
  if (include_learned_morph)
  {
    manifest["model"]["parameter_mode"] = "global_condition";
    manifest["model"]["global_condition_size"] = 1;
    manifest["controls"] = nlohmann::json::array({{
      {"id", "morph"},
      {"label", "Learned Morph"},
      {"unit", "%"},
      {"taper", "linear"},
      {"minimum", 0.0},
      {"maximum", 1.0},
      {"default_value", 0.5},
      {"model_parameter_index", 0},
    }});
  }
  const std::string manifest_text = manifest.dump();
  const auto manifest_digest =
    nam::player::detail::Sha256Bytes(reinterpret_cast<const std::uint8_t*>(manifest_text.data()), manifest_text.size());

  std::vector<std::uint8_t> package{'D', 'E', 'S', 'S', 'A', 'M', 'P', 0U};
  AppendLittleEndian32(package, 2U);
  AppendLittleEndian32(package, 0U);
  AppendLittleEndian64(package, static_cast<std::uint64_t>(manifest_text.size()));
  AppendLittleEndian64(package, static_cast<std::uint64_t>(primary_model.size()));
  AppendLittleEndian64(package, static_cast<std::uint64_t>(secondary_model.size()));
  AppendDigest(package, manifest_digest);
  AppendDigest(package, primary_digest);
  AppendDigest(package, secondary_digest);
  package.insert(package.end(), manifest_text.begin(), manifest_text.end());
  package.insert(package.end(), primary_model.begin(), primary_model.end());
  package.insert(package.end(), secondary_model.begin(), secondary_model.end());
  return package;
}

void WriteBytes(const std::filesystem::path& path, const std::vector<std::uint8_t>& bytes)
{
  std::ofstream output(path, std::ios::binary | std::ios::trunc);
  output.write(reinterpret_cast<const char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
  output.close();
  assert(output.good());
}

bool LoadThrows(const std::filesystem::path& path)
{
  try
  {
    static_cast<void>(nam::player::LoadManifest(path));
    return false;
  }
  catch (const std::exception&)
  {
    return true;
  }
}
} // namespace

void test_dess_package_is_self_contained_and_loadable()
{
  TemporaryDirectory temporary;
  const auto package_path = temporary.Path() / "complete-amp.dess";
  WriteBytes(package_path, MakePackage());

  const auto manifest = nam::player::LoadManifest(package_path);
  assert(manifest.manifest_path == std::filesystem::weakly_canonical(package_path));
  assert(manifest.name == "Native package test");
  assert(manifest.global_condition_size == 0);
  assert(std::filesystem::is_regular_file(manifest.model_path));
  assert(nam::player::detail::Sha256File(manifest.model_path) == manifest.model_sha256);

  std::unique_ptr<nam::DSP> model = nam::player::LoadPlayerDSP(manifest);
  assert(model != nullptr);
  nam::player::ValidateAndApplyRuntime(*model, manifest);
}

void test_custom_import_keeps_verified_packages_after_source_moves()
{
  TemporaryDirectory temporary;
  const auto source = temporary.Path() / "my-amp.dess";
  const auto library = temporary.Path() / "library";
  const auto first = MakePackage();
  WriteBytes(source, first);
  const auto imported = nam::player::ImportDessPackage(source, library);
  const auto stored = imported.manifest.manifest_path;
  assert(imported.model && stored.parent_path() == library);
  assert(stored.stem().string() == nam::player::detail::Sha256File(source));
  assert(nam::player::ImportDessPackage(source, library).manifest.manifest_path == stored);

  // Different valid packages can have the same filename.
  WriteBytes(source, MakeMorphPackage());
  const auto second = nam::player::ImportDessPackage(source, library);
  assert(second.manifest.manifest_path != stored);
  auto corrupt = first;
  corrupt.back() ^= 1U;
  WriteBytes(source, corrupt);
  bool rejected = false;
  try { static_cast<void>(nam::player::ImportDessPackage(source, library)); }
  catch (const std::exception&) { rejected = true; }
  assert(rejected);
  // This fixture's manifest is well-formed but its model has the wrong control count.
  WriteBytes(source, MakeConditionedPackage(true, true));
  rejected = false;
  try { static_cast<void>(nam::player::ImportDessPackage(source, library)); }
  catch (const std::exception&) { rejected = true; }
  assert(rejected);
  std::filesystem::remove(source);
  assert(std::distance(std::filesystem::directory_iterator(library), std::filesystem::directory_iterator()) == 2);
  assert(nam::player::LoadPlayerDSP(nam::player::LoadManifest(stored)) != nullptr);
  assert(nam::player::LoadPlayerDSP(nam::player::LoadManifest(second.manifest.manifest_path)) != nullptr);
}

void test_player_dsp_revalidates_the_bytes_it_parses()
{
  TemporaryDirectory temporary;
  const auto package_path = temporary.Path() / "verified-read.dess";
  WriteBytes(package_path, MakePackage());

  const auto manifest = nam::player::LoadManifest(package_path);
  auto replacement_json = MakeLstm();
  replacement_json["weights"][0] = 0.5f;
  const std::string replacement = replacement_json.dump();
  WriteBytes(manifest.model_path, std::vector<std::uint8_t>(replacement.begin(), replacement.end()));

  bool threw = false;
  try
  {
    static_cast<void>(nam::player::LoadPlayerDSP(manifest));
  }
  catch (const std::exception&)
  {
    threw = true;
  }
  assert(threw);

  // Loading the original package repairs its content-addressed cache and
  // proves the same path remains usable after the failed verification.
  const auto repaired_manifest = nam::player::LoadManifest(package_path);
  assert(nam::player::LoadPlayerDSP(repaired_manifest) != nullptr);
}

void test_package_manifest_parser_binds_the_verified_cache_bytes()
{
  TemporaryDirectory temporary;
  const auto package_path = temporary.Path() / "manifest-verified-read.dess";
  WriteBytes(package_path, MakePackage());

  const auto manifest = nam::player::LoadManifest(package_path);
  const auto cached_manifest_path = manifest.model_path.parent_path() / "player-manifest.json";
  const std::string expected_sha256 = nam::player::detail::Sha256File(cached_manifest_path);
  const std::vector<std::uint8_t> replacement{'{', '}', '\n'};
  WriteBytes(cached_manifest_path, replacement);

  bool threw = false;
  try
  {
    static_cast<void>(nam::player::LoadManifest(
      cached_manifest_path, nam::player::kMaxDAWConditionControls, expected_sha256));
  }
  catch (const std::exception&)
  {
    threw = true;
  }
  assert(threw);

  // The package loader repairs and re-verifies the cache from its own
  // authenticated bytes.
  assert(nam::player::LoadPlayerDSP(nam::player::LoadManifest(package_path)) != nullptr);
}

void test_dess_package_rejects_corruption_and_trailing_data()
{
  TemporaryDirectory temporary;
  const auto original = MakePackage();

  auto manifest_corruption = original;
  manifest_corruption[96U] ^= 0x01U;
  const auto manifest_path = temporary.Path() / "manifest-corrupt.dess";
  WriteBytes(manifest_path, manifest_corruption);
  assert(LoadThrows(manifest_path));

  auto model_corruption = original;
  model_corruption.back() ^= 0x01U;
  const auto model_path = temporary.Path() / "model-corrupt.dess";
  WriteBytes(model_path, model_corruption);
  assert(LoadThrows(model_path));

  auto trailing_data = original;
  trailing_data.push_back(0U);
  const auto trailing_path = temporary.Path() / "trailing-data.dess";
  WriteBytes(trailing_path, trailing_data);
  assert(LoadThrows(trailing_path));
}

void test_morph_package_loads_both_models_and_renders()
{
  TemporaryDirectory temporary;
  const auto package_path = temporary.Path() / "forge-morph.dess";
  WriteBytes(package_path, MakeMorphPackage());

  const auto manifest = nam::player::LoadManifest(package_path);
  assert(manifest.morph.enabled);
  assert(manifest.morph.from_label == "GER");
  assert(manifest.morph.to_label == "BRIT");
  assert(manifest.global_condition_size == 0);
  assert(manifest.PlayerParameterCount() == 1);
  assert(manifest.controls[0].id == "morph");
  assert(manifest.controls[0].model_parameter_index == 0);
  assert(std::filesystem::is_regular_file(manifest.model_path));
  assert(std::filesystem::is_regular_file(manifest.morph.model_path));

  auto model = nam::player::LoadPlayerDSP(manifest);
  assert(model != nullptr);
  assert(model->NumGlobalParams() == 1);
  const auto parameters = manifest.DefaultModelParameters();
  model->SetDefaultParams(parameters.data(), static_cast<int>(parameters.size()));
  model->Reset(48000.0, 8);
  std::array<NAM_SAMPLE, 8> input_samples{1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
  std::array<NAM_SAMPLE, 8> output_samples{};
  NAM_SAMPLE* input = input_samples.data();
  NAM_SAMPLE* output = output_samples.data();
  model->process(&input, &output, 8, parameters.data(), static_cast<int>(parameters.size()));
  assert(std::all_of(
    output_samples.begin(), output_samples.end(), [](const NAM_SAMPLE sample) { return std::isfinite(sample); }));

  auto corrupted = MakeMorphPackage();
  corrupted.back() ^= 0x01U;
  const auto corrupted_path = temporary.Path() / "forge-morph-corrupt.dess";
  WriteBytes(corrupted_path, corrupted);
  assert(LoadThrows(corrupted_path));

  const auto duplicate_semantics_path = temporary.Path() / "learned-and-dual-morph.dess";
  WriteBytes(duplicate_semantics_path, MakeMorphPackage(true));
  assert(LoadThrows(duplicate_semantics_path));
}

void test_morph_blend_law_endpoints_center_and_realtime_safety()
{
  nam::player::MorphSettings settings;
  settings.level_scale = 0.5;
  settings.center_gain = 1.2;
  settings.smoothing_ms = 0.0;
  auto model = std::make_unique<nam::player::MorphingDSP>(
    std::make_unique<GainDSP>(1.0), std::make_unique<GainDSP>(3.0), 48000.0, settings);
  std::array<NAM_SAMPLE, 4> input_samples{1.0, -0.5, 0.25, -0.125};
  std::array<NAM_SAMPLE, 4> output_samples{};
  NAM_SAMPLE* input = input_samples.data();
  NAM_SAMPLE* output = output_samples.data();
  std::array<double, 1> parameters{0.0};
  model->SetDefaultParams(parameters.data(), 1);
  model->Reset(48000.0, 4);

  model->process(&input, &output, 4, parameters.data(), 1);
  assert(std::abs(output_samples[0] - 1.0) < 1.0e-12);
  parameters[0] = 1.0;
  model->process(&input, &output, 4, parameters.data(), 1);
  assert(std::abs(output_samples[0] - 1.5) < 1.0e-12);
  parameters[0] = 0.5;
  model->process(&input, &output, 4, parameters.data(), 1);
  assert(std::abs(output_samples[0] - 1.5) < 1.0e-12);

  allocation_tracking::run_allocation_test_no_allocations(
    nullptr, [&]() { model->process(&input, &output, 4, parameters.data(), 1); }, nullptr,
    "MorphingDSP process must not allocate");
}

void test_condition_control_display_range_preserves_model_mapping()
{
  nam::player::Control control;
  control.unit = "%";
  control.minimum = 0.35;
  control.maximum = 0.80;
  control.default_value = 0.643164;

  const double normalized_default = (control.default_value - control.minimum) / (control.maximum - control.minimum);
  assert(std::abs(control.NormalizedDefault() - normalized_default) < 1.0e-12);
  assert(std::abs(control.DisplayMinimum() - 35.0) < 1.0e-12);
  assert(std::abs(control.DisplayMaximum() - 80.0) < 1.0e-12);
  assert(std::abs(control.DisplayValueFromNormalized(control.NormalizedDefault()) - 64.3164) < 1.0e-9);

  control.display_minimum = 1.0;
  control.display_maximum = 11.0;
  assert(std::abs(control.DisplayMinimum() - 1.0) < 1.0e-12);
  assert(std::abs(control.DisplayMaximum() - 11.0) < 1.0e-12);
  assert(std::abs(control.DisplayValueFromNormalized(0.25) - 3.5) < 1.0e-12);
}

void test_condition_control_display_range_parser()
{
  TemporaryDirectory temporary;

  const auto explicit_path = temporary.Path() / "explicit-display.dess";
  WriteBytes(explicit_path, MakeConditionedPackage(true, true));
  const auto explicit_manifest = nam::player::LoadManifest(explicit_path);
  assert(explicit_manifest.controls.size() == 1U);
  assert(explicit_manifest.controls[0].HasDisplayRange());
  assert(std::abs(explicit_manifest.controls[0].DisplayMinimum() - 35.0) < 1.0e-12);
  assert(std::abs(explicit_manifest.controls[0].DisplayMaximum() - 80.0) < 1.0e-12);

  const auto fallback_path = temporary.Path() / "fallback-display.dess";
  WriteBytes(fallback_path, MakeConditionedPackage(false, false));
  const auto fallback_manifest = nam::player::LoadManifest(fallback_path);
  assert(!fallback_manifest.controls[0].HasDisplayRange());
  assert(std::abs(fallback_manifest.controls[0].DisplayMinimum() - 35.0) < 1.0e-12);
  assert(std::abs(fallback_manifest.controls[0].DisplayMaximum() - 80.0) < 1.0e-12);

  const auto half_range_path = temporary.Path() / "half-display.dess";
  WriteBytes(half_range_path, MakeConditionedPackage(true, false));
  assert(LoadThrows(half_range_path));
}

void test_condition_control_taper_parser()
{
  TemporaryDirectory temporary;

  const auto switch_path = temporary.Path() / "switch-control.dess";
  WriteBytes(switch_path, MakeConditionedPackage(false, false, "switch", 0.35));
  const auto switch_manifest = nam::player::LoadManifest(switch_path);
  assert(switch_manifest.controls.size() == 1U);
  assert(switch_manifest.controls[0].taper == "switch");
  assert(switch_manifest.controls[0].IsSwitch());

  const auto legacy_path = temporary.Path() / "legacy-linear-control.dess";
  auto legacy_package = MakeConditionedPackage(false, false, "linear", 0.643164, false);
  WriteBytes(legacy_path, legacy_package);
  assert(nam::player::LoadManifest(legacy_path).controls[0].taper == "linear");
  assert(!nam::player::LoadManifest(legacy_path).controls[0].IsSwitch());

  const auto invalid_taper_path = temporary.Path() / "invalid-taper.dess";
  WriteBytes(invalid_taper_path, MakeConditionedPackage(false, false, "momentary", 0.35));
  assert(LoadThrows(invalid_taper_path));

  const auto invalid_default_path = temporary.Path() / "invalid-switch-default.dess";
  WriteBytes(invalid_default_path, MakeConditionedPackage(false, false, "switch", 0.50));
  assert(LoadThrows(invalid_default_path));
}

void test_fixed_switch_control_parser()
{
  TemporaryDirectory temporary;
  const nlohmann::json fixed = nlohmann::json::array({
    {{"id", "bright"}, {"label", "Bright"}, {"taper", "switch"}, {"value", 1.0}},
    {{"id", "bottom"}, {"label", "Bottom"}, {"taper", "switch"}, {"value", 0.0}},
    {{"id", "lead_volume"},
     {"label", "Lead Volume"},
     {"taper", "linear"},
     {"value", 3.5},
     {"minimum", 0.0},
     {"maximum", 10.0},
     {"unit", ""}},
  });
  const auto fixed_path = temporary.Path() / "fixed-switches.dess";
  WriteBytes(fixed_path, MakeConditionedPackage(false, false, "linear", 0.643164, true, fixed));
  const auto manifest = nam::player::LoadManifest(fixed_path);
  assert(manifest.fixed_controls.size() == 3U);
  assert(manifest.fixed_controls[0].id == "bright");
  assert(manifest.fixed_controls[0].IsOn());
  assert(!manifest.fixed_controls[1].IsOn());
  assert(manifest.fixed_controls[2].IsLinear());
  assert(manifest.fixed_controls[2].value == 3.5);
  assert(manifest.fixed_controls[2].minimum == 0.0);
  assert(manifest.fixed_controls[2].maximum == 10.0);

  const auto invalid_value_path = temporary.Path() / "invalid-fixed-value.dess";
  const nlohmann::json invalid_value =
    nlohmann::json::array({{{"id", "bright"}, {"label", "Bright"}, {"taper", "switch"}, {"value", 0.5}}});
  WriteBytes(invalid_value_path, MakeConditionedPackage(false, false, "linear", 0.643164, true, invalid_value));
  assert(LoadThrows(invalid_value_path));

  const auto invalid_linear_path = temporary.Path() / "invalid-fixed-linear.dess";
  const nlohmann::json invalid_linear = nlohmann::json::array({{{"id", "lead_volume"},
                                                                {"label", "Lead Volume"},
                                                                {"taper", "linear"},
                                                                {"value", 11.0},
                                                                {"minimum", 0.0},
                                                                {"maximum", 10.0}}});
  WriteBytes(invalid_linear_path, MakeConditionedPackage(false, false, "linear", 0.643164, true, invalid_linear));
  assert(LoadThrows(invalid_linear_path));

  const auto duplicate_path = temporary.Path() / "duplicate-fixed-id.dess";
  const nlohmann::json duplicate =
    nlohmann::json::array({{{"id", "gain"}, {"label", "Gain"}, {"taper", "switch"}, {"value", 1.0}}});
  WriteBytes(duplicate_path, MakeConditionedPackage(false, false, "linear", 0.643164, true, duplicate));
  assert(LoadThrows(duplicate_path));
}

void test_presentation_head_shell_parser()
{
  TemporaryDirectory temporary;

  const auto default_path = temporary.Path() / "default-presentation.dess";
  WriteBytes(default_path, MakeConditionedPackage(false, false));
  assert(nam::player::LoadManifest(default_path).head_shell_style == "studio-black-v1");

  const auto dual_red_path = temporary.Path() / "dual-red-presentation.dess";
  WriteBytes(dual_red_path, MakeConditionedPackage(
                              false, false, "linear", 0.643164, true, nlohmann::json::array(),
                              "dual-red-modern-v1", "redline", "us"));
  const auto dual_red_manifest = nam::player::LoadManifest(dual_red_path);
  assert(dual_red_manifest.head_shell_style == "dual-red-modern-v1");
  assert(dual_red_manifest.product_family == "redline");
  assert(dual_red_manifest.product_variant == "us");

  const auto invalid_path = temporary.Path() / "invalid-presentation.dess";
  WriteBytes(invalid_path, MakeConditionedPackage(
                             false, false, "linear", 0.643164, true, nlohmann::json::array(), "../untrusted-file"));
  assert(LoadThrows(invalid_path));

  const auto invalid_family_path = temporary.Path() / "invalid-product-family.dess";
  WriteBytes(invalid_family_path, MakeConditionedPackage(
                                    false, false, "linear", 0.643164, true, nlohmann::json::array(),
                                    "dual-red-modern-v1", "../redline", "us"));
  assert(LoadThrows(invalid_family_path));

  const auto orphan_variant_path = temporary.Path() / "orphan-product-variant.dess";
  WriteBytes(orphan_variant_path, MakeConditionedPackage(
                                     false, false, "linear", 0.643164, true, nlohmann::json::array(),
                                     "dual-red-modern-v1", "", "brit"));
  assert(LoadThrows(orphan_variant_path));
}
} // namespace test_player_manifest
