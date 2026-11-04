#include <algorithm>
#include <cassert>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "json.hpp"

#include "NAM/get_dsp.h"
#include "NAM/model_validation.h"

namespace test_get_dsp
{
// Config
const std::string basicConfigStr =
  R"({"version": "0.5.4", "metadata": {"date": {"year": 2024, "month": 10, "day": 9, "hour": 18, "minute": 44, "second": 41}, "loudness": -37.8406867980957, "gain": 0.13508800804658277, "name": "Test LSTM", "modeled_by": "Steve", "gear_type": "amp", "gear_make": "Darkglass Electronics", "gear_model": "Microtubes 900 v2", "tone_type": "clean", "input_level_dbu": 18.3, "output_level_dbu": 12.3, "training": {"settings": {"ignore_checks": false}, "data": {"latency": {"manual": null, "calibration": {"algorithm_version": 1, "delays": [-16], "safety_factor": 1, "recommended": -17, "warnings": {"matches_lookahead": false, "disagreement_too_high": false}}}, "checks": {"version": 3, "passed": true}}, "validation_esr": null}}, "architecture": "LSTM", "config": {"input_size": 1, "hidden_size": 3, "num_layers": 1}, "weights": [-0.21677088737487793, -0.6683622002601624, -0.2560940980911255, -0.3588429093360901, 0.17952610552310944, 0.19445613026618958, -0.01662646047770977, 0.5353694558143616, -0.2536540627479553, -0.5132213234901428, -0.020476307719945908, 0.08592455089092255, -0.6891753673553467, 0.3627359867095947, 0.008421811275184155, 0.3113192617893219, 0.14251480996608734, 0.07989779114723206, -0.18211324512958527, 0.7118963003158569, 0.41084015369415283, -0.6571938395500183, -0.13214066624641418, -0.2698603868484497, 0.49387243390083313, -0.3491725027561188, 0.6353667974472046, -0.5005152225494385, 0.2052856683731079, -0.4301638901233673, -0.15770092606544495, -0.7181791067123413, 0.056290093809366226, -0.49049463868141174, 0.6623441576957703, 0.09029324352741241, 0.34005245566368103, 0.16416560113430023, 0.15520110726356506, -0.4155678153038025, -0.36928507685661316, 0.3211132884025574, -0.6769840121269226, -0.1575538069009781, 0.05268515646457672, -0.4191459119319916, 0.599330484867096, 0.21518059074878693, -4.246325492858887, -3.315647840499878, -4.328850746154785, 4.496089458465576, 5.015639305114746, 3.6492037773132324, 0.14431169629096985, -0.6633821725845337, 0.11673200130462646, -0.1418764889240265, -0.4897872805595398, -0.8689419031143188, -0.06714004278182983, -0.4450395107269287, -0.02142983116209507, -0.15136894583702087, -2.775207996368408, -0.08681213855743408, 0.05702732503414154, 0.670292317867279, 0.31442636251449585, 0.30793967843055725], "sample_rate": 48000})";

// Copied over but shouldn't be publicly-exposed.
std::vector<float> GetWeights(nlohmann::json const& j)
{
  auto it = j.find("weights");
  if (it != j.end())
  {
    return *it;
  }
  else
  {
    throw std::runtime_error("Corrupted model file is missing weights.");
  }
}

nam::dspData _GetConfig(const std::string& configStr = basicConfigStr)
{
  nlohmann::json j = nlohmann::json::parse(configStr);

  std::vector<float> weights = GetWeights(j);
  nam::dspData returnedConfig;
  returnedConfig.version = j["version"];
  returnedConfig.architecture = j["architecture"];
  returnedConfig.config = j["config"];
  returnedConfig.metadata = j["metadata"];
  returnedConfig.weights = weights;

  return returnedConfig;
}

nlohmann::json MakeContainerModel(const std::vector<nlohmann::json>& models)
{
  nlohmann::json submodels = nlohmann::json::array();
  for (std::size_t index = 0; index < models.size(); ++index)
  {
    submodels.push_back(
      {{"max_value", static_cast<double>(index + 1U) / static_cast<double>(models.size())},
       {"model", models[index]}});
  }
  return {
    {"version", "0.7.0"},
    {"architecture", "SlimmableContainer"},
    {"sample_rate", 48000.0},
    {"metadata", nullptr},
    {"config", {{"submodels", std::move(submodels)}}},
    {"weights", nlohmann::json::array()},
  };
}

void test_gets_input_level()
{
  nam::dspData config = _GetConfig();
  std::unique_ptr<nam::DSP> dsp = get_dsp(config);
  assert(dsp->HasInputLevel());
}
void test_gets_output_level()
{
  nam::dspData config = _GetConfig();
  std::unique_ptr<nam::DSP> dsp = get_dsp(config);
  assert(dsp->HasOutputLevel());
}

void test_null_input_level()
{
  // Issue 129
  const std::string configStr =
    R"({"version": "0.5.4", "metadata": {"date": {"year": 2024, "month": 10, "day": 9, "hour": 18, "minute": 44, "second": 41}, "loudness": -37.8406867980957, "gain": 0.13508800804658277, "name": "Test LSTM", "modeled_by": "Steve", "gear_type": "amp", "gear_make": "Darkglass Electronics", "gear_model": "Microtubes 900 v2", "tone_type": "clean", "input_level_dbu": null, "output_level_dbu": 12.3, "training": {"settings": {"ignore_checks": false}, "data": {"latency": {"manual": null, "calibration": {"algorithm_version": 1, "delays": [-16], "safety_factor": 1, "recommended": -17, "warnings": {"matches_lookahead": false, "disagreement_too_high": false}}}, "checks": {"version": 3, "passed": true}}, "validation_esr": null}}, "architecture": "LSTM", "config": {"input_size": 1, "hidden_size": 3, "num_layers": 1}, "weights": [-0.21677088737487793, -0.6683622002601624, -0.2560940980911255, -0.3588429093360901, 0.17952610552310944, 0.19445613026618958, -0.01662646047770977, 0.5353694558143616, -0.2536540627479553, -0.5132213234901428, -0.020476307719945908, 0.08592455089092255, -0.6891753673553467, 0.3627359867095947, 0.008421811275184155, 0.3113192617893219, 0.14251480996608734, 0.07989779114723206, -0.18211324512958527, 0.7118963003158569, 0.41084015369415283, -0.6571938395500183, -0.13214066624641418, -0.2698603868484497, 0.49387243390083313, -0.3491725027561188, 0.6353667974472046, -0.5005152225494385, 0.2052856683731079, -0.4301638901233673, -0.15770092606544495, -0.7181791067123413, 0.056290093809366226, -0.49049463868141174, 0.6623441576957703, 0.09029324352741241, 0.34005245566368103, 0.16416560113430023, 0.15520110726356506, -0.4155678153038025, -0.36928507685661316, 0.3211132884025574, -0.6769840121269226, -0.1575538069009781, 0.05268515646457672, -0.4191459119319916, 0.599330484867096, 0.21518059074878693, -4.246325492858887, -3.315647840499878, -4.328850746154785, 4.496089458465576, 5.015639305114746, 3.6492037773132324, 0.14431169629096985, -0.6633821725845337, 0.11673200130462646, -0.1418764889240265, -0.4897872805595398, -0.8689419031143188, -0.06714004278182983, -0.4450395107269287, -0.02142983116209507, -0.15136894583702087, -2.775207996368408, -0.08681213855743408, 0.05702732503414154, 0.670292317867279, 0.31442636251449585, 0.30793967843055725], "sample_rate": 48000})";
  nam::dspData config = _GetConfig(configStr);
  // The first part of this is that the following line doesn't fail:
  std::unique_ptr<nam::DSP> dsp = get_dsp(config);

  assert(!dsp->HasInputLevel());
  assert(dsp->HasOutputLevel());
}

void test_null_output_level()
{
  // Issue 129
  const std::string configStr =
    R"({"version": "0.5.4", "metadata": {"date": {"year": 2024, "month": 10, "day": 9, "hour": 18, "minute": 44, "second": 41}, "loudness": -37.8406867980957, "gain": 0.13508800804658277, "name": "Test LSTM", "modeled_by": "Steve", "gear_type": "amp", "gear_make": "Darkglass Electronics", "gear_model": "Microtubes 900 v2", "tone_type": "clean", "input_level_dbu": 19.0, "output_level_dbu": null, "training": {"settings": {"ignore_checks": false}, "data": {"latency": {"manual": null, "calibration": {"algorithm_version": 1, "delays": [-16], "safety_factor": 1, "recommended": -17, "warnings": {"matches_lookahead": false, "disagreement_too_high": false}}}, "checks": {"version": 3, "passed": true}}, "validation_esr": null}}, "architecture": "LSTM", "config": {"input_size": 1, "hidden_size": 3, "num_layers": 1}, "weights": [-0.21677088737487793, -0.6683622002601624, -0.2560940980911255, -0.3588429093360901, 0.17952610552310944, 0.19445613026618958, -0.01662646047770977, 0.5353694558143616, -0.2536540627479553, -0.5132213234901428, -0.020476307719945908, 0.08592455089092255, -0.6891753673553467, 0.3627359867095947, 0.008421811275184155, 0.3113192617893219, 0.14251480996608734, 0.07989779114723206, -0.18211324512958527, 0.7118963003158569, 0.41084015369415283, -0.6571938395500183, -0.13214066624641418, -0.2698603868484497, 0.49387243390083313, -0.3491725027561188, 0.6353667974472046, -0.5005152225494385, 0.2052856683731079, -0.4301638901233673, -0.15770092606544495, -0.7181791067123413, 0.056290093809366226, -0.49049463868141174, 0.6623441576957703, 0.09029324352741241, 0.34005245566368103, 0.16416560113430023, 0.15520110726356506, -0.4155678153038025, -0.36928507685661316, 0.3211132884025574, -0.6769840121269226, -0.1575538069009781, 0.05268515646457672, -0.4191459119319916, 0.599330484867096, 0.21518059074878693, -4.246325492858887, -3.315647840499878, -4.328850746154785, 4.496089458465576, 5.015639305114746, 3.6492037773132324, 0.14431169629096985, -0.6633821725845337, 0.11673200130462646, -0.1418764889240265, -0.4897872805595398, -0.8689419031143188, -0.06714004278182983, -0.4450395107269287, -0.02142983116209507, -0.15136894583702087, -2.775207996368408, -0.08681213855743408, 0.05702732503414154, 0.670292317867279, 0.31442636251449585, 0.30793967843055725], "sample_rate": 48000})";
  nam::dspData config = _GetConfig(configStr);
  // The first part of this is that the following line doesn't fail:
  std::unique_ptr<nam::DSP> dsp = get_dsp(config);
  assert(dsp->HasInputLevel());
  assert(!dsp->HasOutputLevel());
}

void test_missing_metadata_is_optional()
{
  nlohmann::json config = nlohmann::json::parse(basicConfigStr);
  config.erase("metadata");
  const std::unique_ptr<nam::DSP> dsp = nam::get_dsp(config);
  assert(dsp != nullptr);
  assert(!dsp->HasInputLevel());
  assert(!dsp->HasOutputLevel());
}

void test_missing_required_top_level_field_throws()
{
  nlohmann::json config = nlohmann::json::parse(basicConfigStr);
  config.erase("config");
  bool threw = false;
  try
  {
    nam::get_dsp(config);
  }
  catch (const std::exception&)
  {
    threw = true;
  }
  assert(threw);
}

void test_rejects_malformed_or_unsupported_versions()
{
  const std::vector<std::string> invalid_versions = {
    "0.7", "0.7.0.1", "0.7.0junk", " 0.7.0", "+0.7.0", "0.8.0"};
  for (const auto& version : invalid_versions)
  {
    nlohmann::json config = nlohmann::json::parse(basicConfigStr);
    config["version"] = version;
    bool threw = false;
    try
    {
      nam::get_dsp(config);
    }
    catch (const std::exception&)
    {
      threw = true;
    }
    assert(threw);
  }

  nlohmann::json supported = nlohmann::json::parse(basicConfigStr);
  supported["version"] = "0.7.0";
  assert(nam::get_dsp(supported) != nullptr);
}

void test_rejects_unsafe_numeric_metadata()
{
  nlohmann::json config = nlohmann::json::parse(basicConfigStr);
  config["sample_rate"] = 2.0e9;
  bool threw = false;
  try
  {
    nam::get_dsp(config);
  }
  catch (const std::exception&)
  {
    threw = true;
  }
  assert(threw);

  for (const char* field : {"loudness", "input_level_dbu", "output_level_dbu"})
  {
    config = nlohmann::json::parse(basicConfigStr);
    config["metadata"][field] = 1.0e100;
    threw = false;
    try
    {
      nam::get_dsp(config);
    }
    catch (const std::exception&)
    {
      threw = true;
    }
    assert(threw);
  }
}

void test_null_sample_rate_is_unknown()
{
  nlohmann::json config = nlohmann::json::parse(basicConfigStr);
  config["sample_rate"] = nullptr;
  const std::unique_ptr<nam::DSP> dsp = nam::get_dsp(config);
  assert(dsp != nullptr);
  assert(dsp->GetExpectedSampleRate() == NAM_UNKNOWN_EXPECTED_SAMPLE_RATE);
}

void test_rejects_non_integer_architecture_dimensions()
{
  nlohmann::json config = nlohmann::json::parse(basicConfigStr);
  config["config"]["num_layers"] = 1.5;
  bool threw = false;
  try
  {
    nam::get_dsp(config);
  }
  catch (const std::exception&)
  {
    threw = true;
  }
  assert(threw);

  config = nlohmann::json::parse(basicConfigStr);
  config["config"]["hidden_size"] = 1.0e30;
  threw = false;
  try
  {
    nam::get_dsp(config);
  }
  catch (const std::exception&)
  {
    threw = true;
  }
  assert(threw);
}

void test_rejects_unsafe_linear_resources_before_allocation()
{
  bool threw = false;
  try
  {
    const nam::Linear linear(1, 1, 1 << 20, false, std::vector<float>{0.0f}, 48000.0);
  }
  catch (const std::exception&)
  {
    threw = true;
  }
  assert(threw);

  threw = false;
  try
  {
    const nam::Linear linear(1, 4097, 1, false, std::vector<float>{0.0f}, 48000.0);
  }
  catch (const std::exception&)
  {
    threw = true;
  }
  assert(threw);
}

void test_rejects_unsafe_aggregate_prewarm_buffers()
{
  nlohmann::json config = nlohmann::json::parse(basicConfigStr);
  config["config"]["out_channels"] = 4096;
  // One 3-unit LSTM layer has 66 recurrent values; its 4096x(3+1)
  // output head has 16384 more.
  config["weights"] = std::vector<float>(16450, 0.0f);
  bool threw = false;
  try
  {
    nam::get_dsp(config);
  }
  catch (const std::exception&)
  {
    threw = true;
  }
  assert(threw);
}

void test_rejects_excessive_lstm_depth()
{
  nlohmann::json config = nlohmann::json::parse(basicConfigStr);
  config["config"]["num_layers"] = 17;
  bool threw = false;
  try
  {
    nam::get_dsp(config);
  }
  catch (const std::exception&)
  {
    threw = true;
  }
  assert(threw);
}

void test_rejects_excessive_convolution_prewarm_work()
{
  nlohmann::json config = {
    {"version", "0.5.4"},
    {"metadata", nullptr},
    {"architecture", "ConvNet"},
    {"config",
     {{"channels", 1},
      {"dilations", std::vector<int>(3900, 256)},
      {"batchnorm", false},
      {"activation", "Tanh"}}},
    {"weights", nlohmann::json::array()},
    {"sample_rate", 48000}};
  bool threw = false;
  try
  {
    nam::get_dsp(config);
  }
  catch (const std::exception&)
  {
    threw = true;
  }
  assert(threw);

  config = {{"version", "0.5.4"},
            {"metadata", nullptr},
            {"architecture", "WaveNet"},
            {"config",
             {{"layers",
               {{{"input_size", 1},
                 {"condition_size", 1},
                 {"head_size", 1},
                 {"channels", 1},
                 {"kernel_size", 2},
                 {"dilations", std::vector<int>(1500, 250)},
                 {"activation", "Tanh"},
                 {"gated", false},
                 {"head_bias", false}}}},
              {"head", nullptr},
              {"head_scale", 0.02f}}},
            {"weights", nlohmann::json::array()},
            {"sample_rate", 48000}};
  threw = false;
  try
  {
    nam::get_dsp(config);
  }
  catch (const std::exception&)
  {
    threw = true;
  }
  assert(threw);
}

void test_rejects_width_amplified_prewarm_compute()
{
  nlohmann::json config = {
    {"version", "0.5.4"},
    {"metadata", nullptr},
    {"architecture", "ConvNet"},
    {"config",
     {{"in_channels", 1},
      {"out_channels", 4095},
      {"channels", 30},
      {"dilations", {500000}},
      {"batchnorm", false},
      {"activation", "Tanh"}}},
    {"weights", nlohmann::json::array()},
    {"sample_rate", 48000}};
  bool threw = false;
  try
  {
    nam::get_dsp(config);
  }
  catch (const std::exception&)
  {
    threw = true;
  }
  assert(threw);

  config = nlohmann::json::parse(basicConfigStr);
  config["config"] = {
    {"in_channels", 1}, {"out_channels", 4095}, {"input_size", 1}, {"hidden_size", 30}, {"num_layers", 1}};
  config["weights"] = std::vector<float>(130845, 0.0f);
  config["sample_rate"] = 768000;
  threw = false;
  try
  {
    nam::get_dsp(config);
  }
  catch (const std::exception&)
  {
    threw = true;
  }
  assert(threw);

  config = {{"version", "0.5.4"},
            {"metadata", nullptr},
            {"architecture", "WaveNet"},
            {"config",
             {{"layers",
               {{{"input_size", 1},
                 {"condition_size", 1},
                 {"head_size", 4095},
                 {"channels", 1},
                 {"bottleneck", 30},
                 {"kernel_size", 2},
                 {"dilations", {500000}},
                 {"activation", "Tanh"},
                 {"gated", false},
                 {"head_bias", false}}}},
              {"head", nullptr},
              {"head_scale", 0.02f}}},
            {"weights", nlohmann::json::array()},
            {"sample_rate", 48000}};
  threw = false;
  try
  {
    nam::get_dsp(config);
  }
  catch (const std::exception&)
  {
    threw = true;
  }
  assert(threw);
}

void test_rejects_nonfinite_weights_and_invalid_metadata_shape()
{
  nlohmann::json config = nlohmann::json::parse(basicConfigStr);
  config["weights"][0] = std::numeric_limits<double>::infinity();
  bool threw = false;
  try
  {
    nam::get_dsp(config);
  }
  catch (const std::exception&)
  {
    threw = true;
  }
  assert(threw);

  config = nlohmann::json::parse(basicConfigStr);
  config["metadata"] = nlohmann::json::array();
  threw = false;
  try
  {
    nam::get_dsp(config);
  }
  catch (const std::exception&)
  {
    threw = true;
  }
  assert(threw);
}

void test_rejects_nonfinite_direct_dsp_data_weights()
{
  for (const float nonfinite : {std::numeric_limits<float>::infinity(),
                                -std::numeric_limits<float>::infinity(),
                                std::numeric_limits<float>::quiet_NaN()})
  {
    nam::dspData config = _GetConfig();
    config.weights.front() = nonfinite;
    bool threw = false;
    try
    {
      nam::get_dsp(config);
    }
    catch (const std::invalid_argument& error)
    {
      threw = std::string(error.what()).find("finite") != std::string::npos;
    }
    assert(threw);
  }
}

void test_rejects_excessive_model_weight_count_before_reserve()
{
  nam::model_validation::require_model_weight_limit(nam::model_validation::kMaxModelWeightValues);

  nlohmann::json config = nlohmann::json::parse(basicConfigStr);
  auto& weights = config["weights"].get_ref<nlohmann::json::array_t&>();
  weights.resize(nam::model_validation::kMaxModelWeightValues + 1U, nlohmann::json(0.0));

  bool threw = false;
  try
  {
    nam::get_dsp(config);
  }
  catch (const std::invalid_argument& error)
  {
    threw = std::string(error.what()).find("Model weights") != std::string::npos;
  }
  assert(threw);
}

void test_rejects_aggregate_nested_model_weights()
{
  const nlohmann::json child = nlohmann::json::parse(basicConfigStr);
  nlohmann::json container = MakeContainerModel({child, child});
  auto& submodels = container["config"]["submodels"];
  const std::size_t child_weights = nam::model_validation::kMaxModelWeightValues / 2U + 1U;
  for (auto& entry : submodels)
  {
    auto& weights = entry["model"]["weights"].get_ref<nlohmann::json::array_t&>();
    weights.resize(child_weights, nlohmann::json(0.0));
  }

  bool threw = false;
  try
  {
    nam::get_dsp(container);
  }
  catch (const std::invalid_argument& error)
  {
    threw = std::string(error.what()).find("aggregate element count") != std::string::npos;
  }
  assert(threw);

  nam::dspData direct_config;
  direct_config.version = "0.7.0";
  direct_config.architecture = "SlimmableContainer";
  direct_config.config = std::move(container["config"]);
  direct_config.metadata = nullptr;
  direct_config.expected_sample_rate = 48000.0;
  threw = false;
  try
  {
    nam::get_dsp(direct_config);
  }
  catch (const std::invalid_argument& error)
  {
    threw = std::string(error.what()).find("aggregate element count") != std::string::npos;
  }
  assert(threw);
}

void test_rejects_excessive_aggregate_model_count()
{
  const nlohmann::json leaf = nlohmann::json::parse(basicConfigStr);
  const std::vector<nlohmann::json> leaves(8U, leaf);
  const nlohmann::json branch = MakeContainerModel(leaves);
  const std::vector<nlohmann::json> branches(8U, branch);
  const nlohmann::json root = MakeContainerModel(branches);

  bool threw = false;
  try
  {
    nam::get_dsp(root);
  }
  catch (const std::invalid_argument& error)
  {
    threw = std::string(error.what()).find("model count") != std::string::npos;
  }
  assert(threw);
}

void test_rejects_excessive_config_and_metadata_json_budgets()
{
  nlohmann::json config = nlohmann::json::parse(basicConfigStr);
  config["config"]["oversized_text"] =
    std::string(nam::model_validation::kMaxModelGraphTextBytes + 1U, 'x');
  bool threw = false;
  try
  {
    nam::get_dsp(config);
  }
  catch (const std::invalid_argument& error)
  {
    threw = std::string(error.what()).find("text size") != std::string::npos;
  }
  assert(threw);

  config = nlohmann::json::parse(basicConfigStr);
  config["metadata"]["oversized_nodes"] = nlohmann::json::array();
  auto& nodes = config["metadata"]["oversized_nodes"].get_ref<nlohmann::json::array_t&>();
  nodes.resize(nam::model_validation::kMaxModelGraphJsonNodes + 1U, nlohmann::json(nullptr));
  threw = false;
  try
  {
    nam::get_dsp(config);
  }
  catch (const std::invalid_argument& error)
  {
    threw = std::string(error.what()).find("JSON complexity") != std::string::npos;
  }
  assert(threw);
}

void test_rejects_excessive_nested_condition_models()
{
  nlohmann::json nested = nlohmann::json::parse(basicConfigStr);
  for (int depth = 0; depth < 5; ++depth)
  {
    nested = {{"version", "0.5.4"},
              {"metadata", nullptr},
              {"architecture", "WaveNet"},
              {"config", {{"global_condition_size", 0}, {"condition_dsp", nested}}},
              {"weights", nlohmann::json::array()},
              {"sample_rate", 48000}};
  }

  bool threw = false;
  try
  {
    nam::get_dsp(nested);
  }
  catch (const std::exception&)
  {
    threw = true;
  }
  assert(threw);
}

void test_rejects_mismatched_prelu_channels()
{
  const nlohmann::json config = {
    {"version", "0.5.4"},
    {"metadata", nullptr},
    {"architecture", "ConvNet"},
    {"config",
     {{"channels", 1},
      {"dilations", {1}},
      {"batchnorm", false},
      {"activation", {{"type", "PReLU"}, {"negative_slopes", {0.1f, 0.2f}}}}}},
    {"weights", {0.0f, 0.0f, 0.0f, 0.0f, 0.0f}},
    {"sample_rate", 48000.0}};
  bool threw = false;
  try
  {
    nam::get_dsp(config);
  }
  catch (const std::runtime_error&)
  {
    threw = true;
  }
  assert(threw);
}

// Helper function to process buffers through a DSP model
void process_buffers(nam::DSP* dsp, int num_buffers, int buffer_size)
{
  const int in_channels = dsp->NumInputChannels();
  const int out_channels = dsp->NumOutputChannels();
  const double sample_rate = dsp->GetExpectedSampleRate() > 0 ? dsp->GetExpectedSampleRate() : 48000.0;

  // Reset the model
  dsp->Reset(sample_rate, buffer_size);

  // Allocate buffers for all channels
  std::vector<std::vector<NAM_SAMPLE>> inputBuffers(in_channels);
  std::vector<std::vector<NAM_SAMPLE>> outputBuffers(out_channels);
  std::vector<NAM_SAMPLE*> inputPtrs(in_channels);
  std::vector<NAM_SAMPLE*> outputPtrs(out_channels);

  for (int ch = 0; ch < in_channels; ch++)
  {
    inputBuffers[ch].resize(buffer_size, (NAM_SAMPLE)0.0);
    inputPtrs[ch] = inputBuffers[ch].data();
  }
  for (int ch = 0; ch < out_channels; ch++)
  {
    outputBuffers[ch].resize(buffer_size, (NAM_SAMPLE)0.0);
    outputPtrs[ch] = outputBuffers[ch].data();
  }

  double output_peak = 0.0;
  // Process num_buffers buffers
  for (int buf = 0; buf < num_buffers; buf++)
  {
    // Fill input with some test data (simple sine-like pattern)
    for (int ch = 0; ch < in_channels; ch++)
    {
      for (int i = 0; i < buffer_size; i++)
      {
        inputBuffers[ch][i] = (NAM_SAMPLE)(0.1 * (ch + 1) * ((buf * buffer_size + i) % 100) / 100.0);
      }
    }

    // Process the buffer
    dsp->process(inputPtrs.data(), outputPtrs.data(), buffer_size);

    // Verify output is finite
    for (int ch = 0; ch < out_channels; ch++)
    {
      for (int i = 0; i < buffer_size; i++)
      {
        assert(std::isfinite(outputBuffers[ch][i]));
        output_peak = std::max(output_peak, std::abs(static_cast<double>(outputBuffers[ch][i])));
      }
    }
  }
  assert(output_peak > 1.0e-8);
}

void test_load_and_process_nam_files()
{
  // Test loading and processing three different .nam files
  // Paths are relative to root directory where tests run (./build/tools/run_tests)
  const std::vector<std::string> nam_files = {"example_models/wavenet.nam", "example_models/lstm.nam",
                                              "example_models/wavenet_condition_dsp.nam",
                                              "example_models/wavenet_a2_max.nam"};

  const int num_buffers = 3;
  const int buffer_size = 64;

  for (const auto& nam_file : nam_files)
  {
    std::filesystem::path model_path(nam_file);

    // Load the model
    std::unique_ptr<nam::DSP> dsp = nam::get_dsp(model_path);
    assert(dsp != nullptr);

    // Process buffers through the model
    process_buffers(dsp.get(), num_buffers, buffer_size);
  }
}

void test_rejects_oversized_model_file_before_parsing()
{
  const auto model_path = std::filesystem::temp_directory_path() / "dessamp-oversized-model-test.nam";
  std::error_code cleanup_error;
  std::filesystem::remove(model_path, cleanup_error);
  {
    std::ofstream model(model_path, std::ios::binary);
    assert(model.good());
    model.seekp(static_cast<std::streamoff>(nam::kMaxSerializedModelBytes));
    model.put('\0');
  }

  bool threw = false;
  try
  {
    nam::get_dsp(model_path);
  }
  catch (const std::invalid_argument& error)
  {
    threw = std::string(error.what()).find("Model file size") != std::string::npos;
  }
  std::filesystem::remove(model_path, cleanup_error);
  assert(threw);
}

void test_rejects_parse_time_json_event_amplification()
{
  const auto model_path =
    std::filesystem::temp_directory_path() / "dessamp-json-event-amplification-test.nam";
  std::error_code cleanup_error;
  std::filesystem::remove(model_path, cleanup_error);
  {
    std::ofstream model(model_path, std::ios::binary);
    assert(model.good());
    model << "{\"excessive\":[";
    for (std::size_t index = 0U; index < nam::model_validation::kMaxModelGraphJsonNodes; ++index)
    {
      if (index != 0U)
        model.put(',');
      model.put('0');
    }
    model << "]}";
  }

  bool threw = false;
  try
  {
    nam::get_dsp(model_path);
  }
  catch (const std::invalid_argument& error)
  {
    threw = std::string(error.what()).find("parse-time JSON node count") != std::string::npos;
  }
  std::filesystem::remove(model_path, cleanup_error);
  assert(threw);
}

void test_rejects_parse_time_structural_node_amplification()
{
  const auto model_path =
    std::filesystem::temp_directory_path() / "dessamp-json-structural-amplification-test.nam";
  std::error_code cleanup_error;
  std::filesystem::remove(model_path, cleanup_error);
  {
    std::ofstream model(model_path, std::ios::binary);
    assert(model.good());
    model << "{\"excessive\":[";
    for (std::size_t index = 0U; index < nam::model_validation::kMaxModelGraphJsonNodes; ++index)
    {
      if (index != 0U)
        model.put(',');
      model << "[]";
    }
    model << "]}";
  }

  bool threw = false;
  try
  {
    nam::get_dsp(model_path);
  }
  catch (const std::invalid_argument& error)
  {
    threw = std::string(error.what()).find("parse-time JSON node count") != std::string::npos;
  }
  std::filesystem::remove(model_path, cleanup_error);
  assert(threw);
}

}; // namespace test_get_dsp
