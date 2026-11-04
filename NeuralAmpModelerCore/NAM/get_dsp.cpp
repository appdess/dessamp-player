#include <algorithm>
#include <cctype>
#include <cmath>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <unordered_set>
#include <vector>

#include "dsp.h"
#include "registry.h"
#include "json.hpp"
#include "lstm.h"
#include "model_validation.h"
#include "convnet.h"
#include "wavenet.h"
#include "get_dsp.h"

namespace nam
{
namespace
{
constexpr int kMaxNestedModelDepth = 4;
thread_local int gNestedModelDepth = 0;

struct ModelLoadBudget
{
  std::size_t declared_models = 0U;
  std::size_t dispatched_models = 0U;
  std::size_t weight_values = 0U;
  std::size_t json_nodes = 0U;
  std::size_t text_bytes = 0U;
  bool json_graph_validated = false;
};

thread_local ModelLoadBudget* gModelLoadBudget = nullptr;

class ModelLoadBudgetScope
{
public:
  ModelLoadBudgetScope()
  : mPrevious(gModelLoadBudget)
  {
    if (gModelLoadBudget == nullptr)
      gModelLoadBudget = &mOwned;
  }

  ~ModelLoadBudgetScope()
  {
    if (mPrevious == nullptr)
      gModelLoadBudget = nullptr;
  }

  ModelLoadBudget& Get() const { return *gModelLoadBudget; }

private:
  ModelLoadBudget mOwned;
  ModelLoadBudget* const mPrevious;
};

void ConsumeBounded(std::size_t& total, const std::size_t amount, const std::size_t limit,
                    const char* error_message)
{
  if (amount > limit - total)
    throw std::invalid_argument(error_message);
  total += amount;
}

void ConsumeModelDeclaration(ModelLoadBudget& budget)
{
  ConsumeBounded(budget.declared_models, 1U, model_validation::kMaxModelGraphModels,
                 "Model graph exceeds the supported model count");
}

void ConsumeModelDispatch(ModelLoadBudget& budget)
{
  ConsumeBounded(budget.dispatched_models, 1U, model_validation::kMaxModelGraphModels,
                 "Model graph exceeds the supported model count");
}

void ConsumeJsonNode(ModelLoadBudget& budget)
{
  ConsumeBounded(budget.json_nodes, 1U, model_validation::kMaxModelGraphJsonNodes,
                 "Model config and metadata exceed the supported JSON complexity");
}

void ConsumeText(ModelLoadBudget& budget, const std::size_t bytes)
{
  ConsumeBounded(budget.text_bytes, bytes, model_validation::kMaxModelGraphTextBytes,
                 "Model config and metadata exceed the supported text size");
}

void ConsumeWeights(ModelLoadBudget& budget, const std::size_t values)
{
  model_validation::require_model_weight_limit(values);
  ConsumeBounded(budget.weight_values, values, model_validation::kMaxModelWeightValues,
                 "Model graph weights exceed the supported aggregate element count");
}

// Run a non-DOM SAX pass over file-backed JSON before nlohmann allocates the
// parsed tree. Numeric elements of any direct object member named `weights`
// consume the aggregate weight allowance; every container and every other
// scalar consumes the non-weight JSON-node allowance. The post-DOM validator
// still proves that each such array belongs to a real serialized model and is
// flat/numeric. Treating an unrelated direct `weights` array as weights here is
// safe: it can consume no more memory than the accepted aggregate model-weight
// ceiling and is rejected semantically after the bounded DOM exists.
class SerializedJsonPreflight final : public nlohmann::json_sax<nlohmann::json>
{
public:
  using number_integer_t = nlohmann::json::number_integer_t;
  using number_unsigned_t = nlohmann::json::number_unsigned_t;
  using number_float_t = nlohmann::json::number_float_t;
  using string_t = nlohmann::json::string_t;
  using binary_t = nlohmann::json::binary_t;

  bool null() override { return ConsumeScalar(false); }
  bool boolean(bool) override { return ConsumeScalar(false); }
  bool number_integer(number_integer_t) override { return ConsumeScalar(true); }
  bool number_unsigned(number_unsigned_t) override { return ConsumeScalar(true); }
  bool number_float(number_float_t, const string_t&) override { return ConsumeScalar(true); }

  bool string(string_t& value) override
  {
    ConsumeTextBytes(value.size());
    return ConsumeScalar(false);
  }

  bool binary(binary_t& value) override
  {
    ConsumeTextBytes(value.size());
    return ConsumeScalar(false);
  }

  bool start_object(std::size_t) override
  {
    ConsumeNode();
    ClearParentObjectKey();
    mFrames.push_back({ContainerKind::Object, false, {}});
    return true;
  }

  bool key(string_t& value) override
  {
    ConsumeTextBytes(value.size());
    if (mFrames.empty() || mFrames.back().kind != ContainerKind::Object)
      return false;
    mFrames.back().pending_key = value;
    return true;
  }

  bool end_object() override { return PopFrame(ContainerKind::Object); }

  bool start_array(std::size_t) override
  {
    bool is_weight_array = false;
    if (!mFrames.empty() && mFrames.back().kind == ContainerKind::Object)
      is_weight_array = mFrames.back().pending_key == "weights";
    ConsumeNode();
    ClearParentObjectKey();
    mFrames.push_back({ContainerKind::Array, is_weight_array, {}});
    return true;
  }

  bool end_array() override { return PopFrame(ContainerKind::Array); }

  bool parse_error(std::size_t, const std::string&, const nlohmann::detail::exception&) override
  {
    return false;
  }

private:
  enum class ContainerKind
  {
    Object,
    Array,
  };

  struct Frame
  {
    ContainerKind kind;
    bool is_weight_array;
    string_t pending_key;
  };

  bool ConsumeScalar(const bool numeric)
  {
    if (numeric && !mFrames.empty() && mFrames.back().kind == ContainerKind::Array
        && mFrames.back().is_weight_array)
    {
      ConsumeBounded(mWeightValues, 1U, model_validation::kMaxModelWeightValues,
                     "Model file exceeds the supported parse-time weight count");
    }
    else
    {
      ConsumeNode();
    }
    ClearParentObjectKey();
    return true;
  }

  void ConsumeNode()
  {
    ConsumeBounded(mJsonNodes, 1U, model_validation::kMaxModelGraphJsonNodes,
                   "Model file exceeds the supported parse-time JSON node count");
  }

  void ClearParentObjectKey()
  {
    if (!mFrames.empty() && mFrames.back().kind == ContainerKind::Object)
      mFrames.back().pending_key.clear();
  }

  bool PopFrame(const ContainerKind expected)
  {
    if (mFrames.empty() || mFrames.back().kind != expected)
      return false;
    mFrames.pop_back();
    return true;
  }

  void ConsumeTextBytes(const std::size_t bytes)
  {
    ConsumeBounded(mTextBytes, bytes, model_validation::kMaxModelGraphTextBytes,
                   "Model file exceeds the supported parse-time JSON text size");
  }

  std::vector<Frame> mFrames;
  std::size_t mJsonNodes = 0U;
  std::size_t mWeightValues = 0U;
  std::size_t mTextBytes = 0U;
};

void PreflightSerializedJson(const std::string& serialized)
{
  SerializedJsonPreflight preflight;
  if (!nlohmann::json::sax_parse(serialized.begin(), serialized.end(), &preflight))
    throw std::invalid_argument("Model file contains invalid JSON");
}

bool IsSerializedModelObject(const nlohmann::json& value)
{
  return value.is_object() && value.find("version") != value.end()
         && value.find("architecture") != value.end() && value.find("config") != value.end()
         && value.find("weights") != value.end() && value.at("weights").is_array();
}

void ValidateSerializedWeights(const nlohmann::json& weights, ModelLoadBudget& budget)
{
  ConsumeWeights(budget, weights.size());
  for (const auto& weight : weights)
    (void)model_validation::json_float(weight, "Model weight");
}

void ValidateJsonGraph(const nlohmann::json& root, ModelLoadBudget& budget)
{
  std::vector<const nlohmann::json*> pending;
  ConsumeJsonNode(budget);
  pending.push_back(&root);

  while (!pending.empty())
  {
    const nlohmann::json& value = *pending.back();
    pending.pop_back();

    if (value.is_object())
    {
      const bool serialized_model = IsSerializedModelObject(value);
      if (serialized_model)
        ConsumeModelDeclaration(budget);

      for (auto item = value.begin(); item != value.end(); ++item)
      {
        ConsumeText(budget, item.key().size());
        if (serialized_model && item.key() == "weights")
        {
          ConsumeJsonNode(budget);
          ValidateSerializedWeights(*item, budget);
          continue;
        }
        ConsumeJsonNode(budget);
        pending.push_back(&*item);
      }
    }
    else if (value.is_array())
    {
      for (const auto& element : value)
      {
        ConsumeJsonNode(budget);
        pending.push_back(&element);
      }
    }
    else if (value.is_string())
    {
      ConsumeText(budget, value.get_ref<const nlohmann::json::string_t&>().size());
    }
    else if (value.is_binary())
    {
      ConsumeText(budget, value.get_binary().size());
    }
  }
}

void ValidateSerializedModelGraphOnce(const nlohmann::json& config, ModelLoadBudget& budget)
{
  if (budget.json_graph_validated)
    return;
  ValidateJsonGraph(config, budget);
  budget.json_graph_validated = true;
}

void ValidateDspDataGraphOnce(const dspData& conf, ModelLoadBudget& budget)
{
  if (budget.json_graph_validated)
    return;

  ConsumeModelDeclaration(budget);
  ConsumeText(budget, conf.version.size());
  ConsumeText(budget, conf.architecture.size());
  ConsumeWeights(budget, conf.weights.size());
  ValidateJsonGraph(conf.config, budget);
  ValidateJsonGraph(conf.metadata, budget);
  budget.json_graph_validated = true;
}

void ValidateFiniteWeights(const std::vector<float>& weights)
{
  for (const float weight : weights)
  {
    if (!std::isfinite(weight))
      throw std::invalid_argument("Model weights must be finite and representable as floats");
  }
}

class ModelDepthGuard
{
public:
  ModelDepthGuard()
  {
    if (gNestedModelDepth >= kMaxNestedModelDepth)
      throw std::invalid_argument("Nested condition DSP depth exceeds the supported limit");
    ++gNestedModelDepth;
  }

  ~ModelDepthGuard() { --gNestedModelDepth; }
};

} // namespace

struct Version
{
  int major;
  int minor;
  int patch;
};

Version ParseVersion(const std::string& versionStr)
{
  const std::size_t firstDot = versionStr.find('.');
  const std::size_t secondDot = firstDot == std::string::npos ? std::string::npos : versionStr.find('.', firstDot + 1U);
  if (firstDot == std::string::npos || secondDot == std::string::npos
      || versionStr.find('.', secondDot + 1U) != std::string::npos)
    throw std::invalid_argument("Invalid version string: " + versionStr);

  const std::string components[] = {
    versionStr.substr(0U, firstDot),
    versionStr.substr(firstDot + 1U, secondDot - firstDot - 1U),
    versionStr.substr(secondDot + 1U),
  };
  for (const auto& component : components)
  {
    if (component.empty()
        || !std::all_of(component.begin(), component.end(), [](const unsigned char value) { return std::isdigit(value); }))
      throw std::invalid_argument("Invalid version string: " + versionStr);
  }

  Version version;
  try
  {
    version.major = std::stoi(components[0]);
    version.minor = std::stoi(components[1]);
    version.patch = std::stoi(components[2]);
  }
  catch (const std::invalid_argument&)
  {
    throw std::invalid_argument("Invalid version string: " + versionStr);
  }
  catch (const std::out_of_range&)
  {
    throw std::out_of_range("Version string out of range: " + versionStr);
  }

  return version;
}

void verify_config_version(const std::string versionStr)
{
  Version version = ParseVersion(versionStr);
  // NAM 0.5 introduced the current JSON family; 0.7 adds A2 slimmable
  // containers while keeping each child model self-describing.
  if (version.major != 0 || version.minor < 5 || version.minor > 7)
  {
    std::stringstream ss;
    ss << "Model config is an unsupported version " << versionStr
       << ". Try either converting the model to a more recent version, or "
          "update your version of the NAM plugin.";
    throw std::runtime_error(ss.str());
  }
}

std::vector<float> GetWeights(nlohmann::json const& j)
{
  auto it = j.find("weights");
  if (it == j.end())
    throw std::runtime_error("Corrupted model file is missing weights.");
  if (!it->is_array())
    throw std::invalid_argument("Model weights must be an array");
  model_validation::require_model_weight_limit(it->size());

  std::vector<float> weights;
  weights.reserve(it->size());
  for (const auto& weight : *it)
    weights.push_back(model_validation::json_float(weight, "Model weight"));
  return weights;
}

std::unique_ptr<DSP> get_dsp_without_prewarm(dspData& conf);

std::unique_ptr<DSP> LoadDspData(dspData& conf)
{
  ConsumeModelDispatch(*gModelLoadBudget);
  model_validation::require_model_weight_limit(conf.weights.size());
  ValidateFiniteWeights(conf.weights);
  ModelDepthGuard model_depth_guard;
  std::unique_ptr<DSP> out = get_dsp_without_prewarm(conf);
  out->prewarm();
  return out;
}

std::unique_ptr<DSP> get_dsp(const std::filesystem::path config_filename)
{
  dspData temp;
  return get_dsp(config_filename, temp);
}

std::unique_ptr<DSP> get_dsp(const nlohmann::json& config)
{
  dspData temp;
  return get_dsp(config, temp);
}

std::unique_ptr<DSP> get_dsp(const std::filesystem::path config_filename, dspData& returnedConfig)
{
  if (!std::filesystem::exists(config_filename))
    throw std::runtime_error("Config file doesn't exist!\n");
  if (!std::filesystem::is_regular_file(config_filename))
    throw std::invalid_argument("Config path must identify a regular file");
  std::ifstream input(config_filename, std::ios::binary | std::ios::ate);
  if (!input)
    throw std::runtime_error("Could not open model file");

  const std::streamoff file_size = input.tellg();
  if (file_size <= 0 || static_cast<std::uintmax_t>(file_size) > kMaxSerializedModelBytes)
    throw std::invalid_argument("Model file size is outside the supported resource limit");

  std::string serialized(static_cast<std::size_t>(file_size), '\0');
  input.seekg(0, std::ios::beg);
  input.read(serialized.data(), static_cast<std::streamsize>(serialized.size()));
  if (input.gcount() != static_cast<std::streamsize>(serialized.size()))
    throw std::runtime_error("Model file changed while it was being read");
  if (input.peek() != std::ifstream::traits_type::eof())
    throw std::invalid_argument("Model file grew while it was being read");

  PreflightSerializedJson(serialized);
  const nlohmann::json j = nlohmann::json::parse(serialized.begin(), serialized.end());
  return get_dsp(j, returnedConfig);
}

std::unique_ptr<DSP> get_dsp(const nlohmann::json& config, dspData& returnedConfig)
{
  ModelLoadBudgetScope budget_scope;
  ValidateSerializedModelGraphOnce(config, budget_scope.Get());

  const std::string version = config.at("version").get<std::string>();
  const std::string architecture = config.at("architecture").get<std::string>();
  nlohmann::json config_json = config.at("config");
  verify_config_version(version);

  std::vector<float> weights = GetWeights(config);

  // Assign values to returnedConfig
  returnedConfig.version = version;
  returnedConfig.architecture = architecture;
  returnedConfig.config = config_json;
  const auto metadata = config.find("metadata");
  returnedConfig.metadata = metadata == config.end() ? nlohmann::json(nullptr) : *metadata;
  if (!returnedConfig.metadata.is_null() && !returnedConfig.metadata.is_object())
    throw std::invalid_argument("Model metadata must be an object or null");
  returnedConfig.weights = weights;
  returnedConfig.expected_sample_rate = nam::get_sample_rate_from_nam_file(config);

  /*Copy to a new dsp_config object for get_dsp below,
   since not sure if weights actually get modified as being non-const references on some
   model constructors inside get_dsp(dsp_config& conf).
   We need to return unmodified version of dsp_config via returnedConfig.*/
  dspData conf = returnedConfig;

  return LoadDspData(conf);
}

struct OptionalValue
{
  bool have = false;
  double value = 0.0;
};

std::unique_ptr<DSP> get_dsp(dspData& conf)
{
  ModelLoadBudgetScope budget_scope;
  ValidateDspDataGraphOnce(conf, budget_scope.Get());
  return LoadDspData(conf);
}

std::unique_ptr<DSP> get_dsp_without_prewarm(dspData& conf)
{
  verify_config_version(conf.version);

  auto& architecture = conf.architecture;
  nlohmann::json& config = conf.config;
  std::vector<float>& weights = conf.weights;
  OptionalValue loudness, inputLevel, outputLevel;

  auto AssignOptional = [&conf](const std::string key, OptionalValue& v) {
    if (conf.metadata.find(key) != conf.metadata.end())
    {
      if (!conf.metadata[key].is_null())
      {
        v.value = conf.metadata[key];
        v.have = true;
      }
    }
  };

  if (!conf.metadata.is_null())
  {
    AssignOptional("loudness", loudness);
    AssignOptional("input_level_dbu", inputLevel);
    AssignOptional("output_level_dbu", outputLevel);
  }
  const double expectedSampleRate = conf.expected_sample_rate;

  // Initialize using registry-based factory
  std::unique_ptr<DSP> out =
    nam::factory::FactoryRegistry::instance().create(architecture, config, weights, expectedSampleRate);

  if (loudness.have)
  {
    out->SetLoudness(loudness.value);
  }
  if (inputLevel.have)
  {
    out->SetInputLevel(inputLevel.value);
  }
  if (outputLevel.have)
  {
    out->SetOutputLevel(outputLevel.value);
  }

  return out;
}

double get_sample_rate_from_nam_file(const nlohmann::json& j)
{
  const auto sample_rate = j.find("sample_rate");
  if (sample_rate == j.end() || sample_rate->is_null())
    return NAM_UNKNOWN_EXPECTED_SAMPLE_RATE;
  if (!sample_rate->is_number())
    throw std::invalid_argument("Model sample_rate must be numeric or null");
  return sample_rate->get<double>();
}

}; // namespace nam
