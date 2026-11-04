#pragma once

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include "json.hpp"

namespace nam
{
namespace model_validation
{
inline constexpr std::size_t kMaxDenseLayerValues = 16U << 20;
inline constexpr std::size_t kMaxConfigArrayValues = 1U << 20;
inline constexpr std::size_t kMaxModelWeightValues = 4U << 20;
inline constexpr std::size_t kMaxModelGraphModels = 64U;
inline constexpr std::size_t kMaxModelGraphJsonNodes = 1U << 18;
inline constexpr std::size_t kMaxModelGraphTextBytes = 8U << 20;
inline constexpr std::size_t kMaxActivationChannelValues = 4096U;
inline constexpr std::size_t kMaxPrewarmComputeOperations = std::size_t{4} << 30;

inline std::size_t checked_add(const std::size_t lhs, const std::size_t rhs, const char* context)
{
  if (rhs > std::numeric_limits<std::size_t>::max() - lhs)
    throw std::invalid_argument(std::string(context) + " size overflows");
  return lhs + rhs;
}

inline std::size_t checked_multiply(const std::size_t lhs, const std::size_t rhs, const char* context)
{
  if (lhs != 0 && rhs > std::numeric_limits<std::size_t>::max() / lhs)
    throw std::invalid_argument(std::string(context) + " size overflows");
  return lhs * rhs;
}

inline std::size_t positive_dimension(const int value, const char* name)
{
  if (value <= 0)
    throw std::invalid_argument(std::string(name) + " must be positive");
  return static_cast<std::size_t>(value);
}

inline std::size_t nonnegative_dimension(const int value, const char* name)
{
  if (value < 0)
    throw std::invalid_argument(std::string(name) + " must not be negative");
  return static_cast<std::size_t>(value);
}

inline void require_grouped_channels(const int in_channels, const int out_channels, const int groups,
                                     const char* context)
{
  positive_dimension(in_channels, "in_channels");
  positive_dimension(out_channels, "out_channels");
  positive_dimension(groups, "groups");
  if (in_channels % groups != 0 || out_channels % groups != 0)
    throw std::runtime_error(std::string(context) + " channels must be divisible by groups");
}

inline void require_exact_weight_count(const char* architecture, const std::size_t expected,
                                       const std::size_t actual)
{
  if (actual != expected)
  {
    throw std::runtime_error(std::string(architecture) + " weight count mismatch: expected "
                             + std::to_string(expected) + ", received " + std::to_string(actual));
  }
}

inline void require_dense_layer_limit(const std::size_t values, const char* context)
{
  if (values > kMaxDenseLayerValues)
    throw std::invalid_argument(std::string(context) + " dense storage exceeds the supported resource limit");
}

inline void require_model_weight_limit(const std::size_t values)
{
  if (values > kMaxModelWeightValues)
    throw std::invalid_argument("Model weights exceed the supported element count");
}

inline int json_integer(const nlohmann::json& value, const std::string& name)
{
  if (value.is_number_unsigned())
  {
    const auto number = value.get<std::uint64_t>();
    if (number > static_cast<std::uint64_t>(std::numeric_limits<int>::max()))
      throw std::invalid_argument(name + " is outside the supported integer range");
    return static_cast<int>(number);
  }
  if (value.is_number_integer())
  {
    const auto number = value.get<std::int64_t>();
    if (number < static_cast<std::int64_t>(std::numeric_limits<int>::min())
        || number > static_cast<std::int64_t>(std::numeric_limits<int>::max()))
    {
      throw std::invalid_argument(name + " is outside the supported integer range");
    }
    return static_cast<int>(number);
  }
  throw std::invalid_argument(name + " must be an integer");
}

inline int json_integer_at(const nlohmann::json& object, const char* key)
{
  return json_integer(object.at(key), key);
}

inline int json_integer_value(const nlohmann::json& object, const char* key, const int default_value)
{
  const auto value = object.find(key);
  return value == object.end() ? default_value : json_integer(*value, key);
}

inline std::vector<int> json_integer_vector_at(const nlohmann::json& object, const char* key)
{
  const auto& values = object.at(key);
  if (!values.is_array())
    throw std::invalid_argument(std::string(key) + " must be an array of integers");
  if (values.size() > kMaxConfigArrayValues)
    throw std::invalid_argument(std::string(key) + " exceeds the supported element count");

  std::vector<int> result;
  result.reserve(values.size());
  for (std::size_t index = 0; index < values.size(); ++index)
    result.push_back(json_integer(values.at(index), std::string(key) + "[" + std::to_string(index) + "]"));
  return result;
}

inline float json_float(const nlohmann::json& value, const std::string& name)
{
  if (!value.is_number())
    throw std::invalid_argument(name + " must be numeric");
  const double number = value.get<double>();
  if (!std::isfinite(number) || std::abs(number) > static_cast<double>(std::numeric_limits<float>::max()))
    throw std::invalid_argument(name + " must be finite and representable as a float");
  return static_cast<float>(number);
}

inline float json_float_at(const nlohmann::json& object, const char* key)
{
  return json_float(object.at(key), key);
}

inline float json_float_value(const nlohmann::json& object, const char* key, const float default_value)
{
  const auto value = object.find(key);
  return value == object.end() ? default_value : json_float(*value, key);
}
} // namespace model_validation
} // namespace nam
