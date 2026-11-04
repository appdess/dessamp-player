#pragma once
#include <algorithm>
#include <array>
#include <json.hpp>
#include <stdexcept>
#include <vector>

// Older factory profiles end before the independent pedal trims, which default to 0 dB.
template <size_t N>
std::array<double, N> FactoryTestParameters(const nlohmann::json& entry)
{
  const auto values = entry.at("parameters").get<std::vector<double>>();
  if (values.size() != 29 && values.size() != N) throw std::runtime_error("Unexpected factory parameter count");
  std::array<double, N> params{};
  std::copy(values.begin(), values.end(), params.begin());
  return params;
}
