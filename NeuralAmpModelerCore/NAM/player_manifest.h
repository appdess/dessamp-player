#pragma once

#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "dsp.h"
#include "get_dsp.h"
#include "json.hpp"
#include "player_morph.h"
#include "slimmable.h"

namespace nam::player
{
inline constexpr std::uintmax_t kMaxManifestBytes = 1024U * 1024U;
inline constexpr std::uintmax_t kMaxDessArtifactBytes = 64U * 1024U * 1024U;
inline constexpr int kMaxDAWConditionControls = 8;

struct Control
{
  std::string id;
  std::string label;
  std::string unit;
  // Capture projects already persist this optional field. Older manifests did
  // not, so linear remains the wire-compatible default.
  std::string taper = "linear";
  double minimum = 0.0;
  double maximum = 1.0;
  double default_value = 0.5;
  double display_minimum = std::numeric_limits<double>::quiet_NaN();
  double display_maximum = std::numeric_limits<double>::quiet_NaN();
  int model_parameter_index = -1;

  double ModelValueFromNormalized(const double normalized) const
  {
    return minimum + std::clamp(normalized, 0.0, 1.0) * (maximum - minimum);
  }

  double NormalizedDefault() const { return std::clamp((default_value - minimum) / (maximum - minimum), 0.0, 1.0); }

  bool HasDisplayRange() const
  {
    return std::isfinite(display_minimum) && std::isfinite(display_maximum) && display_minimum < display_maximum;
  }

  double DisplayMinimum() const
  {
    if (HasDisplayRange())
      return display_minimum;
    return unit == "%" && minimum >= 0.0 && maximum <= 1.0 ? minimum * 100.0 : minimum;
  }

  double DisplayMaximum() const
  {
    if (HasDisplayRange())
      return display_maximum;
    return unit == "%" && minimum >= 0.0 && maximum <= 1.0 ? maximum * 100.0 : maximum;
  }

  double DisplayValueFromNormalized(const double normalized) const
  {
    return DisplayMinimum() + std::clamp(normalized, 0.0, 1.0) * (DisplayMaximum() - DisplayMinimum());
  }

  bool IsSwitch() const { return taper == "switch"; }
};

struct FixedControl
{
  std::string id;
  std::string label;
  std::string taper = "switch";
  double value = 0.0;
  double minimum = 0.0;
  double maximum = 1.0;
  std::string unit;

  bool IsSwitch() const { return taper == "switch"; }
  bool IsLinear() const { return taper == "linear"; }
  bool IsOn() const { return value > 0.5; }
};

struct RuntimeQuality
{
  std::string mode = "fixed";
  double default_size = 1.0;
  std::vector<double> breakpoints;
};

struct Morph
{
  bool enabled = false;
  std::filesystem::path model_path;
  std::string model_sha256;
  double sample_rate = 48000.0;
  RuntimeQuality runtime_quality;
  MorphSettings settings;
  std::string from_label = "GER";
  std::string to_label = "BRIT";
};

struct Manifest
{
  std::filesystem::path manifest_path;
  std::filesystem::path model_path;
  std::string artifact_id;
  std::string name;
  // Presentation selects one of the player's own original, compiled shells.
  // It never names an arbitrary file, so a package cannot make the UI read
  // unverified image bytes or escape the application bundle.
  std::string head_shell_style = "studio-black-v1";
  // Optional bounded tokens let the player group separately packaged voices
  // into one product experience. They are presentation/catalog metadata only:
  // the selected package path remains the persisted and hash-verified state.
  std::string product_family;
  std::string product_variant;
  std::string model_sha256;
  std::string parameter_mode;
  int global_condition_size = 0;
  double sample_rate = 48000.0;
  std::vector<Control> controls;
  std::vector<FixedControl> fixed_controls;
  RuntimeQuality runtime_quality;
  Morph morph;

  std::vector<double> DefaultModelParameters() const
  {
    std::vector<double> result(controls.size(), 0.5);
    for (const auto& control : controls)
      result.at(static_cast<std::size_t>(control.model_parameter_index)) = control.default_value;
    return result;
  }

  int PlayerParameterCount() const { return static_cast<int>(controls.size()); }
};

namespace detail
{
inline bool IsWithin(const std::filesystem::path& base, const std::filesystem::path& candidate)
{
  const auto mismatch = std::mismatch(base.begin(), base.end(), candidate.begin(), candidate.end());
  return mismatch.first == base.end();
}

inline std::string RequiredString(const nlohmann::json& object, const char* key, const std::size_t max_length)
{
  const auto it = object.find(key);
  if (it == object.end() || !it->is_string())
    throw std::invalid_argument(std::string("Manifest ") + key + " must be a string");
  const auto value = it->get<std::string>();
  if (value.empty() || value.size() > max_length)
    throw std::invalid_argument(std::string("Manifest ") + key + " length is invalid");
  return value;
}

inline double RequiredFiniteNumber(const nlohmann::json& object, const char* key)
{
  const auto it = object.find(key);
  if (it == object.end() || !it->is_number())
    throw std::invalid_argument(std::string("Manifest ") + key + " must be numeric");
  const double value = it->get<double>();
  if (!std::isfinite(value))
    throw std::invalid_argument(std::string("Manifest ") + key + " must be finite");
  return value;
}

inline RuntimeQuality ParseRuntimeQuality(const nlohmann::json& model)
{
  RuntimeQuality quality;
  const auto quality_it = model.find("runtime_quality");
  if (quality_it == model.end())
    return quality;
  if (!quality_it->is_object())
    throw std::invalid_argument("Manifest runtime_quality must be an object");
  quality.mode = RequiredString(*quality_it, "mode", 32U);
  if (quality.mode != "fixed" && quality.mode != "slimmable")
    throw std::invalid_argument("Manifest runtime_quality mode is unsupported");
  const auto default_size_it = quality_it->find("default_size");
  if (default_size_it != quality_it->end())
  {
    if (!default_size_it->is_number())
      throw std::invalid_argument("Manifest runtime_quality default_size must be numeric");
    quality.default_size = default_size_it->get<double>();
    if (!std::isfinite(quality.default_size) || quality.default_size < 0.0 || quality.default_size > 1.0)
      throw std::invalid_argument("Manifest runtime_quality default_size must be in [0, 1]");
  }
  const auto breakpoints_it = quality_it->find("breakpoints");
  if (breakpoints_it != quality_it->end())
  {
    if (!breakpoints_it->is_array() || breakpoints_it->size() > 64U)
      throw std::invalid_argument("Manifest runtime_quality breakpoints must be a bounded array");
    for (const auto& breakpoint : *breakpoints_it)
    {
      if (!breakpoint.is_number())
        throw std::invalid_argument("Manifest runtime_quality breakpoints must be numeric");
      const double value = breakpoint.get<double>();
      if (!std::isfinite(value))
        throw std::invalid_argument("Manifest runtime_quality breakpoints must be finite");
      quality.breakpoints.push_back(value);
    }
  }
  return quality;
}

inline std::uint32_t RotateRight(const std::uint32_t value, const std::uint32_t amount)
{
  return (value >> amount) | (value << (32U - amount));
}

class Sha256
{
public:
  void Update(const std::uint8_t* bytes, const std::size_t size)
  {
    for (std::size_t index = 0; index < size; ++index)
    {
      mData[mDataLength++] = bytes[index];
      if (mDataLength == mData.size())
      {
        Transform();
        mBitLength += 512U;
        mDataLength = 0U;
      }
    }
  }

  std::array<std::uint8_t, 32> Finalize()
  {
    std::size_t index = mDataLength;
    mData[index++] = 0x80U;
    if (index > 56U)
    {
      std::fill(mData.begin() + static_cast<std::ptrdiff_t>(index), mData.end(), 0U);
      Transform();
      index = 0U;
    }
    std::fill(mData.begin() + static_cast<std::ptrdiff_t>(index), mData.begin() + 56, 0U);

    mBitLength += static_cast<std::uint64_t>(mDataLength) * 8U;
    for (std::size_t byte = 0; byte < 8U; ++byte)
      mData[63U - byte] = static_cast<std::uint8_t>(mBitLength >> (byte * 8U));
    Transform();

    std::array<std::uint8_t, 32> digest{};
    for (std::size_t word = 0; word < mState.size(); ++word)
      for (std::size_t byte = 0; byte < 4U; ++byte)
        digest[word * 4U + byte] = static_cast<std::uint8_t>(mState[word] >> (24U - byte * 8U));
    return digest;
  }

private:
  void Transform()
  {
    static constexpr std::array<std::uint32_t, 64> constants{
      0x428a2f98U, 0x71374491U, 0xb5c0fbcfU, 0xe9b5dba5U, 0x3956c25bU, 0x59f111f1U, 0x923f82a4U, 0xab1c5ed5U,
      0xd807aa98U, 0x12835b01U, 0x243185beU, 0x550c7dc3U, 0x72be5d74U, 0x80deb1feU, 0x9bdc06a7U, 0xc19bf174U,
      0xe49b69c1U, 0xefbe4786U, 0x0fc19dc6U, 0x240ca1ccU, 0x2de92c6fU, 0x4a7484aaU, 0x5cb0a9dcU, 0x76f988daU,
      0x983e5152U, 0xa831c66dU, 0xb00327c8U, 0xbf597fc7U, 0xc6e00bf3U, 0xd5a79147U, 0x06ca6351U, 0x14292967U,
      0x27b70a85U, 0x2e1b2138U, 0x4d2c6dfcU, 0x53380d13U, 0x650a7354U, 0x766a0abbU, 0x81c2c92eU, 0x92722c85U,
      0xa2bfe8a1U, 0xa81a664bU, 0xc24b8b70U, 0xc76c51a3U, 0xd192e819U, 0xd6990624U, 0xf40e3585U, 0x106aa070U,
      0x19a4c116U, 0x1e376c08U, 0x2748774cU, 0x34b0bcb5U, 0x391c0cb3U, 0x4ed8aa4aU, 0x5b9cca4fU, 0x682e6ff3U,
      0x748f82eeU, 0x78a5636fU, 0x84c87814U, 0x8cc70208U, 0x90befffaU, 0xa4506cebU, 0xbef9a3f7U, 0xc67178f2U,
    };

    std::array<std::uint32_t, 64> words{};
    for (std::size_t index = 0; index < 16U; ++index)
    {
      const std::size_t offset = index * 4U;
      words[index] =
        (static_cast<std::uint32_t>(mData[offset]) << 24U) | (static_cast<std::uint32_t>(mData[offset + 1U]) << 16U)
        | (static_cast<std::uint32_t>(mData[offset + 2U]) << 8U) | static_cast<std::uint32_t>(mData[offset + 3U]);
    }
    for (std::size_t index = 16U; index < words.size(); ++index)
    {
      const std::uint32_t s0 =
        RotateRight(words[index - 15U], 7U) ^ RotateRight(words[index - 15U], 18U) ^ (words[index - 15U] >> 3U);
      const std::uint32_t s1 =
        RotateRight(words[index - 2U], 17U) ^ RotateRight(words[index - 2U], 19U) ^ (words[index - 2U] >> 10U);
      words[index] = words[index - 16U] + s0 + words[index - 7U] + s1;
    }

    std::uint32_t a = mState[0];
    std::uint32_t b = mState[1];
    std::uint32_t c = mState[2];
    std::uint32_t d = mState[3];
    std::uint32_t e = mState[4];
    std::uint32_t f = mState[5];
    std::uint32_t g = mState[6];
    std::uint32_t h = mState[7];

    for (std::size_t index = 0; index < words.size(); ++index)
    {
      const std::uint32_t sum1 = RotateRight(e, 6U) ^ RotateRight(e, 11U) ^ RotateRight(e, 25U);
      const std::uint32_t choice = (e & f) ^ (~e & g);
      const std::uint32_t temp1 = h + sum1 + choice + constants[index] + words[index];
      const std::uint32_t sum0 = RotateRight(a, 2U) ^ RotateRight(a, 13U) ^ RotateRight(a, 22U);
      const std::uint32_t majority = (a & b) ^ (a & c) ^ (b & c);
      const std::uint32_t temp2 = sum0 + majority;

      h = g;
      g = f;
      f = e;
      e = d + temp1;
      d = c;
      c = b;
      b = a;
      a = temp1 + temp2;
    }

    mState[0] += a;
    mState[1] += b;
    mState[2] += c;
    mState[3] += d;
    mState[4] += e;
    mState[5] += f;
    mState[6] += g;
    mState[7] += h;
  }

  std::array<std::uint8_t, 64> mData{};
  std::size_t mDataLength = 0U;
  std::uint64_t mBitLength = 0U;
  std::array<std::uint32_t, 8> mState{
    0x6a09e667U, 0xbb67ae85U, 0x3c6ef372U, 0xa54ff53aU, 0x510e527fU, 0x9b05688cU, 0x1f83d9abU, 0x5be0cd19U,
  };
};

inline std::string Sha256File(const std::filesystem::path& path)
{
  std::ifstream file(path, std::ios::binary);
  if (!file.is_open())
    throw std::runtime_error("Unable to open model for SHA-256 verification");
  Sha256 hash;
  std::array<char, 16384> buffer{};
  while (file)
  {
    file.read(buffer.data(), static_cast<std::streamsize>(buffer.size()));
    const auto count = file.gcount();
    if (count > 0)
      hash.Update(reinterpret_cast<const std::uint8_t*>(buffer.data()), static_cast<std::size_t>(count));
  }
  if (!file.eof())
    throw std::runtime_error("Unable to finish model SHA-256 verification");

  const auto digest = hash.Finalize();
  std::ostringstream result;
  result << std::hex << std::setfill('0');
  for (const auto byte : digest)
    result << std::setw(2) << static_cast<unsigned int>(byte);
  return result.str();
}

inline std::array<std::uint8_t, 32> Sha256Bytes(const std::uint8_t* bytes, const std::size_t size)
{
  Sha256 hash;
  hash.Update(bytes, size);
  return hash.Finalize();
}

inline std::string DigestHex(const std::array<std::uint8_t, 32>& digest)
{
  std::ostringstream result;
  result << std::hex << std::setfill('0');
  for (const auto byte : digest)
    result << std::setw(2) << static_cast<unsigned int>(byte);
  return result.str();
}

inline std::uint32_t ReadLittleEndian32(const std::uint8_t* bytes)
{
  return static_cast<std::uint32_t>(bytes[0]) | (static_cast<std::uint32_t>(bytes[1]) << 8U)
         | (static_cast<std::uint32_t>(bytes[2]) << 16U) | (static_cast<std::uint32_t>(bytes[3]) << 24U);
}

inline std::uint64_t ReadLittleEndian64(const std::uint8_t* bytes)
{
  std::uint64_t value = 0U;
  for (std::uint32_t index = 0U; index < 8U; ++index)
    value |= static_cast<std::uint64_t>(bytes[index]) << (index * 8U);
  return value;
}

inline void WriteVerifiedCacheFile(const std::filesystem::path& destination, const std::uint8_t* bytes,
                                   const std::size_t size, const std::string& expected_sha256)
{
  if (std::filesystem::is_regular_file(destination) && Sha256File(destination) == expected_sha256)
    return;
  std::random_device random;
  auto temporary = destination;
  temporary += ".tmp-" + std::to_string(random()) + "-" + std::to_string(random());
  {
    std::ofstream output(temporary, std::ios::binary | std::ios::trunc);
    if (!output.is_open())
      throw std::runtime_error("Unable to create the DessAmp package cache file");
    output.write(reinterpret_cast<const char*>(bytes), static_cast<std::streamsize>(size));
    output.flush();
    if (!output.good())
    {
      output.close();
      std::error_code remove_error;
      std::filesystem::remove(temporary, remove_error);
      throw std::runtime_error("Unable to finish the DessAmp package cache file");
    }
  }
  std::error_code rename_error;
  std::filesystem::rename(temporary, destination, rename_error);
  if (rename_error)
  {
    std::error_code remove_error;
    std::filesystem::remove(temporary, remove_error);
    if (!std::filesystem::is_regular_file(destination) || Sha256File(destination) != expected_sha256)
      throw std::runtime_error("Unable to publish the DessAmp package cache file");
  }
  if (Sha256File(destination) != expected_sha256)
    throw std::runtime_error("DessAmp package cache verification failed");
}

inline bool IsLowerHexSha256(const std::string& value)
{
  return value.size() == 64U && std::all_of(value.begin(), value.end(), [](const char character) {
           return (character >= '0' && character <= '9') || (character >= 'a' && character <= 'f');
         });
}

inline std::unique_ptr<nam::DSP> LoadVerifiedModel(const std::filesystem::path& path,
                                                   const std::string& expected_sha256)
{
  if (!IsLowerHexSha256(expected_sha256))
    throw std::invalid_argument("Verified model SHA-256 is invalid");
  if (!std::filesystem::is_regular_file(path))
    throw std::invalid_argument("Verified model path is not a regular file");
  const auto file_size = std::filesystem::file_size(path);
  if (file_size == 0U || file_size > nam::kMaxSerializedModelBytes)
    throw std::invalid_argument("Verified model is empty or exceeds 64 MiB");

  std::ifstream input(path, std::ios::binary);
  if (!input.is_open())
    throw std::runtime_error("Unable to open verified model");
  std::vector<std::uint8_t> bytes(static_cast<std::size_t>(file_size));
  input.read(reinterpret_cast<char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
  if (input.gcount() != static_cast<std::streamsize>(bytes.size()) || input.peek() != std::char_traits<char>::eof())
    throw std::runtime_error("Verified model changed while it was being read");
  if (DigestHex(Sha256Bytes(bytes.data(), bytes.size())) != expected_sha256)
    throw std::invalid_argument("Verified model SHA-256 does not match the manifest");

  // Parse the exact bytes that were hashed. Reopening the path here would
  // reintroduce a check/use race for mutable package-cache or loose-model
  // files.
  return nam::get_dsp(nlohmann::json::parse(bytes.begin(), bytes.end()));
}

inline bool IsPresentationToken(const std::string& value)
{
  return !value.empty() && value.size() <= 64U
         && std::all_of(value.begin(), value.end(), [](const unsigned char character) {
              return (character >= 'a' && character <= 'z') || (character >= '0' && character <= '9')
                     || character == '-';
            });
}
} // namespace detail

inline Manifest LoadManifest(const std::filesystem::path& requested_manifest_path,
                             const int max_condition_controls = kMaxDAWConditionControls,
                             const std::string& expected_manifest_sha256 = {});

inline std::vector<std::uint8_t> ReadDessPackageBytes(const std::filesystem::path& package_path)
{
  static constexpr std::size_t minimum_header_size = 96U;
  const auto package_size = std::filesystem::file_size(package_path);
  if (package_size <= minimum_header_size || package_size > kMaxDessArtifactBytes)
    throw std::invalid_argument("DessAmp package is empty or exceeds 64 MiB");
  std::ifstream input(package_path, std::ios::binary);
  if (!input.is_open())
    throw std::runtime_error("Unable to open DessAmp package");
  std::vector<std::uint8_t> bytes(static_cast<std::size_t>(package_size));
  input.read(reinterpret_cast<char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
  if (!input || input.gcount() != static_cast<std::streamsize>(bytes.size()))
    throw std::runtime_error("Unable to read the complete DessAmp package");
  return bytes;
}

inline Manifest ParseDessPackage(const std::filesystem::path& package_path, const std::vector<std::uint8_t>& bytes,
                                 const int max_condition_controls)
{
  static constexpr std::array<std::uint8_t, 8> magic{'D', 'E', 'S', 'S', 'A', 'M', 'P', 0U};
  static constexpr std::size_t v1_header_size = 96U;
  static constexpr std::size_t v2_header_size = 136U;
  if (bytes.size() <= v1_header_size || bytes.size() > kMaxDessArtifactBytes)
    throw std::invalid_argument("Invalid DessAmp package size");
  if (!std::equal(magic.begin(), magic.end(), bytes.begin()))
    throw std::invalid_argument("DessAmp package magic is invalid");
  const auto version = detail::ReadLittleEndian32(bytes.data() + 8U);
  if ((version != 1U && version != 2U) || detail::ReadLittleEndian32(bytes.data() + 12U) != 0U)
    throw std::invalid_argument("Unsupported DessAmp package version or flags");
  const std::size_t header_size = version == 1U ? v1_header_size : v2_header_size;
  if (bytes.size() <= header_size)
    throw std::invalid_argument("DessAmp package header is truncated");
  const auto manifest_size = detail::ReadLittleEndian64(bytes.data() + 16U);
  const auto model_size = detail::ReadLittleEndian64(bytes.data() + 24U);
  const auto morph_model_size = version == 2U ? detail::ReadLittleEndian64(bytes.data() + 32U) : 0U;
  if (manifest_size == 0U || manifest_size > kMaxManifestBytes || model_size == 0U || model_size > kMaxDessArtifactBytes
      || (version == 2U && morph_model_size == 0U) || morph_model_size > kMaxDessArtifactBytes
      || header_size + manifest_size + model_size + morph_model_size != bytes.size())
    throw std::invalid_argument("DessAmp package section sizes or trailing data are invalid");

  const auto manifest_digest = detail::Sha256Bytes(bytes.data() + header_size, static_cast<std::size_t>(manifest_size));
  const auto model_offset = header_size + static_cast<std::size_t>(manifest_size);
  const auto model_digest = detail::Sha256Bytes(bytes.data() + model_offset, static_cast<std::size_t>(model_size));
  const auto morph_model_offset = model_offset + static_cast<std::size_t>(model_size);
  const auto morph_model_digest =
    version == 2U ? detail::Sha256Bytes(bytes.data() + morph_model_offset, static_cast<std::size_t>(morph_model_size))
                  : std::array<std::uint8_t, 32>{};
  const std::size_t manifest_digest_offset = version == 1U ? 32U : 40U;
  const std::size_t model_digest_offset = version == 1U ? 64U : 72U;
  if (!std::equal(manifest_digest.begin(), manifest_digest.end(), bytes.begin() + manifest_digest_offset))
    throw std::invalid_argument("DessAmp package manifest hash is invalid");
  if (!std::equal(model_digest.begin(), model_digest.end(), bytes.begin() + model_digest_offset))
    throw std::invalid_argument("DessAmp package model hash is invalid");
  if (version == 2U && !std::equal(morph_model_digest.begin(), morph_model_digest.end(), bytes.begin() + 104U))
    throw std::invalid_argument("DessAmp package morph model hash is invalid");

  const auto manifest_sha256 = detail::DigestHex(manifest_digest);
  const auto model_sha256 = detail::DigestHex(model_digest);
  const auto morph_model_sha256 = detail::DigestHex(morph_model_digest);
  const auto cache_directory = std::filesystem::temp_directory_path()
                               / (version == 1U ? "dessmetal-player-artifacts-v1" : "dessamp-player-artifacts-v2")
                               / manifest_sha256;
  std::filesystem::create_directories(cache_directory);
  const auto cached_manifest = cache_directory / "player-manifest.json";
  const auto cached_model = cache_directory / "model.nam";
  const auto cached_morph_model = cache_directory / "morph-model.nam";
  detail::WriteVerifiedCacheFile(
    cached_manifest, bytes.data() + header_size, static_cast<std::size_t>(manifest_size), manifest_sha256);
  detail::WriteVerifiedCacheFile(
    cached_model, bytes.data() + model_offset, static_cast<std::size_t>(model_size), model_sha256);
  if (version == 2U)
    detail::WriteVerifiedCacheFile(cached_morph_model, bytes.data() + morph_model_offset,
                                   static_cast<std::size_t>(morph_model_size), morph_model_sha256);
  Manifest manifest = LoadManifest(cached_manifest, max_condition_controls, manifest_sha256);
  if (manifest.model_path != std::filesystem::weakly_canonical(cached_model))
    throw std::invalid_argument("DessAmp package manifest does not select its embedded model.nam");
  if (version == 2U
      && (!manifest.morph.enabled
          || manifest.morph.model_path != std::filesystem::weakly_canonical(cached_morph_model)))
    throw std::invalid_argument("DessAmp morph package manifest does not select its embedded morph-model.nam");
  manifest.manifest_path = package_path;
  return manifest;
}

inline Manifest LoadDessPackage(const std::filesystem::path& requested_package_path,
                                const int max_condition_controls = kMaxDAWConditionControls)
{
  const auto path = std::filesystem::weakly_canonical(requested_package_path);
  return ParseDessPackage(path, ReadDessPackageBytes(path), max_condition_controls);
}

inline Manifest LoadManifest(const std::filesystem::path& requested_manifest_path, const int max_condition_controls,
                             const std::string& expected_manifest_sha256)
{
  if (!std::filesystem::is_regular_file(requested_manifest_path))
    throw std::invalid_argument("Manifest path is not a regular file");
  std::string extension = requested_manifest_path.extension().string();
  std::transform(extension.begin(), extension.end(), extension.begin(),
                 [](const unsigned char character) { return static_cast<char>(std::tolower(character)); });
  if (extension == ".dess")
  {
    if (!expected_manifest_sha256.empty())
      throw std::invalid_argument("Nested DessAmp packages cannot be hash-bound as manifests");
    return LoadDessPackage(requested_manifest_path, max_condition_controls);
  }

  const auto manifest_path = std::filesystem::weakly_canonical(requested_manifest_path);
  const auto manifest_size = std::filesystem::file_size(manifest_path);
  if (manifest_size == 0U || manifest_size > kMaxManifestBytes)
    throw std::invalid_argument("Manifest is empty or exceeds the 1 MiB player limit");

  std::ifstream manifest_file(manifest_path, std::ios::binary);
  if (!manifest_file.is_open())
    throw std::runtime_error("Unable to open player manifest");
  std::vector<std::uint8_t> manifest_bytes(static_cast<std::size_t>(manifest_size));
  manifest_file.read(reinterpret_cast<char*>(manifest_bytes.data()),
                     static_cast<std::streamsize>(manifest_bytes.size()));
  if (manifest_file.gcount() != static_cast<std::streamsize>(manifest_bytes.size())
      || manifest_file.peek() != std::char_traits<char>::eof())
    throw std::runtime_error("Player manifest changed while it was being read");
  if (!expected_manifest_sha256.empty()
      && (!detail::IsLowerHexSha256(expected_manifest_sha256)
          || detail::DigestHex(detail::Sha256Bytes(manifest_bytes.data(), manifest_bytes.size()))
               != expected_manifest_sha256))
    throw std::invalid_argument("Player manifest SHA-256 does not match its package");

  nlohmann::json json = nlohmann::json::parse(manifest_bytes.begin(), manifest_bytes.end());
  if (!json.is_object() || json.value("schema_version", 0) != 1)
    throw std::invalid_argument("Unsupported player manifest schema_version");

  Manifest manifest;
  manifest.manifest_path = manifest_path;
  manifest.artifact_id = detail::RequiredString(json, "artifact_id", 128U);
  manifest.name = detail::RequiredString(json, "name", 256U);

  const auto presentation_it = json.find("presentation");
  if (presentation_it != json.end())
  {
    if (!presentation_it->is_object())
      throw std::invalid_argument("Manifest presentation must be an object");
    const auto head_shell_it = presentation_it->find("head_shell");
    if (head_shell_it != presentation_it->end())
    {
      manifest.head_shell_style = detail::RequiredString(*presentation_it, "head_shell", 64U);
      if (!detail::IsPresentationToken(manifest.head_shell_style))
        throw std::invalid_argument("Manifest presentation head_shell token is invalid");
    }
    const auto product_family_it = presentation_it->find("product_family");
    if (product_family_it != presentation_it->end())
    {
      manifest.product_family = detail::RequiredString(*presentation_it, "product_family", 64U);
      if (!detail::IsPresentationToken(manifest.product_family))
        throw std::invalid_argument("Manifest presentation product_family token is invalid");
    }
    const auto product_variant_it = presentation_it->find("product_variant");
    if (product_variant_it != presentation_it->end())
    {
      manifest.product_variant = detail::RequiredString(*presentation_it, "product_variant", 64U);
      if (!detail::IsPresentationToken(manifest.product_variant))
        throw std::invalid_argument("Manifest presentation product_variant token is invalid");
      if (manifest.product_family.empty())
        throw std::invalid_argument("Manifest presentation product_variant requires product_family");
    }
  }

  const auto model_it = json.find("model");
  if (model_it == json.end() || !model_it->is_object())
    throw std::invalid_argument("Manifest model must be an object");
  const auto& model = *model_it;
  if (model.value("format", "") != "nam")
    throw std::invalid_argument("Manifest model format must be 'nam'");

  const std::string relative_model_string = detail::RequiredString(model, "path", 512U);
  const std::filesystem::path relative_model_path(relative_model_string);
  if (relative_model_path.is_absolute())
    throw std::invalid_argument("Manifest model path must be relative");
  const auto artifact_path = std::filesystem::weakly_canonical(manifest_path.parent_path());
  manifest.model_path = std::filesystem::weakly_canonical(artifact_path / relative_model_path);
  if (!detail::IsWithin(artifact_path, manifest.model_path))
    throw std::invalid_argument("Manifest model path escapes the artifact directory");
  if (!std::filesystem::is_regular_file(manifest.model_path))
    throw std::invalid_argument("Manifest model path is not a regular file");

  manifest.model_sha256 = detail::RequiredString(model, "sha256", 64U);
  if (!detail::IsLowerHexSha256(manifest.model_sha256))
    throw std::invalid_argument("Manifest model sha256 must be 64 lowercase hexadecimal characters");
  const std::string observed_sha256 = detail::Sha256File(manifest.model_path);
  if (observed_sha256 != manifest.model_sha256)
    throw std::invalid_argument("Manifest model SHA-256 does not match model.nam");

  const auto condition_it = model.find("global_condition_size");
  if (condition_it == model.end() || !condition_it->is_number_integer())
    throw std::invalid_argument("Manifest global_condition_size must be an integer");
  manifest.global_condition_size = condition_it->get<int>();
  if (manifest.global_condition_size < 0 || manifest.global_condition_size > max_condition_controls)
    throw std::invalid_argument("Manifest global_condition_size exceeds this player's automation slots");
  manifest.parameter_mode = detail::RequiredString(model, "parameter_mode", 32U);
  if ((manifest.global_condition_size == 0 && manifest.parameter_mode != "snapshot")
      || (manifest.global_condition_size > 0 && manifest.parameter_mode != "global_condition"))
    throw std::invalid_argument("Manifest parameter_mode disagrees with global_condition_size");

  const auto sample_rate_it = model.find("sample_rate");
  if (sample_rate_it != model.end())
  {
    if (!sample_rate_it->is_number())
      throw std::invalid_argument("Manifest sample_rate must be numeric");
    manifest.sample_rate = sample_rate_it->get<double>();
    if (!std::isfinite(manifest.sample_rate) || manifest.sample_rate < 8000.0 || manifest.sample_rate > 768000.0)
      throw std::invalid_argument("Manifest sample_rate is outside the supported range");
  }

  nlohmann::json morph_control_json;
  const auto morph_it = json.find("morph");
  if (morph_it != json.end())
  {
    if (!morph_it->is_object())
      throw std::invalid_argument("Manifest morph must be an object");
    if (manifest.global_condition_size >= max_condition_controls)
      throw std::invalid_argument("Manifest leaves no automation slot for Morph");
    const auto& morph_json = *morph_it;
    if (morph_json.value("mode", "") != "dual_model_linear_v1")
      throw std::invalid_argument("Manifest morph mode is unsupported");

    const auto morph_model_it = morph_json.find("model");
    if (morph_model_it == morph_json.end() || !morph_model_it->is_object())
      throw std::invalid_argument("Manifest morph model must be an object");
    const auto& morph_model = *morph_model_it;
    if (morph_model.value("format", "") != "nam")
      throw std::invalid_argument("Manifest morph model format must be 'nam'");
    const std::string relative_morph_string = detail::RequiredString(morph_model, "path", 512U);
    const std::filesystem::path relative_morph_path(relative_morph_string);
    if (relative_morph_path.is_absolute())
      throw std::invalid_argument("Manifest morph model path must be relative");
    manifest.morph.model_path = std::filesystem::weakly_canonical(artifact_path / relative_morph_path);
    if (!detail::IsWithin(artifact_path, manifest.morph.model_path))
      throw std::invalid_argument("Manifest morph model path escapes the artifact directory");
    if (!std::filesystem::is_regular_file(manifest.morph.model_path))
      throw std::invalid_argument("Manifest morph model path is not a regular file");
    manifest.morph.model_sha256 = detail::RequiredString(morph_model, "sha256", 64U);
    if (!detail::IsLowerHexSha256(manifest.morph.model_sha256))
      throw std::invalid_argument("Manifest morph model sha256 must be lowercase hexadecimal");
    if (detail::Sha256File(manifest.morph.model_path) != manifest.morph.model_sha256)
      throw std::invalid_argument("Manifest morph model SHA-256 does not match morph-model.nam");
    const auto morph_condition_it = morph_model.find("global_condition_size");
    if (morph_condition_it == morph_model.end() || !morph_condition_it->is_number_integer()
        || morph_condition_it->get<int>() != 0)
      throw std::invalid_argument("Manifest morph model must be an unconditioned snapshot");
    if (detail::RequiredString(morph_model, "parameter_mode", 32U) != "snapshot")
      throw std::invalid_argument("Manifest morph model parameter_mode must be snapshot");
    manifest.morph.sample_rate = detail::RequiredFiniteNumber(morph_model, "sample_rate");
    if (std::abs(manifest.morph.sample_rate - manifest.sample_rate) > 1.0e-6)
      throw std::invalid_argument("Manifest morph model sample_rate must match the primary model");
    manifest.morph.runtime_quality = detail::ParseRuntimeQuality(morph_model);

    const auto alignment_it = morph_json.find("alignment");
    if (alignment_it == morph_json.end() || !alignment_it->is_object())
      throw std::invalid_argument("Manifest morph alignment must be an object");
    const auto delay_it = alignment_it->find("delay_samples");
    const auto polarity_it = alignment_it->find("polarity");
    if (delay_it == alignment_it->end() || !delay_it->is_number_integer())
      throw std::invalid_argument("Manifest morph delay_samples must be an integer");
    if (polarity_it == alignment_it->end() || !polarity_it->is_number_integer())
      throw std::invalid_argument("Manifest morph polarity must be an integer");
    manifest.morph.settings.delay_samples = delay_it->get<int>();
    manifest.morph.settings.polarity = polarity_it->get<int>();
    manifest.morph.settings.level_scale = detail::RequiredFiniteNumber(*alignment_it, "level_scale");
    const double reference_sample_rate = detail::RequiredFiniteNumber(*alignment_it, "reference_sample_rate");
    if (std::abs(reference_sample_rate - manifest.sample_rate) > 1.0e-6)
      throw std::invalid_argument("Manifest morph alignment reference sample rate must match the models");

    const auto blend_it = morph_json.find("blend");
    if (blend_it == morph_json.end() || !blend_it->is_object())
      throw std::invalid_argument("Manifest morph blend must be an object");
    if (detail::RequiredString(*blend_it, "law", 32U) != "linear")
      throw std::invalid_argument("Manifest morph blend law is unsupported");
    manifest.morph.settings.center_gain = detail::RequiredFiniteNumber(*blend_it, "center_gain");
    manifest.morph.settings.smoothing_ms = detail::RequiredFiniteNumber(*blend_it, "smoothing_ms");
    manifest.morph.from_label = detail::RequiredString(morph_json, "from_label", 16U);
    manifest.morph.to_label = detail::RequiredString(morph_json, "to_label", 16U);

    const auto control_it = morph_json.find("control");
    if (control_it == morph_json.end() || !control_it->is_object())
      throw std::invalid_argument("Manifest morph control must be an object");
    morph_control_json = *control_it;
    morph_control_json["model_parameter_index"] = manifest.global_condition_size;
    manifest.morph.enabled = true;
  }

  const auto controls_it = json.find("controls");
  if (controls_it == json.end() || !controls_it->is_array())
    throw std::invalid_argument("Manifest controls must be an array");
  const int player_parameter_count = manifest.global_condition_size + (manifest.morph.enabled ? 1 : 0);
  manifest.controls.resize(static_cast<std::size_t>(player_parameter_count));
  std::vector<bool> assigned(static_cast<std::size_t>(player_parameter_count), false);
  nlohmann::json player_controls = *controls_it;
  if (manifest.morph.enabled)
    player_controls.push_back(morph_control_json);
  for (const auto& control_json : player_controls)
  {
    if (!control_json.is_object())
      throw std::invalid_argument("Manifest controls must contain objects");
    const auto index_it = control_json.find("model_parameter_index");
    if (index_it == control_json.end() || index_it->is_null())
      continue;
    if (!index_it->is_number_integer())
      throw std::invalid_argument("Control model_parameter_index must be an integer or null");
    const int index = index_it->get<int>();
    if (index < 0 || index >= player_parameter_count)
      throw std::invalid_argument("Control model_parameter_index is outside the parameter vector");
    if (assigned[static_cast<std::size_t>(index)])
      throw std::invalid_argument("Control model_parameter_index is duplicated");

    Control control;
    control.id = detail::RequiredString(control_json, "id", 64U);
    control.label = detail::RequiredString(control_json, "label", 64U);
    const auto unit_it = control_json.find("unit");
    if (unit_it != control_json.end())
    {
      if (!unit_it->is_string() || unit_it->get<std::string>().size() > 24U)
        throw std::invalid_argument("Control unit must be a short string");
      control.unit = unit_it->get<std::string>();
    }
    const auto taper_it = control_json.find("taper");
    if (taper_it != control_json.end())
    {
      if (!taper_it->is_string() || taper_it->get<std::string>().size() > 16U)
        throw std::invalid_argument("Control taper must be a short string");
      control.taper = taper_it->get<std::string>();
      if (control.taper != "linear" && control.taper != "log" && control.taper != "switch")
        throw std::invalid_argument("Control taper is unsupported");
    }
    control.minimum = detail::RequiredFiniteNumber(control_json, "minimum");
    control.maximum = detail::RequiredFiniteNumber(control_json, "maximum");
    control.default_value = detail::RequiredFiniteNumber(control_json, "default_value");
    const auto display_minimum_it = control_json.find("display_minimum");
    const auto display_maximum_it = control_json.find("display_maximum");
    const bool has_display_minimum = display_minimum_it != control_json.end();
    const bool has_display_maximum = display_maximum_it != control_json.end();
    if (has_display_minimum != has_display_maximum)
      throw std::invalid_argument("Control display range must define both minimum and maximum");
    if (has_display_minimum)
    {
      if (!display_minimum_it->is_number() || !display_maximum_it->is_number())
        throw std::invalid_argument("Control display range must be numeric");
      control.display_minimum = display_minimum_it->get<double>();
      control.display_maximum = display_maximum_it->get<double>();
      if (!std::isfinite(control.display_minimum) || !std::isfinite(control.display_maximum)
          || !(control.display_minimum < control.display_maximum))
        throw std::invalid_argument("Control display minimum must be less than maximum");
    }
    control.model_parameter_index = index;
    if (!(control.minimum < control.maximum))
      throw std::invalid_argument("Mapped control minimum must be less than maximum");
    if (control.default_value < control.minimum || control.default_value > control.maximum)
      throw std::invalid_argument("Mapped control default_value is outside its range");
    if (control.IsSwitch() && control.default_value != control.minimum && control.default_value != control.maximum)
      throw std::invalid_argument("Switch control default_value must select one endpoint");
    manifest.controls[static_cast<std::size_t>(index)] = std::move(control);
    assigned[static_cast<std::size_t>(index)] = true;
  }
  if (std::find(assigned.begin(), assigned.end(), false) != assigned.end())
    throw std::invalid_argument("Manifest does not map every global condition index");
  if (manifest.morph.enabled)
  {
    const bool duplicates_learned_morph = std::any_of(
      manifest.controls.begin(), manifest.controls.begin() + manifest.global_condition_size,
      [](const Control& control) { return control.id == "morph"; });
    if (duplicates_learned_morph)
      throw std::invalid_argument("Manifest cannot combine learned Morph with the dual-model Morph slot");
    const auto& morph_control = manifest.controls.back();
    if (morph_control.IsSwitch() || morph_control.minimum != 0.0 || morph_control.maximum != 1.0)
      throw std::invalid_argument("Manifest morph control must be a continuous [0, 1] parameter");
  }

  const auto fixed_controls_it = json.find("fixed_controls");
  if (fixed_controls_it != json.end())
  {
    if (!fixed_controls_it->is_array() || fixed_controls_it->size() > 16U)
      throw std::invalid_argument("Manifest fixed_controls must be a bounded array");
    for (const auto& fixed_json : *fixed_controls_it)
    {
      if (!fixed_json.is_object())
        throw std::invalid_argument("Manifest fixed_controls must contain objects");
      FixedControl fixed;
      fixed.id = detail::RequiredString(fixed_json, "id", 64U);
      fixed.label = detail::RequiredString(fixed_json, "label", 64U);
      fixed.taper = detail::RequiredString(fixed_json, "taper", 16U);
      if (!fixed.IsSwitch() && !fixed.IsLinear())
        throw std::invalid_argument("Fixed controls support switch or linear taper only");
      fixed.value = detail::RequiredFiniteNumber(fixed_json, "value");
      if (fixed.IsSwitch())
      {
        if (fixed.value != 0.0 && fixed.value != 1.0)
          throw std::invalid_argument("Fixed switch value must be 0 or 1");
      }
      else
      {
        fixed.minimum = detail::RequiredFiniteNumber(fixed_json, "minimum");
        fixed.maximum = detail::RequiredFiniteNumber(fixed_json, "maximum");
        if (!(fixed.minimum < fixed.maximum))
          throw std::invalid_argument("Fixed linear control minimum must be less than maximum");
        if (fixed.value < fixed.minimum || fixed.value > fixed.maximum)
          throw std::invalid_argument("Fixed linear control value is outside its range");
        const auto unit_it = fixed_json.find("unit");
        if (unit_it != fixed_json.end())
        {
          if (!unit_it->is_string())
            throw std::invalid_argument("Manifest fixed control unit must be a string");
          fixed.unit = unit_it->get<std::string>();
          if (fixed.unit.size() > 16U)
            throw std::invalid_argument("Manifest fixed control unit length is invalid");
        }
      }
      const bool duplicates_mapped = std::any_of(manifest.controls.begin(), manifest.controls.end(),
                                                 [&fixed](const Control& control) { return control.id == fixed.id; });
      const bool duplicates_fixed =
        std::any_of(manifest.fixed_controls.begin(), manifest.fixed_controls.end(),
                    [&fixed](const FixedControl& existing) { return existing.id == fixed.id; });
      if (duplicates_mapped || duplicates_fixed)
        throw std::invalid_argument("Manifest control ids must be unique across mapped and fixed controls");
      manifest.fixed_controls.push_back(std::move(fixed));
    }
  }

  manifest.runtime_quality = detail::ParseRuntimeQuality(model);
  return manifest;
}

inline void ValidateAndApplyRuntime(nam::DSP& model, const int expected_parameter_count,
                                    const RuntimeQuality& runtime_quality)
{
  if (model.NumGlobalParams() != expected_parameter_count)
    throw std::invalid_argument("Manifest global_condition_size disagrees with the loaded model");

  auto* slimmable = dynamic_cast<nam::SlimmableModel*>(&model);
  const bool manifest_slimmable = runtime_quality.mode == "slimmable";
  if (manifest_slimmable != (slimmable != nullptr))
    throw std::invalid_argument("Manifest runtime_quality mode disagrees with the loaded model");
  if (slimmable == nullptr)
    return;

  const auto actual_breakpoints = slimmable->GetSlimmableSizeBreakpoints();
  if (actual_breakpoints.size() != runtime_quality.breakpoints.size())
    throw std::invalid_argument("Manifest slimmable breakpoints disagree with the loaded model");
  for (std::size_t index = 0; index < actual_breakpoints.size(); ++index)
    if (std::abs(actual_breakpoints[index] - runtime_quality.breakpoints[index]) > 1.0e-12)
      throw std::invalid_argument("Manifest slimmable breakpoints disagree with the loaded model");
  slimmable->SetSlimmableSize(runtime_quality.default_size);
}

inline void ValidateAndApplyRuntime(nam::DSP& model, const Manifest& manifest)
{
  ValidateAndApplyRuntime(model, manifest.global_condition_size, manifest.runtime_quality);
}

inline void ValidateExpectedSampleRate(nam::DSP& model, const double manifest_sample_rate)
{
  const double expected_sample_rate = model.GetExpectedSampleRate();
  if (std::isfinite(expected_sample_rate) && expected_sample_rate > 0.0
      && std::abs(expected_sample_rate - manifest_sample_rate) > 1.0e-6)
    throw std::invalid_argument("Manifest sample_rate disagrees with the loaded model");
}

inline std::unique_ptr<nam::DSP> LoadPlayerDSP(const Manifest& manifest)
{
  auto primary = detail::LoadVerifiedModel(manifest.model_path, manifest.model_sha256);
  if (primary == nullptr)
    throw std::runtime_error("NAM Core returned no primary model");
  ValidateAndApplyRuntime(*primary, manifest);
  ValidateExpectedSampleRate(*primary, manifest.sample_rate);
  if (!manifest.morph.enabled)
    return primary;

  auto secondary = detail::LoadVerifiedModel(manifest.morph.model_path, manifest.morph.model_sha256);
  if (secondary == nullptr)
    throw std::runtime_error("NAM Core returned no morph model");
  ValidateAndApplyRuntime(*secondary, 0, manifest.morph.runtime_quality);
  ValidateExpectedSampleRate(*secondary, manifest.morph.sample_rate);
  return std::make_unique<MorphingDSP>(
    std::move(primary), std::move(secondary), manifest.sample_rate, manifest.morph.settings);
}

struct ImportedAmplifier
{
  Manifest manifest;
  std::unique_ptr<nam::DSP> model;
};

// The picker can return a temporary file on mobile. Publish the exact validated
// bytes under a stable path for host recall and for concurrent plugin instances.
inline ImportedAmplifier ImportDessPackage(const std::filesystem::path& source, const std::filesystem::path& library)
{
  if (!library.is_absolute())
    throw std::invalid_argument("Custom amplifier library needs an absolute path");
  const auto bytes = ReadDessPackageBytes(source);
  const auto hash = detail::DigestHex(detail::Sha256Bytes(bytes.data(), bytes.size()));
  const auto destination = library / (hash + ".dess");
  auto manifest = ParseDessPackage(destination, bytes, kMaxDAWConditionControls);
  auto model = LoadPlayerDSP(manifest);
  std::filesystem::create_directories(library);
  detail::WriteVerifiedCacheFile(destination, bytes.data(), bytes.size(), hash);
  return {std::move(manifest), std::move(model)};
}
} // namespace nam::player
