#pragma once

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cctype>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <limits>
#include <optional>
#include <string>
#include <vector>

#if defined(_WIN32)
#include <io.h>
#else
#include <unistd.h>
#endif

namespace dessamp::preset
{
constexpr std::array<std::uint8_t, 8> kMagic{{'D', 'E', 'S', 'S', 'P', 'R', 'S', 'T'}};
constexpr std::uint32_t kFormatVersion = 1;
constexpr std::size_t kHeaderSize = 16;
constexpr std::size_t kMaxStateBytes = 4U * 1024U * 1024U;
constexpr const char* kExtension = ".desspreset";

struct DecodeResult
{
  std::vector<std::uint8_t> state;
  std::string error;

  explicit operator bool() const { return error.empty(); }
};

inline void PutU32LE(std::vector<std::uint8_t>& bytes, const std::uint32_t value)
{
  for (unsigned int shift = 0; shift < 32; shift += 8)
    bytes.push_back(static_cast<std::uint8_t>((value >> shift) & 0xffU));
}

inline std::uint32_t GetU32LE(const std::uint8_t* bytes)
{
  return static_cast<std::uint32_t>(bytes[0])
         | (static_cast<std::uint32_t>(bytes[1]) << 8U)
         | (static_cast<std::uint32_t>(bytes[2]) << 16U)
         | (static_cast<std::uint32_t>(bytes[3]) << 24U);
}

inline bool Encode(const std::uint8_t* state, const std::size_t stateSize,
                   std::vector<std::uint8_t>& encoded, std::string& error)
{
  encoded.clear();
  error.clear();
  if ((state == nullptr && stateSize != 0U) || stateSize > kMaxStateBytes
      || stateSize > static_cast<std::size_t>(std::numeric_limits<std::uint32_t>::max()))
  {
    error = "Preset state exceeds the supported size limit";
    return false;
  }

  encoded.reserve(kHeaderSize + stateSize);
  encoded.insert(encoded.end(), kMagic.begin(), kMagic.end());
  PutU32LE(encoded, kFormatVersion);
  PutU32LE(encoded, static_cast<std::uint32_t>(stateSize));
  if (stateSize != 0U)
    encoded.insert(encoded.end(), state, state + stateSize);
  return true;
}

inline DecodeResult Decode(const std::uint8_t* bytes, const std::size_t size)
{
  DecodeResult result;
  if (bytes == nullptr || size < kHeaderSize)
  {
    result.error = "Preset file is truncated";
    return result;
  }
  if (!std::equal(kMagic.begin(), kMagic.end(), bytes))
  {
    result.error = "Preset file has an invalid header";
    return result;
  }

  const auto version = GetU32LE(bytes + kMagic.size());
  if (version != kFormatVersion)
  {
    result.error = "Preset file uses an unsupported version";
    return result;
  }

  const auto declaredSize = static_cast<std::size_t>(GetU32LE(bytes + 12U));
  if (declaredSize > kMaxStateBytes)
  {
    result.error = "Preset state exceeds the supported size limit";
    return result;
  }
  if (declaredSize != size - kHeaderSize)
  {
    result.error = declaredSize > size - kHeaderSize ? "Preset file is truncated"
                                                      : "Preset file contains trailing data";
    return result;
  }

  result.state.assign(bytes + kHeaderSize, bytes + size);
  return result;
}

inline DecodeResult Read(const std::filesystem::path& path)
{
  DecodeResult result;
  std::error_code ec;
  const auto fileSize = std::filesystem::file_size(path, ec);
  if (ec)
  {
    result.error = "Preset file could not be opened";
    return result;
  }
  if (fileSize < kHeaderSize)
  {
    result.error = "Preset file is truncated";
    return result;
  }
  if (fileSize > kHeaderSize + kMaxStateBytes)
  {
    result.error = "Preset file exceeds the supported size limit";
    return result;
  }

  std::vector<std::uint8_t> encoded(static_cast<std::size_t>(fileSize));
  std::ifstream input(path, std::ios::binary);
  if (!input || !input.read(reinterpret_cast<char*>(encoded.data()),
                            static_cast<std::streamsize>(encoded.size())))
  {
    result.error = "Preset file could not be read completely";
    return result;
  }
  char trailingByte = 0;
  if (input.read(&trailingByte, 1))
  {
    result.error = "Preset file changed while it was being read";
    return result;
  }
  return Decode(encoded.data(), encoded.size());
}

inline std::filesystem::path EnsureExtension(std::filesystem::path path)
{
  if (path.extension() != kExtension)
    path += kExtension;
  return path;
}

inline std::uint64_t StableScopeHash(const std::string& amplifierId)
{
  std::uint64_t hash = 1469598103934665603ULL;
  for (const unsigned char character : amplifierId)
  {
    hash ^= static_cast<std::uint64_t>(character);
    hash *= 1099511628211ULL;
  }
  return hash;
}

inline std::string ScopeDirectoryName(const std::string& amplifierId)
{
  if (amplifierId.empty())
    return {};

  std::string readable;
  readable.reserve(std::min<std::size_t>(amplifierId.size(), 40U));
  bool previousSeparator = false;
  for (const unsigned char character : amplifierId)
  {
    const bool allowed = std::isalnum(character) || character == '-' || character == '_';
    if (allowed)
    {
      readable.push_back(static_cast<char>(std::tolower(character)));
      previousSeparator = false;
    }
    else if (!readable.empty() && !previousSeparator)
    {
      readable.push_back('-');
      previousSeparator = true;
    }
    if (readable.size() >= 40U)
      break;
  }
  while (!readable.empty() && readable.back() == '-')
    readable.pop_back();
  if (readable.empty())
    readable = "amplifier";

  constexpr char hex[] = "0123456789abcdef";
  const std::uint64_t hash = StableScopeHash(amplifierId);
  readable.push_back('-');
  for (int shift = 60; shift >= 0; shift -= 4)
    readable.push_back(hex[(hash >> static_cast<unsigned int>(shift)) & 0x0fU]);
  return readable;
}

inline std::filesystem::path ScopedDirectory(const std::filesystem::path& root,
                                             const std::string& amplifierId)
{
  const std::string component = ScopeDirectoryName(amplifierId);
  return component.empty() ? std::filesystem::path{} : root / component;
}

inline std::filesystem::path ScopedSavePath(const std::filesystem::path& directory,
                                            const std::filesystem::path& requestedPath)
{
  if (directory.empty() || requestedPath.filename().empty())
    return {};
  return EnsureExtension(directory / requestedPath.filename());
}

// A library name is a label, never a path or the read-only factory entry.
inline std::string UserLabel(std::string name)
{
  const auto first = name.find_first_not_of(" \t");
  if (first == std::string::npos) return {};
  name = name.substr(first, name.find_last_not_of(" \t") - first + 1);
  if (name.front() == '.' || name.back() == '.' || name.size() > 120
      || name.find_first_of("/\\:*?\"<>|") != std::string::npos
      || std::any_of(name.begin(), name.end(), [](unsigned char c){ return c < 32 || c == 127; })) return {};
  auto lower = name;
  std::transform(lower.begin(), lower.end(), lower.begin(), [](unsigned char c){ return std::tolower(c); });
  if (lower == "model default" || std::filesystem::path(lower).extension() == kExtension) return {};
  return name;
}

inline bool WriteAtomic(const std::filesystem::path& requestedPath, const std::uint8_t* state,
                        const std::size_t stateSize, std::filesystem::path& savedPath,
                        std::string& error, bool createNew = false)
{
  std::vector<std::uint8_t> encoded;
  if (!Encode(state, stateSize, encoded, error))
    return false;

  savedPath = EnsureExtension(requestedPath);
  if (savedPath.empty() || savedPath.filename().empty())
  {
    error = "Preset path is invalid";
    return false;
  }

  static std::atomic<std::uint64_t> counter{0};
  const auto nonce = static_cast<std::uint64_t>(
                       std::chrono::steady_clock::now().time_since_epoch().count())
                     ^ counter.fetch_add(1U, std::memory_order_relaxed);
  std::filesystem::path temporary = savedPath;
  temporary += ".tmp-" + std::to_string(nonce);

#if defined(_WIN32)
  FILE* output = _wfopen(temporary.c_str(), L"wb");
#else
  FILE* output = std::fopen(temporary.c_str(), "wb");
#endif
  if (output == nullptr)
  {
    error = "Preset temporary file could not be created";
    return false;
  }

  bool ok = std::fwrite(encoded.data(), 1U, encoded.size(), output) == encoded.size();
  ok = ok && std::fflush(output) == 0;
#if defined(_WIN32)
  if (ok)
    ok = _commit(_fileno(output)) == 0;
#else
  if (ok)
    ok = ::fsync(::fileno(output)) == 0;
#endif
  if (std::fclose(output) != 0)
    ok = false;

  if (!ok)
  {
    std::error_code ignored;
    std::filesystem::remove(temporary, ignored);
    error = "Preset temporary file could not be written completely";
    return false;
  }

  std::error_code renameError;
  if (createNew)
  {
    // Publish a fully written file without replacing an existing sound, even
    // if another open plug-in instance picks the same name simultaneously.
    std::filesystem::create_hard_link(temporary, savedPath, renameError);
    if (!renameError)
    {
      std::error_code ignored;
      std::filesystem::remove(temporary, ignored);
    }
  }
  else
    std::filesystem::rename(temporary, savedPath, renameError);
  if (renameError)
  {
    std::error_code ignored;
    std::filesystem::remove(temporary, ignored);
    error = renameError == std::errc::file_exists ? "That preset name is already in this bank. Choose another name."
                                                  : "The preset file could not be saved.";
    return false;
  }
  return true;
}

inline std::vector<std::filesystem::path> Enumerate(const std::filesystem::path& directory)
{
  std::vector<std::filesystem::path> files;
  std::error_code ec;
  if (directory.empty() || !std::filesystem::is_directory(directory, ec) || ec)
    return files;

  std::filesystem::directory_iterator iterator(directory, ec);
  const std::filesystem::directory_iterator end;
  while (!ec && iterator != end)
  {
    const auto& entry = *iterator;
    std::error_code entryError;
    if (entry.is_regular_file(entryError) && !entryError && entry.path().extension() == kExtension)
      files.push_back(entry.path());
    iterator.increment(ec);
  }

  std::sort(files.begin(), files.end(), [](const auto& lhs, const auto& rhs) {
    const auto lhsName = lhs.filename().generic_string();
    const auto rhsName = rhs.filename().generic_string();
    return lhsName == rhsName ? lhs.generic_string() < rhs.generic_string() : lhsName < rhsName;
  });
  return files;
}

inline std::optional<std::size_t> WrappedIndex(const std::size_t count, const std::size_t current,
                                               const int delta)
{
  if (count == 0U)
    return std::nullopt;
  const auto normalizedCurrent = current < count ? current : 0U;
  const auto signedCount = static_cast<std::int64_t>(count);
  auto wrapped = (static_cast<std::int64_t>(normalizedCurrent) + static_cast<std::int64_t>(delta))
                 % signedCount;
  if (wrapped < 0)
    wrapped += signedCount;
  return static_cast<std::size_t>(wrapped);
}

inline std::optional<std::filesystem::path> Adjacent(
  const std::vector<std::filesystem::path>& files, const std::filesystem::path& current, const int delta)
{
  if (files.empty())
    return std::nullopt;
  const auto found = std::find(files.begin(), files.end(), current);
  const std::size_t currentIndex = found == files.end()
                                     ? (delta < 0 ? files.size() - 1U : 0U)
                                     : static_cast<std::size_t>(std::distance(files.begin(), found));
  if (found == files.end())
    return files[currentIndex];
  return files[*WrappedIndex(files.size(), currentIndex, delta)];
}

inline std::string Label(const std::filesystem::path& path)
{
  return path.empty() ? "Model Default" : path.stem().string();
}
} // namespace dessamp::preset
