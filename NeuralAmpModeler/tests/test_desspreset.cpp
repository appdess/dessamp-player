#include "../DessPresetFile.h"

#include <cassert>
#include <filesystem>
#include <fstream>
#include <iostream>

namespace
{
std::vector<std::uint8_t> Encode(const std::vector<std::uint8_t>& payload)
{
  std::vector<std::uint8_t> encoded;
  std::string error;
  assert(dessamp::preset::Encode(payload.data(), payload.size(), encoded, error));
  assert(error.empty());
  return encoded;
}
} // namespace

int main()
{
  const std::vector<std::uint8_t> payload{0x00, 0x01, 0x7f, 0x80, 0xff, 0x00};
  const auto encoded = Encode(payload);
  const auto decoded = dessamp::preset::Decode(encoded.data(), encoded.size());
  assert(decoded);
  assert(decoded.state == payload);

  for (std::size_t size = 0; size < dessamp::preset::kHeaderSize; ++size)
    assert(!dessamp::preset::Decode(encoded.data(), size));

  auto truncated = encoded;
  truncated.pop_back();
  assert(!dessamp::preset::Decode(truncated.data(), truncated.size()));

  auto trailing = encoded;
  trailing.push_back(0x42);
  assert(!dessamp::preset::Decode(trailing.data(), trailing.size()));

  auto wrongMagic = encoded;
  wrongMagic[0] ^= 0xff;
  assert(!dessamp::preset::Decode(wrongMagic.data(), wrongMagic.size()));

  auto wrongVersion = encoded;
  wrongVersion[8] = 2;
  assert(!dessamp::preset::Decode(wrongVersion.data(), wrongVersion.size()));

  std::vector<std::uint8_t> oversizeHeader(dessamp::preset::kHeaderSize, 0);
  std::copy(dessamp::preset::kMagic.begin(), dessamp::preset::kMagic.end(), oversizeHeader.begin());
  oversizeHeader[8] = 1;
  const auto oversize = static_cast<std::uint32_t>(dessamp::preset::kMaxStateBytes + 1U);
  oversizeHeader[12] = static_cast<std::uint8_t>(oversize & 0xffU);
  oversizeHeader[13] = static_cast<std::uint8_t>((oversize >> 8U) & 0xffU);
  oversizeHeader[14] = static_cast<std::uint8_t>((oversize >> 16U) & 0xffU);
  oversizeHeader[15] = static_cast<std::uint8_t>((oversize >> 24U) & 0xffU);
  assert(!dessamp::preset::Decode(oversizeHeader.data(), oversizeHeader.size()));

  std::vector<std::uint8_t> tooLarge(dessamp::preset::kMaxStateBytes + 1U, 0);
  std::vector<std::uint8_t> rejected;
  std::string error;
  assert(!dessamp::preset::Encode(tooLarge.data(), tooLarge.size(), rejected, error));

  assert(!dessamp::preset::WrappedIndex(0, 0, 1));
  assert(*dessamp::preset::WrappedIndex(1, 0, 1) == 0);
  assert(*dessamp::preset::WrappedIndex(1, 0, -1) == 0);
  assert(*dessamp::preset::WrappedIndex(3, 0, -1) == 2);
  assert(*dessamp::preset::WrappedIndex(3, 2, 1) == 0);

  const auto unique = std::to_string(
    static_cast<unsigned long long>(std::chrono::steady_clock::now().time_since_epoch().count()));
  const auto directory = std::filesystem::temp_directory_path() / ("dessamp-preset-test-" + unique);
  std::filesystem::create_directory(directory);
  const auto a = directory / "A.desspreset";
  const auto b = directory / "B.desspreset";
  const auto ignored = directory / "C.txt";
  std::ofstream(b).put('b');
  std::ofstream(ignored).put('x');
  std::ofstream(a).put('a');

  const auto files = dessamp::preset::Enumerate(directory);
  assert((files == std::vector<std::filesystem::path>{a, b}));
  assert(*dessamp::preset::Adjacent(files, a, -1) == b);
  assert(*dessamp::preset::Adjacent(files, b, 1) == a);
  assert(*dessamp::preset::Adjacent(files, {}, 1) == a);
  assert(*dessamp::preset::Adjacent(files, {}, -1) == b);

  const auto savedWithoutExtension = directory / "Atomic";
  std::filesystem::path savedPath;
  assert(dessamp::preset::WriteAtomic(savedWithoutExtension, payload.data(), payload.size(), savedPath,
                                      error));
  assert(savedPath == directory / "Atomic.desspreset");
  const auto readBack = dessamp::preset::Read(savedPath);
  assert(readBack);
  assert(readBack.state == payload);
  assert(dessamp::preset::Label(savedPath) == "Atomic");
  const std::vector<std::uint8_t> overwrittenPayload{0x51, 0x41, 0x56, 0x45};
  std::filesystem::path overwrittenPath;
  assert(dessamp::preset::WriteAtomic(
    savedPath, overwrittenPayload.data(), overwrittenPayload.size(), overwrittenPath, error));
  assert(overwrittenPath == savedPath);
  assert(dessamp::preset::Read(overwrittenPath).state == overwrittenPayload);

  const auto scopedRoot = directory / "Scoped";
  const auto fireDirectory = dessamp::preset::ScopedDirectory(scopedRoot, "family:fire");
  const auto redlineDirectory = dessamp::preset::ScopedDirectory(scopedRoot, "family:redline");
  assert(!fireDirectory.empty());
  assert(!redlineDirectory.empty());
  assert(fireDirectory != redlineDirectory);
  assert(fireDirectory.parent_path() == scopedRoot);
  assert(redlineDirectory.parent_path() == scopedRoot);
  assert(fireDirectory.filename().string().find("family-fire-") == 0U);
  assert(redlineDirectory.filename().string().find("family-redline-") == 0U);
  assert(dessamp::preset::ScopedDirectory(scopedRoot, {}).empty());

  std::filesystem::create_directories(fireDirectory);
  std::filesystem::create_directories(redlineDirectory);
  const auto fireRequested = dessamp::preset::ScopedSavePath(fireDirectory, "Modern");
  const auto redlineRequested = dessamp::preset::ScopedSavePath(redlineDirectory, "Modern.desspreset");
  assert(fireRequested == fireDirectory / "Modern.desspreset");
  assert(redlineRequested == redlineDirectory / "Modern.desspreset");
  assert(dessamp::preset::ScopedSavePath(fireDirectory, "../Escape").parent_path() == fireDirectory);

  std::filesystem::path fireSaved;
  std::filesystem::path redlineSaved;
  const std::vector<std::uint8_t> firePayload{0xf1, 0x01};
  const std::vector<std::uint8_t> redlinePayload{0xd1, 0x02};
  assert(dessamp::preset::WriteAtomic(
    fireRequested, firePayload.data(), firePayload.size(), fireSaved, error));
  assert(dessamp::preset::WriteAtomic(
    redlineRequested, redlinePayload.data(), redlinePayload.size(), redlineSaved, error));
  assert(fireSaved != redlineSaved);
  assert((dessamp::preset::Enumerate(fireDirectory)
          == std::vector<std::filesystem::path>{fireSaved}));
  assert((dessamp::preset::Enumerate(redlineDirectory)
          == std::vector<std::filesystem::path>{redlineSaved}));
  assert(dessamp::preset::Read(fireSaved).state == firePayload);
  assert(dessamp::preset::Read(redlineSaved).state == redlinePayload);

  assert(dessamp::preset::UserLabel("  Lead 1  ") == "Lead 1");
  for (auto invalid : {"", "   ", "../elsewhere", "A/B", ".hidden", "Model Default", "MODEL DEFAULT", "a.desspreset"})
    assert(dessamp::preset::UserLabel(invalid).empty());
  const auto newSound = fireDirectory / "Lead 1";
  assert(dessamp::preset::WriteAtomic(newSound, firePayload.data(), firePayload.size(), savedPath, error, true));
  assert(!dessamp::preset::WriteAtomic(newSound, redlinePayload.data(), redlinePayload.size(), savedPath, error, true));
  assert(dessamp::preset::Read(savedPath).state == firePayload);
  std::filesystem::remove_all(directory);
  std::cout << "DESS AMP preset envelope, amplifier scope, and navigation tests passed\n";
  return 0;
}
