#include <AudioToolbox/AudioToolbox.h>
#include <dlfcn.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace
{
constexpr UInt32 kDessAmpSubtype = 'DsAP';
constexpr UInt32 kDessAmpManufacturer = 'AdMs';
constexpr std::size_t kLegacyParameterCount = 20;
constexpr std::size_t kCurrentParameterCount = 33;

AudioComponentDescription MakeDescription()
{
  AudioComponentDescription description{};
  description.componentType = kAudioUnitType_Effect;
  description.componentSubType = kDessAmpSubtype;
  description.componentManufacturer = kDessAmpManufacturer;
  return description;
}

AudioComponent RegisterComponentBinary(const char* binaryPath, void*& libraryHandle)
{
  libraryHandle = dlopen(binaryPath, RTLD_NOW | RTLD_LOCAL);
  if (libraryHandle == nullptr)
    throw std::runtime_error(std::string("Could not load staged AU binary: ") + dlerror());
  dlerror();
  void* symbol = dlsym(libraryHandle, "DessAmp_Factory");
  if (const char* error = dlerror())
    throw std::runtime_error(std::string("Could not find DessAmp_Factory: ") + error);
  const auto factory = reinterpret_cast<AudioComponentFactoryFunction>(symbol);
  auto description = MakeDescription();
  description.componentFlags = kAudioComponentFlag_SandboxSafe;
  const AudioComponent component =
    AudioComponentRegister(&description, CFSTR("DessAmp staged migration test"), 0x000200, factory);
  if (component == nullptr)
    throw std::runtime_error("AudioComponentRegister returned no staged component");
  return component;
}

template <typename T>
void AppendNative(std::vector<std::uint8_t>& bytes, const T& value)
{
  const auto offset = bytes.size();
  bytes.resize(offset + sizeof(T));
  std::memcpy(bytes.data() + offset, &value, sizeof(T));
}

void AppendString(std::vector<std::uint8_t>& bytes, const std::string& value)
{
  const auto length = static_cast<std::int32_t>(value.size());
  AppendNative(bytes, length);
  bytes.insert(bytes.end(), value.begin(), value.end());
}

std::vector<std::uint8_t> MakeStateChunk(const std::string& version, const std::vector<double>& parameters,
                                         const std::string& playerManifestPath = "")
{
  std::vector<std::uint8_t> bytes;
  AppendString(bytes, "###NeuralAmpModeler###");
  AppendString(bytes, version);
  AppendString(bytes, playerManifestPath); // Player manifest in 0.2; historical NAMPath before it.
  AppendString(bytes, ""); // Historical IRPath, intentionally no user data.
  for (const double parameter : parameters)
    AppendNative(bytes, parameter);
  return bytes;
}

CFStringRef MakeString(const char* value)
{
  return CFStringCreateWithCString(kCFAllocatorDefault, value, kCFStringEncodingUTF8);
}

void SetDictionaryNumber(CFMutableDictionaryRef dictionary, const char* key, const SInt32 value)
{
  CFStringRef keyString = MakeString(key);
  CFNumberRef number = CFNumberCreate(kCFAllocatorDefault, kCFNumberSInt32Type, &value);
  CFDictionarySetValue(dictionary, keyString, number);
  CFRelease(number);
  CFRelease(keyString);
}

void SetDictionaryString(CFMutableDictionaryRef dictionary, const char* key, const char* value)
{
  CFStringRef keyString = MakeString(key);
  CFStringRef valueString = MakeString(value);
  CFDictionarySetValue(dictionary, keyString, valueString);
  CFRelease(valueString);
  CFRelease(keyString);
}

void SetDictionaryData(CFMutableDictionaryRef dictionary, const char* key, const std::vector<std::uint8_t>& bytes)
{
  CFStringRef keyString = MakeString(key);
  CFDataRef data = CFDataCreate(kCFAllocatorDefault, bytes.data(), static_cast<CFIndex>(bytes.size()));
  CFDictionarySetValue(dictionary, keyString, data);
  CFRelease(data);
  CFRelease(keyString);
}

OSStatus ApplyState(AudioUnit unit, const std::string& version, const std::vector<double>& parameters,
                    const std::string& playerManifestPath = "")
{
  const auto bytes = MakeStateChunk(version, parameters, playerManifestPath);
  CFMutableDictionaryRef dictionary = CFDictionaryCreateMutable(kCFAllocatorDefault, 0,
                                                                 &kCFTypeDictionaryKeyCallBacks,
                                                                 &kCFTypeDictionaryValueCallBacks);
  SetDictionaryNumber(dictionary, kAUPresetVersionKey, 0x000200);
  SetDictionaryNumber(dictionary, kAUPresetTypeKey, static_cast<SInt32>(kAudioUnitType_Effect));
  SetDictionaryNumber(dictionary, kAUPresetSubtypeKey, static_cast<SInt32>(kDessAmpSubtype));
  SetDictionaryNumber(dictionary, kAUPresetManufacturerKey, static_cast<SInt32>(kDessAmpManufacturer));
  // Exercise the real factory-preset name. The AU wrapper restores this preset
  // before parsing and must roll it back if the state is rejected.
  SetDictionaryString(dictionary, kAUPresetNameKey, "Empty");
  SetDictionaryData(dictionary, kAUPresetDataKey, bytes);

  CFPropertyListRef propertyList = dictionary;
  const OSStatus status = AudioUnitSetProperty(unit, kAudioUnitProperty_ClassInfo, kAudioUnitScope_Global, 0,
                                               &propertyList, sizeof(propertyList));
  CFRelease(dictionary);
  return status;
}

template <std::size_t N>
bool ExpectParameters(AudioUnit unit, const std::array<double, N>& expected, const char* fixtureName)
{
  for (AudioUnitParameterID parameter = 0; parameter < expected.size(); ++parameter)
  {
    AudioUnitParameterValue actual = 0.0f;
    const OSStatus status =
      AudioUnitGetParameter(unit, parameter, kAudioUnitScope_Global, 0, &actual);
    if (status != noErr || std::abs(static_cast<double>(actual) - expected[parameter]) > 1.0e-4)
    {
      std::cerr << fixtureName << " parameter " << parameter << " mismatch: expected " << expected[parameter]
                << ", got " << actual << ", OSStatus " << status << '\n';
      return false;
    }
  }
  return true;
}

template <std::size_t N>
bool ExpectAppliedState(AudioUnit unit, const std::string& version, const std::vector<double>& parameters,
                        const std::array<double, N>& expected, const char* fixtureName)
{
  const OSStatus status = ApplyState(unit, version, parameters);
  if (status != noErr)
  {
    std::cerr << fixtureName << " was rejected with OSStatus " << status << '\n';
    return false;
  }
  return ExpectParameters(unit, expected, fixtureName);
}

template <std::size_t N>
bool ExpectRejectedState(AudioUnit unit, const std::string& version, const std::vector<double>& parameters,
                         const std::array<double, N>& unchanged, const char* fixtureName)
{
  const OSStatus status = ApplyState(unit, version, parameters);
  if (status != kAudioUnitErr_InvalidPropertyValue)
  {
    std::cerr << fixtureName << " returned OSStatus " << status << ", expected "
              << kAudioUnitErr_InvalidPropertyValue << '\n';
    return false;
  }
  return ExpectParameters(unit, unchanged, fixtureName);
}
} // namespace

int main(int argc, char* argv[])
{
  const char* componentBinary = nullptr;
  if (argc == 3 && std::string(argv[1]) == "--component-binary")
    componentBinary = argv[2];
  else if (argc != 1)
  {
    std::cerr << "usage: test_au_state_migration [--component-binary /absolute/DessAmp]\n";
    return 2;
  }

  void* stagedLibrary = nullptr;
  AudioComponent component = nullptr;
  try
  {
    if (componentBinary != nullptr)
      component = RegisterComponentBinary(componentBinary, stagedLibrary);
    else
    {
      auto description = MakeDescription();
      component = AudioComponentFindNext(nullptr, &description);
    }
  }
  catch (const std::exception& error)
  {
    std::cerr << error.what() << '\n';
    if (stagedLibrary != nullptr)
      dlclose(stagedLibrary);
    return 1;
  }
  if (component == nullptr)
  {
    std::cerr << "DessAmp AU is not registered\n";
    if (stagedLibrary != nullptr)
      dlclose(stagedLibrary);
    return 1;
  }

  AudioUnit unit = nullptr;
  const OSStatus createStatus = AudioComponentInstanceNew(component, &unit);
  if (createStatus != noErr || unit == nullptr)
  {
    std::cerr << "AudioComponentInstanceNew failed with OSStatus " << createStatus << '\n';
    if (stagedLibrary != nullptr)
      dlclose(stagedLibrary);
    return 1;
  }

  // Exact f55d020 positional shape. Gain 10 deliberately verifies clamping to
  // the current trained maximum 8. Legacy DessBlock index 1 migrates to Green
  // at stable current host index 2. The absent drive stage remains disabled.
  const std::vector<double> legacy010{
    -3.0, 10.0, -45.0, 2.0, 3.0, 4.0, -2.0, 0.0, 1.0, 0.0, 0.0, 10.0, 0.0, 1.0,
  };
  const std::array<double, kCurrentParameterCount> expectedLegacy{
    -3.0, 8.0, -45.0, 2.0, 3.0, 4.0, -2.0, 0.0, 5.0, 0.0,
    1.0, 0.0, 5.0, 0.0, 0.0, 0.0, 10.0, 0.0, 2.0, 1.0,
    0.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0,
  };

  // Existing 20-value states were also mislabeled 0.1.0 and must remain exact.
  const std::vector<double> currentTagged010{
    -1.0, 7.0, -55.0, 6.0, 5.0, 4.0, 1.0, -4.0, 8.0, 1.0,
    0.0, 1.0, 7.0, 1.0, 3.0, 1.0, 11.0, 0.0, 4.0, 0.0,
  };
  const std::array<double, kCurrentParameterCount> expectedCurrent010{
    -1.0, 7.0, -55.0, 6.0, 5.0, 4.0, 1.0, -4.0, 8.0, 1.0,
    0.0, 1.0, 7.0, 1.0, 3.0, 1.0, 11.0, 0.0, 4.0, 0.0,
    0.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0,
  };

  // 0.1.1 establishes the unambiguous current layout.
  const std::vector<double> current011{
    2.0, 3.0, -65.0, 4.0, 5.0, 6.0, -3.0, 2.0, 1.0, 0.0,
    1.0, 0.0, 4.0, 0.0, 2.0, 0.0, 12.0, 0.0, 0.0, 1.0,
  };
  const std::array<double, kCurrentParameterCount> expectedCurrent011{
    2.0, 3.0, -65.0, 4.0, 5.0, 6.0, -3.0, 2.0, 1.0, 0.0,
    1.0, 0.0, 4.0, 0.0, 2.0, 0.0, 12.0, 0.0, 0.0, 1.0,
    0.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0,
  };

  // 0.2 appends Player Mode plus eight stable automation slots. Apply it
  // before the old fixtures so the migration checks prove those appended
  // values are reset rather than inherited from a previously loaded state.
  const std::vector<double> current020{
    1.0, 4.0, -70.0, 5.0, 5.0, 5.0, -1.0, 0.0, 5.0, 0.0,
    0.0, 0.0, 5.0, 0.0, 0.0, 0.0, 12.0, 0.0, 2.0, 1.0,
    0.0, 5.0, 15.0, 25.0, 35.0, 45.0, 55.0, 65.0, 75.0,
  };
  static_assert(kLegacyParameterCount == 20);
  const std::array<double, kCurrentParameterCount> expectedCurrent020{
    1.0, 4.0, -70.0, 5.0, 5.0, 5.0, -1.0, 0.0, 5.0, 0.0,
    0.0, 0.0, 5.0, 0.0, 0.0, 0.0, 12.0, 0.0, 2.0, 1.0,
    0.0, 5.0, 15.0, 25.0, 35.0, 45.0, 55.0, 65.0, 75.0,
  };

  // Unsupported historical custom slots depended on path-based loading, which
  // no longer exists. The recognized 18-value WIP layout used incompatible
  // drive semantics. Both must be rejected without changing any parameter.
  auto unsupportedLegacyCustom = legacy010;
  unsupportedLegacyCustom.back() = 2.0;
  const std::vector<double> unsupportedWip18(currentTagged010.begin(), currentTagged010.begin() + 18);

  auto levels020 = current020;
  levels020.insert(levels020.end(), {-3.0, 1.0, -7.0, 8.0});
  std::array<double, kCurrentParameterCount> expectedLevels020{};
  std::copy(levels020.begin(), levels020.end(), expectedLevels020.begin());
  const bool passed = ExpectAppliedState(unit, "0.2.0", levels020, expectedLevels020,
                                          "33-value state with four pedal levels")
                      && ExpectAppliedState(unit, "0.2.0", current020, expectedCurrent020,
                                          "current 29-value 0.2.0")
                      && ExpectAppliedState(unit, "0.1.0", legacy010, expectedLegacy,
                                            "legacy 14-value 0.1.0")
                      && ExpectAppliedState(unit, "0.1.0", currentTagged010, expectedCurrent010,
                                            "current 20-value 0.1.0")
                      && ExpectAppliedState(unit, "0.1.1", current011, expectedCurrent011,
                                            "current 20-value 0.1.1")
                      && ExpectRejectedState(unit, "0.1.0", unsupportedLegacyCustom, expectedCurrent011,
                                             "unsupported legacy custom amp")
                      && ExpectRejectedState(unit, "0.1.0", unsupportedWip18, expectedCurrent011,
                                             "unsupported 18-value WIP layout");

  AudioComponentInstanceDispose(unit);
  if (stagedLibrary != nullptr)
    dlclose(stagedLibrary);
  if (!passed)
    return 1;

  std::cout << "DessAmp AU state migration passed: 33 pedal and 29 original parameters, "
               "legacy 14 and 20 parameters, append-only resets, atomic rejection for legacy custom and WIP18\n";
  return 0;
}
