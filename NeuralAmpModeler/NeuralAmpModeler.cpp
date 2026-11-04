#include <algorithm> // std::clamp, std::min
#include <cctype>
#include <cmath> // pow
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <set>
#include <sstream> // std::stringstream
#include <utility>
#include <vector>

#include "Colors.h"
#include "../NeuralAmpModelerCore/NAM/activations.h"
#include "../NeuralAmpModelerCore/NAM/get_dsp.h"
// clang-format off
// These includes need to happen in this order or else the latter won't know
// a bunch of stuff.
#include "NeuralAmpModeler.h"
#include "IPlug_include_in_plug_src.h"
// clang-format on
#include "architecture.hpp"
#include "../iPlug2/IPlug/IPlugPaths.h"

#include "DessPresetFile.h"
#include "NeuralAmpModelerControls.h"
#include "PlayerMorphPresentation.h"
#include "PlayerShellIdentity.h"
#ifdef OS_IOS
#include <CoreFoundation/CoreFoundation.h>
#endif

namespace
{
int GetBoundedChunkString(const IByteChunk& chunk, WDL_String& output, const int startPos, const int maxLength)
{
  if (startPos < 0 || maxLength < 0 || startPos > chunk.Size() - static_cast<int>(sizeof(int)))
    return -1;

  int length = 0;
  const int stringStart = chunk.Get(&length, startPos);
  if (stringStart < 0 || length < 0 || length > maxLength || length > chunk.Size() - stringStart)
    return -1;

  if (length == 0)
    output.Set("");
  else
    output.Set(reinterpret_cast<const char*>(chunk.GetData() + stringStart), length);
  return stringStart + length;
}

// iPlug locates a Windows bundle by module handle and an Apple bundle by ID.
void PlayerResourcePath(WDL_String& path, const char* bundleId)
{
#if defined(OS_WIN) && defined(APP_API)
  PluginPath(path, gHINSTANCE);
  if (path.GetLength() > 0)
    path.Append("Resources");
#elif defined(OS_WIN)
  BundleResourcePath(path, gHINSTANCE);
#else
  BundleResourcePath(path, bundleId);
#endif
}

using BundledIRChoice = std::pair<std::string, std::string>;

std::vector<BundledIRChoice> GetBundledIRChoices(const char* bundleId)
{
  WDL_String resourcePath;
  PlayerResourcePath(resourcePath, bundleId);
  if (!resourcePath.GetLength()) return {};

  // Stable shelf positions and paths across private builds and in-place upgrades.
  const auto directory = std::filesystem::path(resourcePath.Get()) / "IRs";
  const std::pair<const char*, const char*> cabinets[] = {
    {"DZL", "Desfire.wav"}, {"Gleichrichter", "Gleichrichter.wav"},
    {"Green", "Green.wav"}, {"SHFLD Mix", "SHFLD-Mix.wav"}
  };
  std::vector<BundledIRChoice> choices;
  for (const auto& cabinet : cabinets)
  {
    const auto path = directory / cabinet.second;
    std::error_code error;
    if (std::filesystem::is_regular_file(path, error))
      choices.emplace_back(cabinet.first, path.string());
  }
  return choices;
}

std::string GetBundledDefaultIRPath(const char* bundleId)
{
  const auto choices = GetBundledIRChoices(bundleId);
  return choices.empty() ? "" : choices.front().second;
}

void AppendBundledAmplifierPaths(const std::filesystem::path& directory, std::vector<std::string>& paths)
{
  std::error_code error;
  if (!std::filesystem::is_directory(directory, error) || error)
    return;

  std::vector<std::filesystem::path> packages;
  for (std::filesystem::directory_iterator iterator(directory, error), end; !error && iterator != end;
       iterator.increment(error))
  {
    const auto& entry = *iterator;
    if (!entry.is_regular_file())
      continue;
    std::string extension = entry.path().extension().string();
    std::transform(extension.begin(), extension.end(), extension.begin(),
                   [](const unsigned char character) { return static_cast<char>(std::tolower(character)); });
    if (extension == ".dess")
      packages.push_back(entry.path());
  }
  std::sort(packages.begin(), packages.end());
  for (const auto& package : packages)
    paths.push_back(package.string());
}

std::vector<std::string> GetBundledAmplifierPaths(const char* bundleId)
{
  WDL_String resourcePath;
  PlayerResourcePath(resourcePath, bundleId);
  if (!resourcePath.GetLength())
    return {};
  const std::filesystem::path resources(resourcePath.Get());

  std::vector<std::string> paths;
  AppendBundledAmplifierPaths(resources / "Amplifiers", paths);
  const std::size_t beforePrivate = paths.size();
  AppendBundledAmplifierPaths(resources / "PrivateValidation" / "Amplifiers", paths);

  // Read old one-off private validation bundles without requiring them to be
  // rebuilt. New builds use the catalog directory above.
  if (paths.size() == beforePrivate)
  {
    const auto legacy = resources / "PrivateValidation" / "amp.dess";
    std::error_code error;
    if (std::filesystem::is_regular_file(legacy, error) && !error)
      paths.push_back(legacy.string());
  }
  return paths;
}

double GetVisibleAmpModelValue(const int modelIndex)
{
  switch (modelIndex)
  {
    case 1:
    case 3: return 1.0 / 3.0;
    case 2: return 2.0 / 3.0;
    case 4: return 1.0;
    default: return 0.0;
  }
}

int GetHostAmpModelIndex(const int visibleIndex)
{
  constexpr int hostIndices[] = {0, 1, 2, 4};
  return hostIndices[std::clamp(visibleIndex, 0, 3)];
}
} // namespace

using namespace iplug;
using namespace igraphics;

const double kDCBlockerFrequency = 5.0;

// Styles
const IVColorSpec colorSpec{
  DEFAULT_BGCOLOR, // Background
  PluginColors::NAM_THEMECOLOR, // Foreground
  PluginColors::NAM_THEMECOLOR.WithOpacity(0.3f), // Pressed
  PluginColors::NAM_THEMECOLOR.WithOpacity(0.4f), // Frame
  PluginColors::MOUSEOVER, // Highlight
  DEFAULT_SHCOLOR, // Shadow
  PluginColors::NAM_THEMECOLOR, // Extra 1
  COLOR_RED, // Extra 2 --> color for clipping in meters
  PluginColors::NAM_THEMECOLOR.WithContrast(0.1f), // Extra 3
};

const IVStyle style =
  IVStyle{true, // Show label
          true, // Show value
          colorSpec,
          {DEFAULT_TEXT_SIZE + 10.f, EVAlign::Middle, PluginColors::NAM_THEMEFONTCOLOR}, // Knob label text
          {DEFAULT_TEXT_SIZE + 10.f, EVAlign::Bottom, PluginColors::NAM_THEMEFONTCOLOR}, // Knob value text
          DEFAULT_HIDE_CURSOR,
          DEFAULT_DRAW_FRAME,
          false,
          DEFAULT_EMBOSS,
          0.2f,
          2.f,
          DEFAULT_SHADOW_OFFSET,
          DEFAULT_WIDGET_FRAC,
          DEFAULT_WIDGET_ANGLE};
const IVStyle radioButtonStyle =
  style
    .WithColor(EVColor::kON, PluginColors::NAM_THEMECOLOR) // Pressed buttons and their labels
    .WithColor(EVColor::kOFF, PluginColors::NAM_THEMECOLOR.WithOpacity(0.1f)) // Unpressed buttons
    .WithColor(EVColor::kX1, PluginColors::NAM_THEMECOLOR.WithOpacity(0.6f)); // Unpressed buttons' labels

EMsgBoxResult _ShowMessageBox(iplug::igraphics::IGraphics* pGraphics, const char* str, const char* caption,
                              EMsgBoxType type)
{
#ifdef OS_MAC
  // macOS is backwards?
  return pGraphics->ShowMessageBox(caption, str, type);
#else
  return pGraphics->ShowMessageBox(str, caption, type);
#endif
}

const std::string kCalibrateInputParamName = "CalibrateInput";
const bool kDefaultCalibrateInput = false;
const std::string kInputCalibrationLevelParamName = "InputCalibrationLevel";
const double kDefaultInputCalibrationLevel = 12.0;


NeuralAmpModeler::NeuralAmpModeler(const InstanceInfo& info)
: Plugin(info, MakeConfig(kNumParams, kNumPresets))
{
  _InitToneStack();
  _InitBoost();
  nam::activations::Activation::enable_fast_tanh();
  GetParam(kInputLevel)->InitGain("Input", 0.0, -20.0, 20.0, 0.1);
  GetParam(kGain)->InitDouble("Gain", 5.0, 2.0, 8.0, 0.1, ""); // Restricted to trained range (2-8)
  GetParam(kAmpModel)->InitEnum(
    "Amp Model", 0, {"DessTortion-blue", "DessTortion-red", "DessBlock-green", "DessBlock-red", "SickDess"});
  GetParam(kToneBass)->InitDouble("Bass", 5.0, 0.0, 10.0, 0.1);
  GetParam(kToneMid)->InitDouble("Middle", 5.0, 0.0, 10.0, 0.1);
  GetParam(kToneTreble)->InitDouble("Treble", 5.0, 0.0, 10.0, 0.1);
  GetParam(kOutputLevel)->InitGain("Output", 0.0, -40.0, 40.0, 0.1);
  GetParam(kBoostLevel)->InitGain("Boost Level", 0.0, -20.0, 20.0, 0.1);
  GetParam(kOxideLevel)->InitGain("Oxide Level", 0.0, -20.0, 20.0, 0.1);
  GetParam(kAmberLevel)->InitGain("Amber Level", 0.0, -20.0, 20.0, 0.1);
  GetParam(kVerdantLevel)->InitGain("Verdant Level", 0.0, -20.0, 20.0, 0.1);
  GetParam(kNocturneLevel)->InitGain("Nocturne Level", 0.0, -20.0, 20.0, 0.1);
  GetParam(kBoostTone)->InitDouble("Boost Tone", 5.0, 0.0, 10.0, 0.1);
  GetParam(kNoiseGateThreshold)->InitGain("Threshold", -85.0, -100.0, 0.0, 0.1);
  GetParam(kNoiseGateActive)->InitBool("NoiseGateActive", true);
  GetParam(kEQActive)->InitBool("ToneStack", false);
  // Raw is the universally safe default. Model-aware options are enabled only
  // when the selected capture publishes the required metadata.
  GetParam(kOutputMode)->InitEnum("OutputMode", 0, {"Raw", "Normalized", "Calibrated"}); // TODO DRY w/ control
  GetParam(kIRToggle)->InitBool("IRToggle", true);
  GetParam(kCalibrateInput)->InitBool(kCalibrateInputParamName.c_str(), kDefaultCalibrateInput);
  GetParam(kInputCalibrationLevel)
    ->InitDouble(kInputCalibrationLevelParamName.c_str(), kDefaultInputCalibrationLevel, -60.0, 60.0, 0.1, "dBu");

  // Drive Params. The restored fixed captures use Boost Level for their actual
  // post-model trim. Boost Tone and BoostOutput remain host-visible only for
  // compatibility with older projects; neither is exposed as a working knob.
  // A portable conditioned amp starts with no separately licensed drive model.
  // Existing sessions can still restore or enable the append-only boost slot.
  GetParam(kBoostActive)->InitBool("Boost", false);
  GetParam(kBoostOutput)->InitDouble("BoostOutput", 5.0, 0.0, 10.0, 0.1);
  GetParam(kBoostModel)->InitEnum("Boost Model", 0, {"OD808", "SD1", "TS9", "aesahaettr"});

  GetParam(kNAMActive)->InitBool("Amp Enabled", true); // Default Amp On
  // DESS AMP is a conditioned-model player. Keep this append-only compatibility
  // parameter for old state chunks, but make the player path the fresh-instance
  // default and do not expose a mode switch in the product UI.
  GetParam(kPlayerMode)->InitBool("Player Mode", true);
  for (int index = 0; index < kNumPlayerConditionParams; ++index)
  {
    const std::string name = "Player Condition " + std::to_string(index + 1);
    // Preserve the append-only 0..100 host contract, but do not quantize a
    // live Morph sweep to whole percentage points.
    GetParam(kFirstPlayerConditionParam + index)->InitDouble(name.c_str(), 50.0, 0.0, 100.0, 0.01, "%");
    mPlayerConditionTargets[static_cast<std::size_t>(index)].store(0.5, std::memory_order_relaxed);
    mPendingPlayerDefaults[static_cast<std::size_t>(index)].store(0.5, std::memory_order_relaxed);
  }

  mAmpModelIdx.store(GetParam(kAmpModel)->Int(), std::memory_order_relaxed);
  mPlayerModeTarget.store(GetParam(kPlayerMode)->Bool(), std::memory_order_relaxed);
  mAmpActiveTarget.store(GetParam(kNAMActive)->Bool(), std::memory_order_relaxed);
  mBoostModelIdx.store(GetParam(kBoostModel)->Int(), std::memory_order_relaxed);
  mBoostActiveTarget.store(GetParam(kBoostActive)->Bool(), std::memory_order_relaxed);
  mIRActiveTarget.store(GetParam(kIRToggle)->Bool(), std::memory_order_relaxed);

  // Ordinary builds with no rights-cleared catalog still start empty. A build
  // may carry multiple verified .dess packages and load the first one without
  // a picker; later selection only changes this path and uses the normal
  // off-audio-thread package loader.
  WDL_String appSupportPath;
  AppSupportPath(appSupportPath, false);
  mDessPresetRootDirectory =
    std::filesystem::u8path(appSupportPath.Get()) / BUNDLE_NAME / "Presets";
  if (appSupportPath.GetLength()) _LoadStandaloneRig();
  mBundledAmplifierPaths = GetBundledAmplifierPaths(GetBundleID());
  mBundledAmplifierProductFamilies.resize(mBundledAmplifierPaths.size());
  mBundledAmplifierProductVariants.resize(mBundledAmplifierPaths.size());
  std::set<std::string> observedProductKeys;
  for (std::size_t index = 0; index < mBundledAmplifierPaths.size(); ++index)
  {
    std::string name = "HEAD " + std::to_string(index + 1);
    try
    {
      const auto manifest = nam::player::LoadManifest(mBundledAmplifierPaths[index]);
      switch (dessamp::player::ResolveHeadShellIdentity(manifest.head_shell_style, manifest.morph.enabled, {}))
      {
        case dessamp::player::HeadShellIdentity::Fire: name = "FIRE"; break;
        case dessamp::player::HeadShellIdentity::Forge: name = "FORGE"; break;
        case dessamp::player::HeadShellIdentity::Redline: name = "REDLINE"; break;
        default: break;
      }
      mBundledAmplifierProductFamilies[index] = manifest.product_family;
      mBundledAmplifierProductVariants[index] = manifest.product_variant;
    }
    catch (const std::exception&)
    {
      // Keep the package discoverable so the ordinary loader can surface its
      // validation error. It simply remains an independent catalog entry.
    }
    const std::string productKey = mBundledAmplifierProductFamilies[index].empty()
                                     ? "path:" + std::filesystem::path(mBundledAmplifierPaths[index]).lexically_normal().string()
                                     : "family:" + mBundledAmplifierProductFamilies[index];
    if (observedProductKeys.insert(productKey).second)
    {
      mBundledAmplifierProductRootIndices.push_back(index);
      mAmplifierSlotNames.push_back(std::move(name));
    }
  }
  mAmplifierSlotNames.emplace_back("CUSTOM");
  if (!mBundledAmplifierPaths.empty())
  {
    _LoadBundledFactoryDefaults();
    _ApplyBundledFactoryDefaults(mBundledAmplifierPaths.front(), false);
    _SetPlayerManifestPathAndRequest(mBundledAmplifierPaths.front().c_str(), true);
    _SetDessPresetAmplifierScope(
      _DessPresetAmplifierIDForManifestPath(mBundledAmplifierPaths.front()));
  }

  mNoiseGateTrigger.AddListener(&mNoiseGateGain);
  mCurrentParams.resize(kNumPlayerConditionParams, 0.5); // No render-thread growth for conditioned models.
  // AUv2 prunes declared-but-uninitialized factory presets during component
  // construction. Initialize the one advertised preset so ClassInfo rollback
  // and readback always have a valid current preset index.
  MakeDefaultPreset(mBundledAmplifierPaths.empty() ? "Empty" : "Model Default", kNumPresets);

  mMakeGraphicsFunc = [&]() {

#ifdef OS_IOS
    int width = PLUG_WIDTH;
    int height = PLUG_HEIGHT;
    GetScreenDimensions(width, height);
    return MakeGraphics(*this, width, height, PLUG_FPS, 1.f);
#else
    return MakeGraphics(*this, PLUG_WIDTH, PLUG_HEIGHT, PLUG_FPS, 1.f);
#endif
  };

  mLayoutFunc = [&](IGraphics* pGraphics) {
#ifdef OS_IOS
    if (pGraphics->NControls())
    {
      _LayoutIOS(pGraphics);
      return;
    }
    pGraphics->SetLayoutOnResize(true);
#else
    pGraphics->AttachCornerResizer(EUIResizerMode::Scale, false);
#endif
#ifndef OS_IOS
    pGraphics->AttachTextEntryControl();
#endif
    pGraphics->EnableMouseOver(true);
    pGraphics->EnableTooltips(true);
    pGraphics->EnableMultiTouch(true);

    pGraphics->LoadFont("Roboto-Regular", ROBOTO_FN);
    pGraphics->LoadFont("Roboto-Semibold", ROBOTO_SEMIBOLD_FN);
    pGraphics->LoadFont("Michroma-Regular", MICHROMA_FN);

    {
      using namespace DessAmpUI;
      const auto bounds = pGraphics->GetBounds();
      const ResponsiveLayout layout(bounds);
      const auto headShell = pGraphics->LoadBitmap(DESSAMP_HEAD_SHELL_FN);
      const auto dualRedHeadShell = pGraphics->LoadBitmap(DESSAMP_DUAL_RED_HEAD_SHELL_FN);
      const auto forgeHeadShell = pGraphics->LoadBitmap(DESSAMP_FORGE_HEAD_SHELL_FN);
      const auto fireHeadShell = pGraphics->LoadBitmap(DESSAMP_FIRE_HEAD_SHELL_FN);
      const auto knobFace = pGraphics->LoadBitmap(DESSAMP_KNOB_FACE_FN);
      const auto dessBadge = pGraphics->LoadBitmap(DESSAMP_BADGE_FN);

      pGraphics->AttachControl(new DessAmpBackdropControl(bounds), kCtrlTagBackground);
      pGraphics->AttachControl(
        new DessAmpHeadControl(layout.head, headShell, dualRedHeadShell, forgeHeadShell, fireHeadShell, dessBadge),
        kCtrlTagPlayerHead, "AMP_STAGE");
      pGraphics->AttachControl(new DessAmpGlassChromeControl(bounds), kCtrlTagPlayerGlassChrome);

      pGraphics->AttachControl(
        new DessAmpWordmark(layout.wordmark, pGraphics->LoadBitmap(DESSAMP_WORDMARK_FN)),
        kCtrlTagPlayerWordmark);

      auto viewAction = [this](IControl* caller) {
        mPlayerPage = caller->As<DessAmpTabsControl>()->Selected();
        _UpdatePlayerModeUI();
      };
      pGraphics->AttachControl(new DessAmpTabsControl(layout.tabs, viewAction, mPlayerPage), kCtrlTagPlayerTabs);

      auto* settingsButton = pGraphics->AttachControl(
        new DessAmpIconButtonControl(layout.settings, DessAmpIconButtonControl::Icon::Settings,
                                     [pGraphics](IControl*) {
                                       if (auto* panel = pGraphics->GetControlWithTag(kCtrlTagSettingsBox))
                                         panel->Hide(false);
                                     }),
        kCtrlTagPlayerSettingsButton);
      settingsButton->SetTooltip("Settings");
      pGraphics->AttachControl(new DessAmpIconButtonControl(layout.tuner, DessAmpIconButtonControl::Icon::Tune,
        [this](IControl*) { _ShowTuner(true); }), kCtrlTagTunerButton)->SetTooltip("Chromatic guitar tuner");

      auto* power = pGraphics->AttachControl(
        new DessAmpPowerControl(layout.power, kNAMActive), kCtrlTagPlayerPowerButton);
      power->SetTooltip("Enable or bypass the amplifier");

      // Persistent studio utility strip.
      pGraphics->AttachControl(new DessAmpMeterControl(layout.inputMeter), kCtrlTagInputMeter);
      auto* input =
        pGraphics
          ->AttachControl(new DessAmpKnobControl(layout.input, kInputLevel, "Input", knobFace, 52.f),
                          kCtrlTagPlayerInputKnob)
          ->As<DessAmpKnobControl>();
      input->SetPresentation("Input", -20.0, 20.0, 0.0, "dB");
      input->SetTooltip("Input trim");

      auto* gateThreshold = pGraphics
                              ->AttachControl(new DessAmpKnobControl(layout.gate,
                                                                     kNoiseGateThreshold, "Threshold", knobFace, 48.f),
                                              kCtrlTagPlayerGateKnob, "GATE_CONTROLS")
                              ->As<DessAmpKnobControl>();
      gateThreshold->SetPresentation("Threshold", -100.0, 0.0, GetParam(kNoiseGateThreshold)->GetDefault(), "dB");
      auto* gateToggle = pGraphics->AttachControl(
        new DessAmpToggleControl(layout.gateToggle, kNoiseGateActive, "Gate"), kCtrlTagPlayerGateToggle);
      gateToggle->SetTooltip("Enable or bypass the noise gate");

      auto loadPreset = [this](const char* path) { _LoadDessPreset(path); };
      auto savePreset = [this](const char* path, bool createNew) { _SaveDessPreset(path, createNew); };
      auto navigatePreset = [this](const int delta) { _NavigateDessPreset(delta); };
      const auto bundledIRs = GetBundledIRChoices(GetBundleID());
      if (GetParam(kIRToggle)->Bool() && _GetIRPathSnapshot().GetLength() == 0 && !bundledIRs.empty())
      {
        const std::string defaultIRPath = _DefaultCabinetForAmp(std::filesystem::u8path(_GetPlayerManifestPathSnapshot().Get()));
        if (!defaultIRPath.empty())
          _SetIRPathAndRequest(defaultIRPath.c_str());
      }
      auto* modelSelector =
        pGraphics
          ->AttachControl(new DessAmpModelSelectorControl(layout.model, loadPreset, savePreset, navigatePreset),
                          kCtrlTagPlayerManifest)
          ->As<DessAmpModelSelectorControl>();
      modelSelector->SetTooltip("Active factory amplifier and presets");
      auto loadIR = [this](const char* path) {
        if (path == nullptr || path[0] == '\0')
          return;
        if (GetParam(kIRToggle)->Bool() && std::string(path) == _GetIRPathSnapshot().Get())
          return;
        if (!GetParam(kIRToggle)->Bool())
          _SetParameterValueFromMainThread(kIRToggle, 1.0);
        _SetIRPathAndRequest(path);
        mDessPresetModified = true;
        _UpdateDessPresetUI();
      };
      DessAmpCabSelectorControl::Images cabinetImages{{"", pGraphics->LoadBitmap(DESSAMP_CAB_FN)}};
#if defined(OS_MAC) || defined(OS_IOS)
      const std::pair<const char*, const char*> shells[] = {
        {"DZL", "dzl"}, {"SHFLD Mix", "blend"}, {"Gleichrichter", "recto"}, {"Green", "green"}
      };
      for (const auto& shell : shells)
      {
        const auto offered = std::find_if(bundledIRs.begin(), bundledIRs.end(),
          [&](const BundledIRChoice& ir) { return ir.first == shell.first; });
        if (offered == bundledIRs.end())
          continue;
        const auto name = std::filesystem::path(DESSAMP_CAB_FN).parent_path()
                          / (std::string("studio-cab-") + shell.second + "@2x.png");
        cabinetImages.emplace_back(shell.first, pGraphics->LoadBitmap(name.string().c_str()));
      }
#endif
      auto* cabSelector = pGraphics
                            ->AttachControl(new DessAmpCabSelectorControl(
                                              layout.preStage, bundledIRs, std::move(cabinetImages), loadIR,
                                              [this](IControl*) {
                                                _SetParameterValueFromMainThread(kIRToggle, GetParam(kIRToggle)->Bool() ? 0.0 : 1.0);
                                                mDessPresetModified = true;
                                                _UpdateDessPresetUI();
                                              }),
                                            kCtrlTagIRFileBrowser, "CAB_STAGE")
                            ->As<DessAmpCabSelectorControl>();
      cabSelector->SetTooltip("Choose a cabinet IR (Command-I)");
      pGraphics->SetKeyHandlerFunc([this, pGraphics, modelSelector, cabSelector](const IKeyPress& key, bool isUp) {
        auto* choice = pGraphics->GetControlWithTag(kCtrlTagChoicePanel);
        if (choice && !choice->IsHidden())
        {
          if (!isUp && (key.VK == kVK_ESCAPE || key.utf8[0] == 27))
            choice->As<DessAmpChoicePanel>()->Choose(0);
          return true;
        }
        if (auto* tuner = pGraphics->GetControlWithTag(kCtrlTagTunerPanel); tuner && !tuner->IsHidden())
        {
          if (!isUp && (key.VK == kVK_ESCAPE || key.utf8[0] == 27)) _ShowTuner(false);
          return true;
        }
        auto* settings = pGraphics->GetControlWithTag(kCtrlTagSettingsBox);
        if (settings != nullptr && !settings->IsHidden())
        {
          if (!isUp && (key.VK == kVK_ESCAPE || key.utf8[0] == 27))
            settings->Hide(true);
          // Modal settings must not dispatch amp/preset shortcuts underneath.
          return true;
        }
        if (isUp || !key.C)
          return false;

        const char pressed = key.utf8[0];
        if (pressed == 's' || pressed == 'S')
        {
          key.S ? modelSelector->SavePresetAs() : modelSelector->SavePreset();
          return true;
        }
        if (pressed == 'o' || pressed == 'O')
        {
          modelSelector->ImportPreset();
          return true;
        }
        if (pressed == 'i' || pressed == 'I')
        {
          cabSelector->ChooseFile();
          return true;
        }
        return false;
      });

      auto* output = pGraphics
                       ->AttachControl(new DessAmpKnobControl(
                                         layout.output, kOutputLevel, "Output", knobFace, 52.f),
                                       kCtrlTagPlayerOutputKnob)
                       ->As<DessAmpKnobControl>();
      output->SetPresentation("Output", -40.0, 40.0, 0.0, "dB");
      output->SetTooltip("Output trim");
      pGraphics->AttachControl(new DessAmpMeterControl(layout.outputMeter), kCtrlTagOutputMeter);

      constexpr std::array<const char*, 6> initialLabels{
        "Lead Gain", "Bass", "Middle", "Treble", "Lead Volume", "Presence"};
      for (int index = 0; index < kNumPlayerConditionParams; ++index)
      {
        const char* label =
          index < static_cast<int>(initialLabels.size()) ? initialLabels[static_cast<std::size_t>(index)] : "Condition";
        auto* condition =
          pGraphics
            ->AttachControl(new DessAmpKnobControl(layout.conditions[static_cast<std::size_t>(index)],
                                                   kFirstPlayerConditionParam + index, label, knobFace, 84.f),
                            kFirstPlayerConditionCtrlTag + index, "AMP_STAGE")
            ->As<DessAmpKnobControl>();
        condition->SetPresentation(label, 0.0, 10.0, 5.0, "");
        condition->SetTooltip("Load an amplifier to enable this control");
        condition->SetDisabled(true);
        condition->Hide(index >= static_cast<int>(initialLabels.size()));
      }

      // Every learned host slot has both a knob and a binary editor widget.
      // Manifest taper metadata chooses which one is visible; both remain
      // bound to the same append-only parameter ID.
      for (int index = 0; index < kNumPlayerConditionParams; ++index)
      {
        auto* condition = pGraphics
                            ->AttachControl(new DessAmpToggleControl(layout.conditions[static_cast<std::size_t>(index)],
                                                                     kFirstPlayerConditionParam + index, "Condition"),
                                            kFirstPlayerConditionToggleCtrlTag + index, "AMP_STAGE")
                            ->As<DessAmpToggleControl>();
        condition->SetTooltip("Load an amplifier to enable this switch");
        condition->SetDisabled(true);
        condition->Hide(true);
      }

      auto variantAction = [this](const std::string& variant) { _SelectBundledAmplifierVariant(variant); };
      pGraphics
        ->AttachControl(new DessAmpVariantSelectorControl(layout.bottom, variantAction),
                        kCtrlTagPlayerVariantSelector, "AMP_STAGE")
        ->Hide(true);

      pGraphics->AttachControl(new DessAmpReservedToggleControl(layout.bright, "Bright"),
                               kCtrlTagPlayerBrightPlaceholder, "AMP_STAGE");
      pGraphics->AttachControl(new DessAmpReservedToggleControl(layout.bottom, "Bottom"),
                               kCtrlTagPlayerBottomPlaceholder, "AMP_STAGE");
      pGraphics->AttachControl(new DessAmpReservedToggleControl(layout.midBoost, "Mid Boost"),
                               kCtrlTagPlayerMidBoostPlaceholder, "AMP_STAGE");

      pGraphics->AttachControl(new DessAmpEmptyStateControl(layout.empty),
                               kCtrlTagPlayerMode, "AMP_STAGE");
      WDL_String resources;
      PlayerResourcePath(resources, GetBundleID());
      const std::filesystem::path drives = std::filesystem::path(resources.Get()) / "models" / "DessDrive";
      bool driveBankAvailable = resources.GetLength() > 0;
      for (int i = 0; i < 4 && driveBankAvailable; ++i)
      {
        std::error_code ec;
        driveBankAvailable = std::filesystem::is_regular_file(drives / (_GetBoostModelName(i) + ".nam"), ec);
      }
      // A phone or DAW can open the editor before the first audio block. Read
      // the bundled model once here so a supported Tone knob is editable then.
      if (driveBankAvailable && !mVerdantHasTone.has_value())
      {
        try
        {
          const auto pedal = nam::get_dsp(drives / (_GetBoostModelName(kVerdantDriveIndex) + ".nam"));
          mVerdantHasTone = pedal && pedal->NumGlobalParams() == 1;
        }
        catch (const std::exception&) { mVerdantHasTone = false; }
      }
      const std::array<IBitmap, 4> pedals{{pGraphics->LoadBitmap(DESSAMP_OXIDE_FN), pGraphics->LoadBitmap(DESSAMP_AMBER_FN),
                                           pGraphics->LoadBitmap(DESSAMP_VERDANT_FN), pGraphics->LoadBitmap(DESSAMP_NOCTURNE_FN)}};
      auto* bank = pGraphics->AttachControl(new DessAmpPedalBank(layout.pedals, pedals, driveBankAvailable),
        kCtrlTagPlayerPreStage, "PRE_STAGE")->As<DessAmpPedalBank>();
      for (int i = 0; i < 4; ++i)
      {
        auto* knob = pGraphics->AttachControl(new DessAmpKnobControl(bank->LevelBounds(i), kOxideLevel + i, "Level",
          knobFace, 52.f), kCtrlTagPlayerDriveKnob + i, "PRE_STAGE")->As<DessAmpKnobControl>();
        knob->SetPresentation("Level", -20.0, 20.0, 0.0, "dB");
        knob->SetTooltip("Output level of this pedal");
      }
      auto* tone = pGraphics->AttachControl(new DessAmpKnobControl(bank->ToneBounds(), kBoostTone, "Tone",
        knobFace, 52.f), kCtrlTagVerdantTone, "PRE_STAGE")->As<DessAmpKnobControl>();
      tone->SetPresentation("Tone", 0.0, 10.0, 5.0, "");
      tone->SetTooltip("Tone of the conditioned Verdant capture");
      bank->SetHasTone(mVerdantHasTone.value_or(false));

      auto* strip = pGraphics->AttachControl(new DessAmpFactoryStrip(layout.factoryStrip, mAmplifierSlotNames,
        [this](int index) {
          if (index == static_cast<int>(mBundledAmplifierProductRootIndices.size()))
            _SelectCustomAmplifier();
          else
            _SelectBundledAmplifier(index);
        }), kCtrlTagPlayerFactoryStrip)->As<DessAmpFactoryStrip>();
      strip->SetTooltip("Select a head. Select CUSTOM again to choose another .dess file.");
      strip->Select(_AmplifierStripIndex());

      auto* footerStatus = pGraphics->AttachControl(
        new ITextControl(IRECT(layout.footer.L, layout.footer.T, layout.factoryStrip.L - 5.f, layout.footer.B),
                         "NO AMPLIFIER",
                         IText(12.f, TextSecondary, "Roboto-Semibold", EAlign::Near), COLOR_TRANSPARENT),
        kCtrlTagPlayerFooterStatus, "PLAYER_FOOTER");
      footerStatus->SetTooltip("Player status");
      auto* settingsPanel = pGraphics->AttachControl(
        new DessAmpSettingsPanelControl(bounds, [this]() { _ToggleAutoInputMatch(); }, [this]() { _ResetInput(); },
                                        [this]() { _ShowFeedback(); }), kCtrlTagSettingsBox);
      settingsPanel->Hide(true);
      pGraphics->AttachControl(new DessAmpTunerPanel(bounds, [this](IControl*) { _ShowTuner(false); }),
        kCtrlTagTunerPanel)->Hide(true);
      pGraphics->AttachControl(new DessAmpChoicePanel(bounds), kCtrlTagChoicePanel)->Hide(true);

#ifdef OS_IOS
      // Only passive surfaces accept a page swipe. A drag that begins on a
      // knob goes straight to the knob, even if it ends over another control.
      static_cast<IGraphicsIOS*>(pGraphics)->SetPageSwipeHandler([settingsPanel](IControl* hit) {
        if (!settingsPanel->IsHidden()) return false;
        if (!hit) return true;
        switch (hit->GetTag())
        {
          case kCtrlTagBackground: case kCtrlTagPlayerHead: case kCtrlTagPlayerGlassChrome:
          case kCtrlTagPlayerWordmark: case kCtrlTagPlayerTabs:
          case kCtrlTagPlayerPreStage: case kCtrlTagIRFileBrowser: return true;
          default: return false;
        }
      }, [this, pGraphics](bool forward) {
        const bool phone = ResponsiveLayout(pGraphics->GetBounds()).phone;
        const auto next = static_cast<DessAmpPage>(std::clamp(
          int(mPlayerPage) + (forward ? 1 : -1),
          int(kPrePage), int(phone ? kRigPage : kCabPage)));
        if (next == mPlayerPage) return;
        mPlayerPage = next;
        pGraphics->GetControlWithTag(kCtrlTagPlayerTabs)->As<DessAmpTabsControl>()->Configure(next, phone);
        _UpdatePlayerModeUI();
      });
#endif
      pGraphics->ForControlInGroup("PRE_STAGE", [this](IControl* c) { c->Hide(mPlayerPage != kPrePage); });
      pGraphics->ForControlInGroup("CAB_STAGE", [this](IControl* c) { c->Hide(mPlayerPage != kCabPage); });
      pGraphics->ForAllControlsFunc([](IControl* control) {
        control->SetMouseEventsWhenDisabled(false);
        control->SetMouseOverWhenDisabled(false);
      });
      _UpdatePlayerModeUI();
#ifdef OS_IOS
      _LayoutIOS(pGraphics);
#endif
    }
  };
}

#ifdef OS_IOS
void NeuralAmpModeler::_LayoutIOS(IGraphics* graphics)
{
  if (graphics == nullptr)
    return;

  using namespace DessAmpUI;
  const ResponsiveLayout layout(graphics->GetBounds());
  if (!layout.phone && mPlayerPage == kRigPage)
    mPlayerPage = kAmpPage;
  if (auto* tabs = graphics->GetControlWithTag(kCtrlTagPlayerTabs))
    tabs->As<DessAmpTabsControl>()->Configure(mPlayerPage, layout.phone);
  const auto place = [graphics](const int tag, const IRECT& rect) {
    if (auto* control = graphics->GetControlWithTag(tag))
      control->SetTargetAndDrawRECTs(rect);
  };

  place(kCtrlTagBackground, layout.b);
  place(kCtrlTagPlayerHead, layout.head);
  place(kCtrlTagPlayerGlassChrome, layout.b);
  place(kCtrlTagPlayerWordmark, layout.wordmark);
  place(kCtrlTagPlayerTabs, layout.tabs);
  place(kCtrlTagPlayerSettingsButton, layout.settings);
  place(kCtrlTagTunerButton, layout.tuner);
  place(kCtrlTagTunerPanel, layout.b);
  place(kCtrlTagChoicePanel, layout.b);
  place(kCtrlTagPlayerPowerButton, layout.power);

  place(kCtrlTagInputMeter, layout.inputMeter);
  place(kCtrlTagPlayerInputKnob, layout.input);
  place(kCtrlTagPlayerGateKnob, layout.gate);
  place(kCtrlTagPlayerGateToggle, layout.gateToggle);
  place(kCtrlTagPlayerManifest, layout.model);
  place(kCtrlTagIRFileBrowser, layout.preStage);
  place(kCtrlTagPlayerOutputKnob, layout.output);
  place(kCtrlTagOutputMeter, layout.outputMeter);

  for (int index = 0; index < kNumPlayerConditionParams; ++index)
  {
    place(kFirstPlayerConditionCtrlTag + index, layout.conditions[static_cast<std::size_t>(index)]);
    place(kFirstPlayerConditionToggleCtrlTag + index, layout.conditions[static_cast<std::size_t>(index)]);
  }
  place(kCtrlTagPlayerBrightPlaceholder, layout.bright);
  place(kCtrlTagPlayerBottomPlaceholder, layout.bottom);
  place(kCtrlTagPlayerMidBoostPlaceholder, layout.midBoost);
  place(kCtrlTagPlayerVariantSelector, layout.bottom);
  place(kCtrlTagPlayerMode, layout.empty);

  place(kCtrlTagPlayerPreStage, layout.pedals);
  if (auto* bank = graphics->GetControlWithTag(kCtrlTagPlayerPreStage))
  {
    for (int i = 0; i < 4; ++i)
      place(kCtrlTagPlayerDriveKnob + i, bank->As<DessAmpPedalBank>()->LevelBounds(i));
    place(kCtrlTagVerdantTone, bank->As<DessAmpPedalBank>()->ToneBounds());
  }
  place(kCtrlTagPlayerFactoryStrip, layout.factoryStrip);
  place(kCtrlTagPlayerFooterStatus, IRECT(layout.footer.L, layout.footer.T, layout.factoryStrip.L - 5.f, layout.footer.B));
  if (auto* footer = graphics->GetControlWithTag(kCtrlTagPlayerFooterStatus))
    footer->Hide(layout.phone);

  place(kCtrlTagSettingsBox, layout.b);
  _UpdatePlayerModeUI();
  graphics->SetAllControlsDirty();
}

void NeuralAmpModeler::OnParentWindowResize(const int width, const int height)
{
  if (GetUI() != nullptr && width > 0 && height > 0)
    GetUI()->Resize(width, height, 1.f, false);
}

void NeuralAmpModeler::OnHostSelectedViewConfiguration(const int width, const int height)
{
  if (GetUI() != nullptr && width > 0 && height > 0)
    GetUI()->Resize(width, height, 1.f, true);
}
#endif

NeuralAmpModeler::~NeuralAmpModeler()
{
  delete mModel.exchange(nullptr, std::memory_order_acq_rel);
  delete mPendingModel.exchange(nullptr, std::memory_order_acq_rel);
  delete mRetiredModel.exchange(nullptr, std::memory_order_acq_rel);
  delete mIR.exchange(nullptr, std::memory_order_acq_rel);
  delete mPendingIR.exchange(nullptr, std::memory_order_acq_rel);
  delete mRetiredIR.exchange(nullptr, std::memory_order_acq_rel);
  delete mBoostModel.exchange(nullptr, std::memory_order_acq_rel);
  delete mPendingBoostModel.exchange(nullptr, std::memory_order_acq_rel);
  delete mRetiredBoostModel.exchange(nullptr, std::memory_order_acq_rel);
}

void NeuralAmpModeler::ProcessBlock(iplug::sample** inputs, iplug::sample** outputs, int nFrames)
{
  const size_t numChannelsExternalIn = (size_t)NInChansConnected();
  const size_t numChannelsExternalOut = (size_t)NOutChansConnected();
  const size_t numChannelsInternal = kNumChannelsInternal;
  const size_t numFrames = (size_t)nFrames;
  const double sampleRate = GetSampleRate();

  const auto silenceOutputs = [&]() {
    for (size_t c = 0; c < numChannelsExternalOut; ++c)
      if (outputs != nullptr && outputs[c] != nullptr)
        std::fill(outputs[c], outputs[c] + std::max(nFrames, 0), 0.0);
  };

  // Never grow heap-backed buffers from the render callback. Hosts must call
  // OnReset() before processing or increasing the maximum block size.
  if (!mAudioConfigInitialized.load(std::memory_order_acquire) || nFrames < 0
      || nFrames > mAudioMaxBlockSize.load(std::memory_order_acquire))
  {
    silenceOutputs();
    return;
  }

  // One AMP state snapshot governs both handoff adoption and processing for
  // this block. A concurrent automation edge is applied on the next block,
  // never halfway through a fresh-resampler adoption.
  const bool ampActive = mAmpActiveTarget.load(std::memory_order_acquire);
  if (!GetRenderingOffline())
    mTuner.Push(inputs, static_cast<int>(numChannelsExternalIn), nFrames, sampleRate);
  if (!GetRenderingOffline())
    mAutoInputMatch.Process(inputs, nFrames, static_cast<int>(numChannelsExternalIn), sampleRate,
#ifdef APP_API
                           false
#else
                           true
#endif
    );
  _ApplyDSPStaging(ampActive);

  // Offline bounces may synchronously prepare resources. This does not depend
  // on the UI/main-thread OnIdle timer and never runs in realtime mode.
  if (GetRenderingOffline())
  {
    {
      std::lock_guard<std::mutex> lock(mDSPLoadMutex);
      _ServiceDSPRequests();
      _ApplyDSPStaging(ampActive);
    }
    // Offline hosts may never pump OnIdle. Establish calibration, output
    // normalization, and host-visible latency before rendering this block,
    // but notify the host only after releasing the loader mutex.
    _SetInputGain();
    _SetOutputGain();
    _UpdateLatency();
  }

  const bool boostActive = mBoostActiveTarget.load(std::memory_order_acquire);
  const bool irActive = mIRActiveTarget.load(std::memory_order_acquire);
  const std::uint64_t audioEpoch = mAudioConfigEpoch.load(std::memory_order_acquire);
  auto* model = mModel.load(std::memory_order_acquire);
  auto* boostModel = mBoostModel.load(std::memory_order_acquire);
  auto* ir = mIR.load(std::memory_order_acquire);
  // A target selection can change before its replacement has finished loading.
  // Keep rendering through the already-adopted model until the new object is
  // ready, especially while AMP is bypassed: its persistent resampler carries
  // the phase/history that implements the latency reported to the host.
  const bool modelUsable = model != nullptr && model->GetPreparedEpoch() == audioEpoch;
  const bool ampReady = !ampActive || modelUsable;
  const bool boostReady =
    !boostActive
    || (boostModel != nullptr && boostModel->GetPreparedEpoch() == audioEpoch
        && boostModel->GetPreparedRequestToken() == mBoostLoadRequest.load(std::memory_order_acquire)
        && boostModel->GetPreparedSelection() == mBoostModelIdx.load(std::memory_order_acquire));
  const bool irReady = !irActive
                       || (ir != nullptr && ir->GetPreparedEpoch() == audioEpoch
                           && ir->GetPreparedRequestToken() == mIRLoadRequest.load(std::memory_order_acquire));
  if (!ampReady || !boostReady || !irReady)
  {
    silenceOutputs();
    return;
  }

  // Disable floating point denormals
  std::fenv_t fe_state;
  std::feholdexcept(&fe_state);
  disable_denormals();

  // Input is collapsed to mono in preparation for the NAM.
  _ProcessInput(inputs, numFrames, numChannelsExternalIn, numChannelsInternal);

  // SAFETY: Sanitize inputs to prevent NaN poisoning
  for (size_t c = 0; c < numChannelsInternal; ++c)
  {
    for (size_t s = 0; s < numFrames; ++s)
    {
      if (!std::isfinite(mInputPointers[c][s]))
      {
        mInputPointers[c][s] = 0.0;
      }
    }
  }

  const bool noiseGateActive = GetParam(kNoiseGateActive)->Value();
  const bool toneStackActive = GetParam(kEQActive)->Value();

  // Boost processing
  // The mono processor overwrites every output frame; unused capacity is never
  // consumed. Do not clear the entire maximum-block buffer on every callback.
  sample** ampInput = mInputPointers;
  if (boostActive)
  {
    if (boostModel->NumGlobalParams() == 1)
    {
      const double tone = GetParam(kBoostTone)->GetNormalized();
      boostModel->process(mInputPointers, mBoostOutputPointers, nFrames, &tone, 1);
    }
    else
      boostModel->process(mInputPointers, mBoostOutputPointers, nFrames);
    const double boostLeveldB = GetParam(kBoostLevel)->Value()
      + GetParam(kOxideLevel + boostModel->GetPreparedSelection())->Value();
    const double boostLevel = std::pow(10.0, boostLeveldB / 20.0);
    for (int s = 0; s < nFrames; ++s)
      mBoostOutputPointers[0][s] *= boostLevel;
    ampInput = mBoostOutputPointers;

    // SAFETY: Sanitize Boost Output to prevent NaN poisoning in Trigger/Model
    for (size_t c = 0; c < numChannelsInternal; ++c)
    {
      for (size_t s = 0; s < numFrames; ++s)
      {
        if (!std::isfinite(ampInput[c][s]))
        {
          ampInput[c][s] = 0.0;
        }
      }
    }
  }


  // Noise gate trigger
  if (noiseGateActive)
  {
    const double time = 0.01;
    const double threshold = GetParam(kNoiseGateThreshold)->Value(); // GetParam...
    const double ratio = 0.1; // Quadratic...
    const double openTime = 0.005;
    const double holdTime = 0.01;
    const double closeTime = 0.05;
    const dsp::noise_gate::TriggerParams triggerParams(time, threshold, ratio, openTime, holdTime, closeTime);
    mNoiseGateTrigger.SetParams(triggerParams);
    mNoiseGateTrigger.SetSampleRate(sampleRate);
    mNoiseGateTrigger.Process(ampInput, numChannelsInternal, numFrames);
  }

  // Alias ampInput to triggerOutput for model compatibility (bypass trigger audio output)
  sample** triggerOutput = ampInput;

  if (ampActive)
  {
    const int globalParams = model->NumGlobalParams();
    if (model->IsPreparedPlayerMode())
    {
      const auto* manifest = model->GetPlayerManifest();
      if (globalParams > 0 && globalParams <= kNumPlayerConditionParams && manifest != nullptr
          && manifest->controls.size() == static_cast<std::size_t>(globalParams))
      {
        for (int index = 0; index < globalParams; ++index)
        {
          const double normalized =
            mPlayerConditionTargets[static_cast<std::size_t>(index)].load(std::memory_order_relaxed);
          mCurrentParams[static_cast<std::size_t>(index)] =
            manifest->controls[static_cast<std::size_t>(index)].ModelValueFromNormalized(normalized);
        }
        model->process(triggerOutput, mOutputPointers, nFrames, mCurrentParams.data(), globalParams);
      }
      else
      {
        // Snapshot manifests, including A2 Lite/Full containers, have no amp
        // control vector. A validated conditioned artifact always takes the
        // branch above.
        model->process(triggerOutput, mOutputPointers, nFrames);
      }
    }
    else if (globalParams == 1)
    {
      // Read gain from atomic (written by UI thread) into audio-thread-local buffer.
      mCurrentParams[0] = mTargetGain.load(std::memory_order_relaxed);
      model->process(triggerOutput, mOutputPointers, nFrames, mCurrentParams.data(), 1);
    }
    else
    {
      // Fixed snapshots, including A2 Lite/Full containers, have no amp-control vector.
      model->process(triggerOutput, mOutputPointers, nFrames);
    }
  }

  else
  {
    if (modelUsable)
      model->processBypass(triggerOutput, mOutputPointers, nFrames);
    else
      _FallbackDSP(triggerOutput, mOutputPointers, numChannelsInternal, numFrames);
  }
  // Apply the noise gate after the NAM
  sample** gateGainOutput =
    noiseGateActive ? mNoiseGateGain.Process(mOutputPointers, numChannelsInternal, numFrames) : mOutputPointers;

  const double bass = mToneBassTarget.load(std::memory_order_acquire);
  const double middle = mToneMidTarget.load(std::memory_order_acquire);
  const double treble = mToneTrebleTarget.load(std::memory_order_acquire);
  if (bass != mAppliedToneBass)
  {
    mToneStack->SetParam("bass", bass);
    mAppliedToneBass = bass;
  }
  if (middle != mAppliedToneMid)
  {
    mToneStack->SetParam("middle", middle);
    mAppliedToneMid = middle;
  }
  if (treble != mAppliedToneTreble)
  {
    mToneStack->SetParam("treble", treble);
    mAppliedToneTreble = treble;
  }

  sample** toneStackOutPointers = (toneStackActive && mToneStack != nullptr)
                                    ? mToneStack->Process(gateGainOutput, numChannelsInternal, nFrames)
                                    : gateGainOutput;

  sample** irPointers = toneStackOutPointers;
  if (irActive)
    irPointers = ir->Process(toneStackOutPointers, numChannelsInternal, numFrames);

  // And the HPF for DC offset (Issue 271)
  // Fixed coefficients are prepared on sample-rate reset, not per callback.
  sample** hpfPointers = mHighPass.Process(irPointers, numChannelsInternal, numFrames);

  // restore previous floating point state
  std::feupdateenv(&fe_state);

  // Let's get outta here
  // This is where we exit mono for whatever the output requires.
  _ProcessOutput(hpfPointers, outputs, numFrames, numChannelsInternal, numChannelsExternalOut);
  // * Output of input leveling (inputs -> mInputPointers),
  // * Output of output leveling (mOutputPointers -> outputs)
  _UpdateMeters(mInputPointers, outputs, numFrames, numChannelsInternal, numChannelsExternalOut);
}

void NeuralAmpModeler::OnReset()
{
  const auto sampleRate = GetSampleRate();
  const int maxBlockSize = std::max(GetBlockSize(), 1);
  const bool hadConfig = mAudioConfigInitialized.load(std::memory_order_acquire);
  const bool configChanged = !hadConfig || mAudioSampleRate.load(std::memory_order_acquire) != sampleRate
                             || mAudioMaxBlockSize.load(std::memory_order_acquire) != maxBlockSize;
  if (configChanged)
  {
    mAudioConfigInitialized.store(false, std::memory_order_release);
    // Odd epochs are transitions and are never accepted by a staging producer.
    mAudioConfigEpoch.fetch_add(1, std::memory_order_acq_rel);
    mAudioSampleRate.store(sampleRate, std::memory_order_release);
    mAudioMaxBlockSize.store(maxBlockSize, std::memory_order_release);
    mAudioConfigEpoch.fetch_add(1, std::memory_order_acq_rel);
  }

  // Tail is because the HPF DC blocker has a decay.
  // 10 cycles should be enough to pass the VST3 tests checking tail behavior.
  // I'm ignoring the model & IR, but it's not the end of the world.
  const int tailCycles = 10;
  SetTailSize(tailCycles * (int)(sampleRate / kDCBlockerFrequency));
  mInputSender.Reset(sampleRate);
  mOutputSender.Reset(sampleRate);

  // Reserve every plugin-owned processing buffer here. DSP modules are built,
  // reset, prewarmed, and published later from OnIdle(). Existing modules are
  // bypassed until their prepared epoch matches this reset.
  for (std::size_t c = 0; c < kNumChannelsInternal; ++c)
  {
    mInputArray[c].resize(maxBlockSize);
    mOutputArray[c].resize(maxBlockSize);
    mBoostOutputArray[c].resize(maxBlockSize);
    mInputPointers[c] = mInputArray[c].data();
    mOutputPointers[c] = mOutputArray[c].data();
    mBoostOutputPointers[c] = mBoostOutputArray[c].data();
  }
  mNoiseGateTrigger.PrepareBuffers(kNumChannelsInternal, maxBlockSize);
  mNoiseGateGain.PrepareBuffers(kNumChannelsInternal, maxBlockSize);
  mHighPass.PrepareBuffers(kNumChannelsInternal, maxBlockSize);
  mHighPass.SetParams(recursive_linear_filter::HighPassParams(sampleRate, kDCBlockerFrequency));

  const double bass = GetParam(kToneBass)->Value();
  const double middle = GetParam(kToneMid)->Value();
  const double treble = GetParam(kToneTreble)->Value();
  mToneBassTarget.store(bass, std::memory_order_release);
  mToneMidTarget.store(middle, std::memory_order_release);
  mToneTrebleTarget.store(treble, std::memory_order_release);
  mAppliedToneBass = bass;
  mAppliedToneMid = middle;
  mAppliedToneTreble = treble;
  mToneStack->Reset(sampleRate, maxBlockSize);
  mToneStack->SetParam("bass", bass);
  mToneStack->SetParam("middle", middle);
  mToneStack->SetParam("treble", treble);

  mAmpModelIdx.store(GetParam(kAmpModel)->Int(), std::memory_order_release);
  mPlayerModeTarget.store(GetParam(kPlayerMode)->Bool(), std::memory_order_release);
  for (int index = 0; index < kNumPlayerConditionParams; ++index)
    mPlayerConditionTargets[static_cast<std::size_t>(index)].store(
      std::clamp(GetParam(kFirstPlayerConditionParam + index)->Value() / 100.0, 0.0, 1.0), std::memory_order_release);
  mAmpActiveTarget.store(GetParam(kNAMActive)->Bool(), std::memory_order_release);
  mAmpRetryNeeded.store(false, std::memory_order_release);
  mBoostModelIdx.store(GetParam(kBoostModel)->Int(), std::memory_order_release);
  mBoostActiveTarget.store(GetParam(kBoostActive)->Bool(), std::memory_order_release);
  mIRActiveTarget.store(GetParam(kIRToggle)->Bool(), std::memory_order_release);
  {
    std::lock_guard<std::mutex> lock(mPathMutex);
    unsigned int previous = mAmpLoadRequest.fetch_add(1, std::memory_order_acq_rel);
    const unsigned int next = previous + 1U;
    // A device reset can precede adoption of a freshly selected package.
    // Keep pending manifest defaults, while recalled sessions keep their own values.
    if (previous != 0U)
      mApplyPlayerDefaultsRequest.compare_exchange_strong(previous, next, std::memory_order_acq_rel);
  }
  mBoostLoadRequest.fetch_add(1, std::memory_order_release);
  _RequestIRReload();

  mAudioConfigInitialized.store(true, std::memory_order_release);

  _UpdateLatency();
}

void NeuralAmpModeler::_ServiceDSPRequests()
{
  // This service never runs in realtime and is serialized by mDSPLoadMutex.
  // Every non-render dereference either holds that mutex or consumes the
  // immutable metadata cache, so retired objects can be reclaimed here. This
  // also frees each handoff slot for multi-transition offline bounces.
  delete mRetiredModel.exchange(nullptr, std::memory_order_acq_rel);
  delete mRetiredIR.exchange(nullptr, std::memory_order_acq_rel);
  delete mRetiredBoostModel.exchange(nullptr, std::memory_order_acq_rel);

  const unsigned int ampRequest = mAmpLoadRequest.load(std::memory_order_acquire);
  if (ampRequest != mHandledAmpLoadRequest.load(std::memory_order_acquire))
  {
    const int ampModelInt = mAmpModelIdx.load(std::memory_order_acquire);
    const bool playerMode = mPlayerModeTarget.load(std::memory_order_acquire);
    // Keep the selected amp resident even while bypassed. This makes AMP ON/OFF
    // an immediate DSP switch instead of an asynchronous disk-load operation,
    // and it preloads a model selected while the amp is bypassed.
    std::string errorMsg;
    if (playerMode)
    {
      const WDL_String manifestPath = _GetPlayerManifestPathSnapshot();
      if (manifestPath.GetLength() > 0)
        errorMsg = _StagePlayerManifest(manifestPath, ampRequest, ampModelInt);
      else
      {
        // A fresh conditioned player intentionally starts empty. Treat that as
        // a stable no-model state rather than a failed load: the UI already
        // presents the model chooser and the dry fallback remains available.
        delete mPendingModel.exchange(nullptr, std::memory_order_acq_rel);
        mShouldRemoveModel.store(true, std::memory_order_release);
        mStagedPlayerManifestForUI.reset();
        mStagedPlayerManifestRequest = 0;
        mStagedPlayerManifestSelection = -1;
        mStagedPlayerManifestEpoch = 0;
        mStagedPlayerManifestUIChanged.store(true, std::memory_order_release);
      }
    }
    else
    {
      mStagedPlayerManifestForUI.reset();
      mStagedPlayerManifestRequest = 0;
      mStagedPlayerManifestSelection = -1;
      mStagedPlayerManifestEpoch = 0;
      mStagedPlayerManifestUIChanged.store(true, std::memory_order_release);
      errorMsg = _LoadModelForGain(_GetAmpModelName(ampModelInt), GetParam(kGain)->Int(), ampRequest, ampModelInt);
    }
    mHandledAmpLoadRequest.store(ampRequest, std::memory_order_release);
    if (!errorMsg.empty() && ampRequest == mAmpLoadRequest.load(std::memory_order_acquire)
        && ampModelInt == mAmpModelIdx.load(std::memory_order_acquire)
        && playerMode == mPlayerModeTarget.load(std::memory_order_acquire))
    {
      delete mPendingModel.exchange(nullptr, std::memory_order_acq_rel);
      // Preserve an already-adopted model as the latency-compensated bypass
      // path. A failed replacement remains recoverable by the next explicit
      // AMP ON request without collapsing the host's PDC in the meantime.
      if (mModel.load(std::memory_order_acquire) == nullptr)
        mShouldRemoveModel.store(true, std::memory_order_release);
      mAmpRetryNeeded.store(true, std::memory_order_release);
      if (mAmpActiveTarget.exchange(false, std::memory_order_acq_rel))
      {
        GetParam(kNAMActive)->Set(false);
        mAmpDisableNotificationPending.store(true, std::memory_order_release);
      }
      mDeferredAmpLoadError = errorMsg;
    }
    else if (errorMsg.empty())
      mAmpRetryNeeded.store(false, std::memory_order_release);
  }

  const unsigned int boostRequest = mBoostLoadRequest.load(std::memory_order_acquire);
  if (boostRequest != mHandledBoostLoadRequest.load(std::memory_order_acquire))
  {
    std::string errorMsg;
    const int boostModelInt = mBoostModelIdx.load(std::memory_order_acquire);
    if (mBoostActiveTarget.load(std::memory_order_acquire))
      errorMsg = _LoadBoostModel(_GetBoostModelName(boostModelInt), boostRequest, boostModelInt);
    else
    {
      delete mPendingBoostModel.exchange(nullptr, std::memory_order_acq_rel);
      mShouldRemoveBoostModel.store(true, std::memory_order_release);
    }
    mHandledBoostLoadRequest.store(boostRequest, std::memory_order_release);
    if (!errorMsg.empty() && boostRequest == mBoostLoadRequest.load(std::memory_order_acquire)
        && boostModelInt == mBoostModelIdx.load(std::memory_order_acquire)
        && mBoostActiveTarget.load(std::memory_order_acquire))
    {
      delete mPendingBoostModel.exchange(nullptr, std::memory_order_acq_rel);
      mShouldRemoveBoostModel.store(true, std::memory_order_release);
      GetParam(kBoostActive)->Set(false);
      mBoostActiveTarget.store(false, std::memory_order_release);
      mBoostDisableNotificationPending.store(true, std::memory_order_release);
      mDeferredBoostLoadError = errorMsg;
    }
  }

  unsigned int irRequest = 0;
  WDL_String requestedIRPath;
  {
    std::lock_guard<std::mutex> lock(mPathMutex);
    irRequest = mIRLoadRequest.load(std::memory_order_acquire);
    requestedIRPath = mIRPath;
  }
  if (irRequest != mHandledIRLoadRequest.load(std::memory_order_acquire))
  {
    if (!mIRActiveTarget.load(std::memory_order_acquire))
    {
      delete mPendingIR.exchange(nullptr, std::memory_order_acq_rel);
      mShouldRemoveIR.store(true, std::memory_order_release);
      mHandledIRLoadRequest.store(irRequest, std::memory_order_release);
      return;
    }

    if (requestedIRPath.GetLength() == 0)
    {
      const std::string defaultIRPath = _DefaultCabinetForAmp(std::filesystem::u8path(_GetPlayerManifestPathSnapshot().Get()));
      if (!defaultIRPath.empty())
      {
        std::lock_guard<std::mutex> lock(mPathMutex);
        if (irRequest == mIRLoadRequest.load(std::memory_order_acquire) && mIRPath.GetLength() == 0)
          mIRPath.Set(defaultIRPath.c_str());
        requestedIRPath = mIRPath;
      }
    }

    const auto wavState =
      requestedIRPath.GetLength() > 0 ? _StageIR(requestedIRPath, irRequest) : dsp::wav::LoadReturnCode::ERROR_OTHER;
    mHandledIRLoadRequest.store(irRequest, std::memory_order_release);
    bool failedCurrentRequest = false;
    if (wavState != dsp::wav::LoadReturnCode::SUCCESS)
    {
      std::lock_guard<std::mutex> lock(mPathMutex);
      failedCurrentRequest = irRequest == mIRLoadRequest.load(std::memory_order_acquire);
      if (failedCurrentRequest)
        mIRPath.Set("");
    }
    if (failedCurrentRequest)
    {
      delete mPendingIR.exchange(nullptr, std::memory_order_acq_rel);
      mShouldRemoveIR.store(true, std::memory_order_release);
      mIRActiveTarget.store(false, std::memory_order_release);
      GetParam(kIRToggle)->Set(false);
      mIRDisableNotificationPending.store(true, std::memory_order_release);
      if (requestedIRPath.GetLength() > 0)
      {
        std::stringstream message;
        message << "Failed to load IR file " << requestedIRPath.Get() << ":\n";
        message << dsp::wav::GetMsgForLoadReturnCode(wavState);
        mDeferredIRLoadError = message.str();
      }
    }
    else if (wavState == dsp::wav::LoadReturnCode::SUCCESS
             && irRequest == mIRLoadRequest.load(std::memory_order_acquire))
      mStagedIRUIChanged.store(true, std::memory_order_release);
  }
}

void NeuralAmpModeler::OnRestoreState()
{
  Plugin::OnRestoreState();
  mHostStateRepublishPending.store(true, std::memory_order_release);
}

void NeuralAmpModeler::OnIdle()
{
  _SaveStandaloneRig();
  if (auto* ui = GetUI())
    if (auto* panel = ui->GetControlWithTag(kCtrlTagTunerPanel); panel && !panel->IsHidden())
      panel->As<DessAmpTunerPanel>()->Update(mTuner);
  _UpdateAutoInputMatch();
  std::string ampLoadError;
  std::string boostLoadError;
  std::string irLoadError;
  {
    std::lock_guard<std::mutex> lock(mDSPLoadMutex);
    _ServiceDSPRequests();
    ampLoadError.swap(mDeferredAmpLoadError);
    boostLoadError.swap(mDeferredBoostLoadError);
    irLoadError.swap(mDeferredIRLoadError);
  }

  const unsigned int playerDefaultsRequest = mPendingPlayerDefaultsRequest.exchange(0U, std::memory_order_acq_rel);
  if (playerDefaultsRequest != 0U && playerDefaultsRequest == mAmpLoadRequest.load(std::memory_order_acquire)
      && mPlayerModeTarget.load(std::memory_order_acquire))
  {
    // Publishing the selected model's defaults is not a player edit. Preserve
    // any genuine edit already made while the asynchronous model load finished.
    const bool presetWasModified = mDessPresetModified;
    for (int index = 0; index < kNumPlayerConditionParams; ++index)
      _SetParameterValueFromMainThread(
        kFirstPlayerConditionParam + index,
        mPendingPlayerDefaults[static_cast<std::size_t>(index)].load(std::memory_order_acquire));
    mDessPresetModified = presetWasModified;
    _UpdateDessPresetUI();
  }

  if (mStagedPlayerManifestUIChanged.exchange(false, std::memory_order_acq_rel))
    _UpdatePlayerControls();
  if (mStagedIRUIChanged.exchange(false, std::memory_order_acq_rel))
  {
    const WDL_String loadedPath = _GetIRPathSnapshot();
    if (GetUI() && GetUI()->GetControlWithTag(kCtrlTagIRFileBrowser) && loadedPath.GetLength() > 0)
      SendControlMsgFromDelegate(kCtrlTagIRFileBrowser, kMsgTagLoadedIR, loadedPath.GetLength(), loadedPath.Get());
  }

  if (mAmpDisableNotificationPending.exchange(false, std::memory_order_acq_rel))
    _SetParameterValueFromMainThread(kNAMActive, 0.0);
  if (mBoostDisableNotificationPending.exchange(false, std::memory_order_acq_rel))
    _SetParameterValueFromMainThread(kBoostActive, 0.0);
  if (mIRDisableNotificationPending.exchange(false, std::memory_order_acq_rel))
    _SetParameterValueFromMainThread(kIRToggle, 0.0);

  // Never present modal UI while holding mDSPLoadMutex. Apart from avoiding
  // user-controlled loader stalls, this permits timer/modal-loop re-entry.
  if (auto* ui = GetUI())
  {
    if (!ampLoadError.empty())
    {
      if (auto* selector = ui->GetControlWithTag(kCtrlTagPlayerManifest))
      {
        const WDL_String requestedPath = _GetPlayerManifestPathSnapshot();
        std::error_code pathError;
        const bool missing = requestedPath.GetLength() > 0
                             && !std::filesystem::exists(std::filesystem::u8path(requestedPath.Get()), pathError);
        if (missing && !pathError)
          selector->As<DessAmpModelSelectorControl>()->SetMissing();
        else
          selector->As<DessAmpModelSelectorControl>()->SetError();
        if (!missing || pathError)
          _ShowMessageBox(ui, ampLoadError.c_str(), "Failed to load model!", kMB_OK);
      }
    }
    if (!boostLoadError.empty())
      _ShowMessageBox(ui, boostLoadError.c_str(), "Boost model unavailable", kMB_OK);
    if (!irLoadError.empty())
    {
      if (ui->GetControlWithTag(kCtrlTagIRFileBrowser))
        SendControlMsgFromDelegate(kCtrlTagIRFileBrowser, kMsgTagLoadFailed);
      _ShowMessageBox(ui, irLoadError.c_str(), "Failed to load IR!", kMB_OK);
    }
  }

  mInputSender.TransmitData(*this);
  mOutputSender.TransmitData(*this);

  if (mNewModelLoadedInDSP.exchange(false, std::memory_order_acq_rel))
  {
    std::string preparedPath;
    {
      std::lock_guard<std::mutex> lock(mDSPLoadMutex);
      if (auto* model = mModel.load(std::memory_order_acquire))
        preparedPath = model->GetPreparedPath();
    }
    if (!preparedPath.empty())
      _SetNAMPath(preparedPath.c_str());
    _UpdateLatency();
    _SetInputGain();
    _SetOutputGain();
    if (mPlayerModeTarget.load(std::memory_order_acquire))
      _SyncDessPresetScopeToCurrentAmplifier();
    if (GetUI())
    {
      _UpdateControlsFromModel();
      _UpdatePlayerControls();
      const WDL_String manifestPath = _GetPlayerManifestPathSnapshot();
      if (mPlayerModeTarget.load(std::memory_order_acquire) && manifestPath.GetLength()
          && GetUI()->GetControlWithTag(kCtrlTagPlayerManifest))
        SendControlMsgFromDelegate(
          kCtrlTagPlayerManifest, kMsgTagLoadedPlayerManifest, manifestPath.GetLength(), manifestPath.Get());
    }
  }
  if (mNewIRLoadedInDSP.exchange(false, std::memory_order_acq_rel))
  {
    const WDL_String loadedPath = _GetIRPathSnapshot();
    if (GetUI() && GetUI()->GetControlWithTag(kCtrlTagIRFileBrowser) && loadedPath.GetLength() > 0)
      SendControlMsgFromDelegate(kCtrlTagIRFileBrowser, kMsgTagLoadedIR, loadedPath.GetLength(), loadedPath.Get());
  }
  if (mModelCleared.exchange(false, std::memory_order_acq_rel))
  {
    _SetNAMPath("");
    _UpdateLatency();
    _SetInputGain();
    _SetOutputGain();
    if (auto* pGraphics = GetUI())
      _UpdatePlayerControls();
  }

  if (mNewBoostModelLoadedInDSP.exchange(false, std::memory_order_acq_rel))
  {
    std::string preparedPath;
    {
      std::lock_guard<std::mutex> lock(mDSPLoadMutex);
      if (auto* model = mBoostModel.load(std::memory_order_acquire))
      {
        preparedPath = model->GetPreparedPath();
        if (model->GetPreparedSelection() == kVerdantDriveIndex)
          mVerdantHasTone = model->NumGlobalParams() == 1;
      }
    }
    if (auto* ui = GetUI())
      if (auto* bank = ui->GetControlWithTag(kCtrlTagPlayerPreStage))
        bank->As<DessAmpPedalBank>()->SetHasTone(mVerdantHasTone.value_or(false));
    if (!preparedPath.empty())
    {
      _SetBoostNAMPath(preparedPath.c_str());
      const WDL_String loadedPath = _GetBoostNAMPathSnapshot();
      if (GetUI() && GetUI()->GetControlWithTag(kCtrlTagBoostModelFileBrowser))
        SendControlMsgFromDelegate(
          kCtrlTagBoostModelFileBrowser, kMsgTagLoadedBoostModel, loadedPath.GetLength(), loadedPath.Get());
    }
  }
  if (mBoostModelCleared.exchange(false, std::memory_order_acq_rel))
  {
    _SetBoostNAMPath("");
  }
  // Boost enable/disable changes the processing path independently of amp
  // adoption. Notify host PDC here, never from the realtime render callback.
  _UpdateLatency();
  if (auto* graphics = GetUI())
    if (auto* settings = graphics->GetControlWithTag(kCtrlTagSettingsBox))
      settings->As<DessAmpSettingsPanelControl>()->SetAudioStatus(GetSampleRate(), GetBlockSize(), GetLatency());

  if (mHostStateRepublishPending.exchange(false, std::memory_order_acq_rel))
    DirtyParametersFromUI();
}

bool NeuralAmpModeler::SerializeState(IByteChunk& chunk) const
{
  return _SerializePlayerState(chunk, false);
}

bool NeuralAmpModeler::_SerializePlayerState(IByteChunk& chunk, bool ampPreset) const
{
  // If this isn't here when unserializing, then we know we're dealing with something before v0.8.0.
  WDL_String header("###NeuralAmpModeler###"); // Don't change this!
  chunk.PutStr(header.Get());
  // Plugin version, so we can load legacy serialized states in the future!
  WDL_String version(PLUG_VERSION_STR);
  chunk.PutStr(version.Get());
  // The historical NAMPath slot is now the portable player-manifest path.
  // Fixed DessMetal amps still load from bundle resources and ignore it.
  const WDL_String namPath = _GetPlayerManifestPathSnapshot();
  const WDL_String irPath = _GetIRPathSnapshot();
  chunk.PutStr(namPath.Get());
  chunk.PutStr(irPath.Get()); // Still serialize IR path
  if (!ampPreset) return SerializeParams(chunk);
  // Keep the version 1 preset envelope readable by older players, without
  // including this player's input chain in a sound shared with someone else.
  for (int i = 0; i < kNumParams; ++i)
  {
    const auto* param = GetParam(i);
    const double value = _IsRigParam(i) ? param->GetDefault() : param->Value();
    if (chunk.Put(&value) <= 0) return false;
  }
  return true;
}

int NeuralAmpModeler::UnserializeState(const IByteChunk& chunk, int startPos)
{
  return _ReadPlayerState(chunk, startPos, false);
}

int NeuralAmpModeler::_ReadPlayerState(const IByteChunk& chunk, int startPos, bool ampPreset)
{
  try
  {
    // Look for the expected header. If it's there, then we'll know what to do.
    WDL_String header;
    int pos = GetBoundedChunkString(chunk, header, startPos, 64);
    if (pos < 0)
      return -1;

    const char* kExpectedHeader = "###NeuralAmpModeler###";
    if (strcmp(header.Get(), kExpectedHeader) == 0)
      return _UnserializeStateWithKnownVersion(chunk, pos, ampPreset);
    return _UnserializeStateWithUnknownVersion(chunk, startPos, ampPreset);
  }
  catch (const std::exception& e)
  {
    std::cerr << "Ignoring invalid DessMetal preset state: " << e.what() << std::endl;
    return -1;
  }
}

std::string NeuralAmpModeler::_DessPresetAmplifierIDForManifestPath(
  const std::filesystem::path& path) const
{
  if (path.empty())
    return {};
  try
  {
    const auto manifest = nam::player::LoadManifest(path);
    if (path.parent_path().lexically_normal() == (mDessPresetRootDirectory.parent_path() / "Amplifiers").lexically_normal())
      return "custom:" + manifest.artifact_id;
    if (!manifest.product_family.empty())
      return "family:" + manifest.product_family;
    if (!manifest.artifact_id.empty())
      return "artifact:" + manifest.artifact_id;
    if (!manifest.model_sha256.empty())
      return "model:" + manifest.model_sha256;
  }
  catch (const std::exception&)
  {
  }
  return {};
}

std::string NeuralAmpModeler::_DessPresetAmplifierIDFromState(
  const std::vector<std::uint8_t>& state, std::string& error) const
{
  error.clear();
  if (state.empty() || state.size() > static_cast<std::size_t>(std::numeric_limits<int>::max()))
  {
    error = "Preset contains no supported amplifier state";
    return {};
  }

  IByteChunk chunk;
  chunk.PutBytes(state.data(), static_cast<int>(state.size()));
  WDL_String value;
  int pos = GetBoundedChunkString(chunk, value, 0, 64);
  if (pos < 0 || strcmp(value.Get(), "###NeuralAmpModeler###") != 0)
  {
    error = "Preset does not contain a scoped DessAmp amplifier state";
    return {};
  }
  pos = GetBoundedChunkString(chunk, value, pos, 64);
  if (pos < 0)
  {
    error = "Preset is truncated before its state version";
    return {};
  }
  pos = GetBoundedChunkString(chunk, value, pos, 4096);
  if (pos < 0 || value.GetLength() == 0)
  {
    error = "Preset does not identify its amplifier package";
    return {};
  }

  const std::string amplifierId =
    _DessPresetAmplifierIDForManifestPath(std::filesystem::u8path(value.Get()));
  if (amplifierId.empty())
    error = "Preset amplifier package is unavailable or invalid";
  return amplifierId;
}

void NeuralAmpModeler::_SetDessPresetAmplifierScope(const std::string& amplifierId)
{
  if (amplifierId.empty() || amplifierId == mDessPresetAmplifierId)
    return;

  mDessPresetAmplifierId = amplifierId;
  mDessPresetDirectory =
    dessamp::preset::ScopedDirectory(mDessPresetRootDirectory, mDessPresetAmplifierId);
  mDessPresetPath.clear();
  mDessPresetLabel = "Model Default";
  mDessPresetModified = false;

  std::error_code ignored;
  if (!mDessPresetDirectory.empty())
    std::filesystem::create_directories(mDessPresetDirectory, ignored);
  _UpdateDessPresetUI();
}

void NeuralAmpModeler::_SyncDessPresetScopeToCurrentAmplifier()
{
  const WDL_String manifestPath = _GetPlayerManifestPathSnapshot();
  if (manifestPath.GetLength() == 0)
    return;
  _SetDessPresetAmplifierScope(
    _DessPresetAmplifierIDForManifestPath(std::filesystem::u8path(manifestPath.Get())));
}

void NeuralAmpModeler::_ConfirmRigChange(std::function<void()> change)
{
  const auto path = _GetPlayerManifestPathSnapshot();
  if (!mDessPresetModified || !path.GetLength() || !GetUI())
  {
    change();
    return;
  }
  if (mPendingRigChange) return;
  mPendingRigChange = std::move(change);
  mPendingRigPath = path.Get();
  auto* panel = GetUI()->GetControlWithTag(kCtrlTagChoicePanel)->As<DessAmpChoicePanel>();
  panel->Present("Save this sound?", "Keep your edits in the current amp's preset bank.",
    {"KEEP EDITING", "DISCARD", "SAVE"}, [this](int choice) {
      if (choice == 2 && mPendingRigPath == _GetPlayerManifestPathSnapshot().Get())
      {
        // Keep the prompt open if naming is cancelled or the write fails.
        GetUI()->GetControlWithTag(kCtrlTagPlayerManifest)->As<DessAmpModelSelectorControl>()->SavePreset();
      }
      else _FinishRigChange(choice == 1);
    });
}

void NeuralAmpModeler::_FinishRigChange(bool proceed)
{
  if (!mPendingRigChange) return;
  auto change = std::move(mPendingRigChange);
  mPendingRigChange = {};
  const bool sameRig = mPendingRigPath == _GetPlayerManifestPathSnapshot().Get();
  mPendingRigPath.clear();
  if (auto* ui = GetUI())
    if (auto* panel = ui->GetControlWithTag(kCtrlTagChoicePanel))
      panel->As<DessAmpChoicePanel>()->Dismiss();
  if (proceed && sameRig) change();
}

void NeuralAmpModeler::_ShowFeedback()
{
  auto* ui = GetUI();
  if (!ui || mPendingRigChange) return;
  bool copyOnly = false;
#ifdef OS_IOS
  copyOnly = IsOOPAuv3AppExtension();
#endif
  auto* panel = ui->GetControlWithTag(kCtrlTagChoicePanel)->As<DessAmpChoicePanel>();
  panel->Present("DessAmp feedback", copyOnly ? "Copy a destination to open outside this host." : "Use GitHub or email the developer directly.",
    {"CANCEL", copyOnly ? "COPY EMAIL" : "EMAIL", copyOnly ? "COPY ISSUE LINK" : "REPORT ISSUE"},
    [this, copyOnly](int choice) {
      auto* ui = GetUI();
      if (!ui) return;
      ui->GetControlWithTag(kCtrlTagChoicePanel)->As<DessAmpChoicePanel>()->Dismiss();
      if (choice == 0) return;
      const char* issue = "https://github.com/appdess/dessamp-player/issues/new?template=bug_report.yml";
      const char* email = "info@dessmetal.com";
      const bool mail = choice == 1;
      if (!copyOnly && ui->OpenURL(mail ? "mailto:info@dessmetal.com?subject=DessAmp%20feedback" : issue))
        return;
      const char* destination = mail ? email : issue;
      const bool copied = ui->SetTextInClipboard(destination);
      _ShowMessageBox(ui, destination, copied ? "Copied to clipboard" : "Feedback destination", kMB_OK);
    });
}

void NeuralAmpModeler::_SaveDessPreset(const char* path, bool createNew)
{
  if (path == nullptr || path[0] == '\0')
    return;

  _SyncDessPresetScopeToCurrentAmplifier();
  if (mDessPresetAmplifierId.empty() || mDessPresetDirectory.empty())
  {
    if (auto* ui = GetUI())
      _ShowMessageBox(ui, "Load an amplifier before saving a preset.", "Preset was not saved", kMB_OK);
    return;
  }

  std::error_code directoryError;
  std::filesystem::create_directories(mDessPresetDirectory, directoryError);
  if (directoryError)
  {
    if (auto* ui = GetUI())
      _ShowMessageBox(ui, "The amplifier preset directory could not be created.", "Preset was not saved", kMB_OK);
    return;
  }
  const std::string label = createNew ? dessamp::preset::UserLabel(path) : path;
  const std::filesystem::path requested = label.empty() ? std::filesystem::path{} :
    dessamp::preset::ScopedSavePath(mDessPresetDirectory, std::filesystem::u8path(label));
  if (requested.empty())
  {
    if (auto* ui = GetUI())
      _ShowMessageBox(ui, "Choose a name other than Model Default, without path separators or punctuation used in filenames.", "Preset was not saved", kMB_OK);
    return;
  }

  IByteChunk state;
  if (!_SerializePlayerState(state, true) || state.Size() < 0)
  {
    if (auto* ui = GetUI())
      _ShowMessageBox(ui, "The current amplifier state could not be serialized.", "Preset was not saved", kMB_OK);
    return;
  }

  std::filesystem::path savedPath;
  std::string error;
  if (!dessamp::preset::WriteAtomic(
        requested, state.GetData(), static_cast<std::size_t>(state.Size()), savedPath, error, createNew))
  {
    if (auto* ui = GetUI())
      _ShowMessageBox(ui, error.c_str(), "Preset was not saved", kMB_OK);
    return;
  }

  _SelectNamedPreset(savedPath);
  if (mPendingRigChange) _FinishRigChange(true);
}

bool NeuralAmpModeler::_RecallDessPreset(const std::filesystem::path& path, std::string& error, bool allowRelink)
{
  error.clear();
  auto decoded = dessamp::preset::Read(path);
  if (!decoded)
  {
    error = decoded.error;
    return false;
  }

  // iOS may assign a different bundle/container UUID on update; the Mac
  // standalone and AU also have separate factory-resource paths. Rebind only
  // recognized resources in our own per-amp bank. Leave external user IRs alone.
  if (path.parent_path().lexically_normal() == mDessPresetDirectory.lexically_normal()
      && decoded.state.size() <= static_cast<std::size_t>(std::numeric_limits<int>::max()))
  {
    IByteChunk chunk;
    chunk.PutBytes(decoded.state.data(), static_cast<int>(decoded.state.size()));
    WDL_String header, version, model, ir;
    int pos = GetBoundedChunkString(chunk, header, 0, 64);
    pos = GetBoundedChunkString(chunk, version, pos, 64);
    pos = GetBoundedChunkString(chunk, model, pos, 4096);
    pos = GetBoundedChunkString(chunk, ir, pos, 4096);
    if (pos >= 0 && !strcmp(header.Get(), "###NeuralAmpModeler###"))
    {
      const auto original = std::filesystem::u8path(model.Get());
      const auto speaker = std::filesystem::u8path(ir.Get());
      std::error_code ec;
      if (!original.empty() && !std::filesystem::is_regular_file(original, ec)
          && original.parent_path().filename() == "Amplifiers")
      {
        std::vector<std::filesystem::path> candidates;
        for (const auto& factory : mBundledAmplifierPaths) candidates.emplace_back(factory);
        candidates.push_back(mDessPresetRootDirectory.parent_path() / "Amplifiers" / original.filename());
        for (const auto& candidate : candidates)
          if (candidate.filename() == original.filename() && std::filesystem::is_regular_file(candidate, ec)
              && _DessPresetAmplifierIDForManifestPath(candidate) == mDessPresetAmplifierId)
          {
            model.Set(candidate.u8string().c_str());
            break;
          }
      }
      if (!speaker.empty() && speaker.parent_path().filename() == "IRs"
          && !std::filesystem::is_regular_file(speaker, ec))
        for (const auto& choice : GetBundledIRChoices(GetBundleID()))
          if (std::filesystem::u8path(choice.second).filename() == speaker.filename())
          {
            ir.Set(choice.second.c_str());
            break;
          }
      if (model.Get() != original.u8string() || ir.Get() != speaker.u8string())
      {
        IByteChunk rebased;
        rebased.PutStr(header.Get()); rebased.PutStr(version.Get());
        rebased.PutStr(model.Get()); rebased.PutStr(ir.Get());
        rebased.PutBytes(chunk.GetData() + pos, chunk.Size() - pos);
        decoded.state.assign(rebased.GetData(), rebased.GetData() + rebased.Size());
      }
    }
  }
  std::string scopeError;
  std::string storedAmplifierId = _DessPresetAmplifierIDFromState(decoded.state, scopeError);
  if (storedAmplifierId.empty())
  {
    // A preset whose original package has moved cannot be identified by
    // opening that package. Preserve the existing relink workflow only for a
    // preset already stored inside the active amplifier's scoped library. The
    // directory itself is the durable amplifier boundary; an arbitrary file
    // selected from elsewhere must never be allowed to cross that boundary.
    const auto selectedDirectory = path.parent_path().lexically_normal();
    const auto activeDirectory = mDessPresetDirectory.lexically_normal();
    if (allowRelink && scopeError == "Preset amplifier package is unavailable or invalid"
        && !activeDirectory.empty() && selectedDirectory == activeDirectory)
      storedAmplifierId = mDessPresetAmplifierId;
    else
    {
      error = scopeError;
      return false;
    }
  }
  if (mDessPresetAmplifierId.empty() || storedAmplifierId != mDessPresetAmplifierId)
  {
    error = "This preset belongs to another amplifier. Select that amplifier before loading it.";
    return false;
  }

  IByteChunk snapshot;
  if (!SerializeState(snapshot))
  {
    error = "The current state could not be snapshotted safely";
    return false;
  }

  IByteChunk incoming;
  if (!decoded.state.empty())
    incoming.PutBytes(decoded.state.data(), static_cast<int>(decoded.state.size()));
  const int end = _ReadPlayerState(incoming, 0, true);
  if (end != incoming.Size())
  {
    const int rollbackEnd = _ReadPlayerState(snapshot, 0, true);
    error = "Preset state is invalid or was not consumed completely";
    if (rollbackEnd != snapshot.Size())
      error += "; the previous state could not be restored completely";
    OnRestoreState();
    return false;
  }

  OnRestoreState();
  _UpdatePlayerModeUI();
  _UpdatePlayerControls();
  _SelectNamedPreset(path);
  return true;
}

void NeuralAmpModeler::_SelectNamedPreset(const std::filesystem::path& path)
{
  mDessPresetPath = path.lexically_normal();
  mLastPresetByAmp[mDessPresetAmplifierId] = mDessPresetPath;
  mDessPresetLabel = dessamp::preset::Label(mDessPresetPath);
  mDessPresetModified = false;
  _UpdateDessPresetUI();
}

bool NeuralAmpModeler::_RestoreLastAmpPreset(const std::filesystem::path& amplifierPath)
{
  const auto scope = _DessPresetAmplifierIDForManifestPath(amplifierPath);
  if (scope.empty()) return false;
  _SetDessPresetAmplifierScope(scope);
  const auto found = mLastPresetByAmp.find(mDessPresetAmplifierId);
  if (found == mLastPresetByAmp.end()) return false;
  std::string error;
  if (_RecallDessPreset(found->second, error, false)) return true;
  // A deleted or unreadable preset falls back to this amp's normal defaults.
  mLastPresetByAmp.erase(found);
  return false;
}

void NeuralAmpModeler::_LoadDessPreset(const char* path)
{
  if (!path) return;
  _ConfirmRigChange([this, selected = std::string(path)] { _ApplyDessPreset(selected); });
}

void NeuralAmpModeler::_ApplyDessPreset(const std::string& path)
{
  _SyncDessPresetScopeToCurrentAmplifier();
  if (path.empty())
  {
    const auto manifest = std::filesystem::u8path(_GetPlayerManifestPathSnapshot().Get());
    if (!_BundledFactoryDefaults(manifest)) return;
    mLastPresetByAmp.erase(mDessPresetAmplifierId);
    _ApplyBundledFactoryDefaults(manifest, true);
    _SetPlayerManifestPathAndRequest(manifest.u8string().c_str(), true);
    mDessPresetPath.clear();
    mDessPresetLabel = "Model Default";
    mDessPresetModified = false;
    _UpdateDessPresetUI();
    return;
  }

  const std::filesystem::path selected(path);
  std::string error;
  if (!_RecallDessPreset(selected, error))
  {
    if (auto* ui = GetUI())
      _ShowMessageBox(ui, error.c_str(), "Preset was not loaded", kMB_OK);
    _UpdateDessPresetUI();
  }
}

void NeuralAmpModeler::_NavigateDessPreset(const int delta)
{
  const auto files = dessamp::preset::Enumerate(mDessPresetDirectory);
  const auto manifestPath = std::filesystem::u8path(_GetPlayerManifestPathSnapshot().Get());
  if (_BundledFactoryDefaults(manifestPath) && delta != 0)
  {
    const auto selected = std::find(files.begin(), files.end(), mDessPresetPath);
    const int current = selected == files.end() ? 0 : static_cast<int>(selected - files.begin()) + 1;
    const int count = static_cast<int>(files.size()) + 1;
    const int next = (current + (delta < 0 ? -1 : 1) + count) % count;
    if (next == 0)
      _LoadDessPreset("");
    else
      _LoadDessPreset(files[static_cast<std::size_t>(next - 1)].string().c_str());
    return;
  }
  const auto adjacent = dessamp::preset::Adjacent(files, mDessPresetPath, delta);
  if (adjacent)
    _LoadDessPreset(adjacent->string().c_str());
}

int NeuralAmpModeler::_CurrentBundledAmplifierIndex() const
{
  const std::filesystem::path current(_GetPlayerManifestPathSnapshot().Get());
  if (current.empty())
    return -1;
  const auto normalizedCurrent = current.lexically_normal();
  for (std::size_t index = 0; index < mBundledAmplifierPaths.size(); ++index)
  {
    if (std::filesystem::path(mBundledAmplifierPaths[index]).lexically_normal() == normalizedCurrent)
      return static_cast<int>(index);
  }
  return -1;
}

int NeuralAmpModeler::_CurrentBundledAmplifierProductIndex() const
{
  const int current = _CurrentBundledAmplifierIndex();
  if (current < 0)
    return -1;
  const std::string& currentFamily = mBundledAmplifierProductFamilies[static_cast<std::size_t>(current)];
  for (std::size_t productIndex = 0; productIndex < mBundledAmplifierProductRootIndices.size(); ++productIndex)
  {
    const std::size_t root = mBundledAmplifierProductRootIndices[productIndex];
    if (!currentFamily.empty())
    {
      if (mBundledAmplifierProductFamilies[root] == currentFamily)
        return static_cast<int>(productIndex);
    }
    else if (root == static_cast<std::size_t>(current))
      return static_cast<int>(productIndex);
  }
  return -1;
}

std::string NeuralAmpModeler::_BundledAmplifierVariantPath(const std::string& productFamily,
                                                           const std::string& productVariant) const
{
  if (productFamily.empty() || productVariant.empty())
    return "";
  for (std::size_t index = 0; index < mBundledAmplifierPaths.size(); ++index)
  {
    if (mBundledAmplifierProductFamilies[index] == productFamily
        && mBundledAmplifierProductVariants[index] == productVariant)
      return mBundledAmplifierPaths[index];
  }
  return "";
}

void NeuralAmpModeler::_SelectBundledAmplifierVariant(const std::string& productVariant)
{
  nam::player::Manifest activeManifest;
  bool haveManifest = false;
  {
    std::lock_guard<std::mutex> lock(mDSPLoadMutex);
    if (auto* model = mModel.load(std::memory_order_acquire))
    {
      if (model->IsPreparedPlayerMode() && model->GetPlayerManifest() != nullptr)
      {
        activeManifest = *model->GetPlayerManifest();
        haveManifest = true;
      }
    }
    if (!haveManifest && mStagedPlayerManifestForUI != nullptr)
    {
      activeManifest = *mStagedPlayerManifestForUI;
      haveManifest = true;
    }
  }
  if (!haveManifest || activeManifest.product_family.empty() || activeManifest.product_variant == productVariant)
    return;
  const std::string target = _BundledAmplifierVariantPath(activeManifest.product_family, productVariant);
  if (target.empty())
    return;
  _ConfirmRigChange([this, target] {
    // Variant changes share the ordinary player/package load path. Restore the
    // player and amp gates just as catalog navigation does so a saved bypassed
    // or legacy state cannot make the visible selector appear ineffective.
    if (!GetParam(kPlayerMode)->Bool())
      _SetParameterValueFromMainThread(kPlayerMode, 1.0);
    if (!GetParam(kNAMActive)->Bool())
      _SetParameterValueFromMainThread(kNAMActive, 1.0);
    _SetPlayerManifestPathAndRequest(target.c_str(), true);
    mDessPresetModified = true;
    _UpdateDessPresetUI();
  });
}

void NeuralAmpModeler::_LoadBundledFactoryDefaults()
{
  mBundledFactoryDefaults.resize(mBundledAmplifierPaths.size());
  mBundledFactoryDefaultsValid.assign(mBundledAmplifierPaths.size(), false);
  WDL_String resourcePath;
  PlayerResourcePath(resourcePath, GetBundleID());
  if (!resourcePath.GetLength())
    return;
  const std::filesystem::path resources(resourcePath.Get());
  auto path = resources / "FactoryDefaults.json";
  if (!std::filesystem::exists(path))
    path = resources / "PrivateValidation" / "FactoryDefaults.json";
  if (!std::filesystem::exists(path))
    return;
  try
  {
    if (std::filesystem::file_size(path) > 65536U)
      throw std::runtime_error("Factory defaults exceed size limit");
    std::ifstream input(path);
    const auto document = nlohmann::json::parse(input);
    if (document.at("schema_version") != 1 || !document.at("amplifiers").is_array()
        || document.at("amplifiers").size() != mBundledAmplifierPaths.size())
      throw std::runtime_error("Factory defaults do not match the bundled catalogue");
    const auto choices = GetBundledIRChoices(GetBundleID());
    std::map<std::filesystem::path, std::string> cabinets;
    // Validate the whole bank before publishing any entry.
    for (std::size_t index = 0; index < mBundledAmplifierPaths.size(); ++index)
    {
      const auto manifest = nam::player::LoadManifest(mBundledAmplifierPaths[index]);
      const auto& entry = document.at("amplifiers").at(index);
      const auto cabinet = entry.value("cabinet", std::string("Desfire.wav"));
      const auto choice = std::find_if(choices.begin(), choices.end(), [&](const BundledIRChoice& ir) {
        return std::filesystem::path(ir.second).filename() == cabinet;
      });
      if (choice == choices.end()) throw std::runtime_error("Factory default cabinet is unavailable: " + cabinet);
      cabinets.emplace(std::filesystem::path(mBundledAmplifierPaths[index]).lexically_normal(), choice->second);
      const auto& values = entry.at("parameters");
      if (entry.at("artifact_id") != manifest.artifact_id || entry.at("model_sha256") != manifest.model_sha256
          || !values.is_array() || (values.size() != kOxideLevel && values.size() != kNumParams))
        throw std::runtime_error("Factory default identity mismatch");
      for (int param = 0; param < kNumParams; ++param)
      {
        // The original factory profiles have no per-pedal trim.
        const double value = static_cast<size_t>(param) < values.size() ? values.at(param).get<double>() : 0.0;
        if (!std::isfinite(value) || value < GetParam(param)->GetMin() || value > GetParam(param)->GetMax())
          throw std::runtime_error("Factory default parameter is outside its host range");
        mBundledFactoryDefaults[index][param] = value;
      }
      if (mBundledFactoryDefaults[index][kNAMActive] != 1.0
          || mBundledFactoryDefaults[index][kPlayerMode] != 1.0
          || mBundledFactoryDefaults[index][kIRToggle] != 1.0)
        throw std::runtime_error("Factory defaults must enable the player and cabinet");
    }
    mFactoryCabinetByAmp = std::move(cabinets);
    mBundledFactoryDefaultsValid.assign(mBundledAmplifierPaths.size(), true);
  }
  catch (const std::exception& error)
  {
    std::cerr << "Factory defaults unavailable: " << error.what() << std::endl;
  }
}

const std::array<double, kNumParams>* NeuralAmpModeler::_BundledFactoryDefaults(
  const std::filesystem::path& path) const
{
  for (std::size_t index = 0; index < mBundledAmplifierPaths.size(); ++index)
    if (index < mBundledFactoryDefaultsValid.size() && mBundledFactoryDefaultsValid[index]
        && path.lexically_normal() == std::filesystem::path(mBundledAmplifierPaths[index]).lexically_normal())
      return &mBundledFactoryDefaults[index];
  return nullptr;
}

std::string NeuralAmpModeler::_DefaultCabinetForAmp(const std::filesystem::path& path) const
{
  const auto found = mFactoryCabinetByAmp.find(path.lexically_normal());
  return found == mFactoryCabinetByAmp.end() ? GetBundledDefaultIRPath(GetBundleID()) : found->second;
}

void NeuralAmpModeler::_ApplyBundledFactoryDefaults(const std::filesystem::path& path, const bool notifyHost)
{
  const auto* defaults = _BundledFactoryDefaults(path);
  if (!defaults)
    return;
  for (int param = 0; param < kNumParams; ++param)
  {
    // The legacy selector is not the conditioned catalogue index.
    if (param == kAmpModel || _IsRigParam(param))
      continue;
    if (notifyHost)
      _SetParameterValueFromMainThread(param, GetParam(param)->ToNormalized(defaults->at(param)));
    else
      GetParam(param)->Set(defaults->at(param));
  }
  if (!notifyHost)
  {
    mBoostModelIdx.store(GetParam(kBoostModel)->Int(), std::memory_order_relaxed);
    mBoostActiveTarget.store(GetParam(kBoostActive)->Bool(), std::memory_order_relaxed);
    for (int index = 0; index < kNumPlayerConditionParams; ++index)
      mPlayerConditionTargets[index].store(defaults->at(kFirstPlayerConditionParam + index) / 100.0,
                                           std::memory_order_relaxed);
  }
  _SetIRPathAndRequest(_DefaultCabinetForAmp(path).c_str());
}

int NeuralAmpModeler::_AmplifierStripIndex() const
{
  const int factory = _CurrentBundledAmplifierProductIndex();
  if (factory >= 0 || !_GetPlayerManifestPathSnapshot().GetLength())
    return factory;
  return static_cast<int>(mBundledAmplifierProductRootIndices.size());
}

void NeuralAmpModeler::_SelectCustomAmplifier()
{
  auto* ui = GetUI();
  if (!ui)
    return;
  if (!mLastCustomAmpPath.empty() && _CurrentBundledAmplifierIndex() >= 0)
  {
    std::error_code error;
    if (std::filesystem::is_regular_file(mLastCustomAmpPath, error))
    {
      _LoadCustomAmplifier(mLastCustomAmpPath);
      return;
    }
  }
  WDL_String fileName, directory;
  ui->PromptForFile(fileName, directory, EFileAction::Open, "dess",
    [this](const WDL_String& selected, const WDL_String&) {
      if (selected.GetLength())
        _LoadCustomAmplifier(std::filesystem::u8path(selected.Get()));
    });
}

void NeuralAmpModeler::_LoadCustomAmplifier(const std::filesystem::path& path)
{
  _ConfirmRigChange([this, path] { _ApplyCustomAmplifier(path); });
}

void NeuralAmpModeler::_ApplyCustomAmplifier(const std::filesystem::path& path)
{
  try
  {
    auto imported = std::make_unique<nam::player::ImportedAmplifier>(
      nam::player::ImportDessPackage(path, mDessPresetRootDirectory.parent_path() / "Amplifiers"));
    const auto libraryPath = imported->manifest.manifest_path;
    // Only change the active rig after the package and its neural runtime pass validation.
    if (!GetParam(kPlayerMode)->Bool())
      _SetParameterValueFromMainThread(kPlayerMode, 1.0);
    if (!GetParam(kNAMActive)->Bool())
      _SetParameterValueFromMainThread(kNAMActive, 1.0);
    {
      std::lock_guard<std::mutex> lock(mDSPLoadMutex);
      mImportedAmp = std::move(imported);
      _SetPlayerManifestPathAndRequest(libraryPath.u8string().c_str(), true);
    }
    mLastCustomAmpPath = libraryPath;
    if (_RestoreLastAmpPreset(libraryPath)) return;
    mDessPresetPath.clear();
    mDessPresetLabel = "Model Default";
    mDessPresetModified = false;
    _UpdateDessPresetUI();
    _UpdatePlayerModeUI();
  }
  catch (const std::exception& error)
  {
    if (path == mLastCustomAmpPath)
      mLastCustomAmpPath.clear();
    if (auto* ui = GetUI())
      _ShowMessageBox(ui, error.what(), "Could not load custom amplifier", kMB_OK);
  }
}

void NeuralAmpModeler::_SelectBundledAmplifier(const int next)
{
  if (next < 0 || next >= static_cast<int>(mBundledAmplifierProductRootIndices.size())
      || (next == _CurrentBundledAmplifierProductIndex() && GetParam(kPlayerMode)->Bool() && GetParam(kNAMActive)->Bool()))
    return;
  _ConfirmRigChange([this, next] { _ApplyBundledAmplifier(next); });
}

void NeuralAmpModeler::_ApplyBundledAmplifier(const int next)
{
  const auto previous = _GetPlayerManifestPathSnapshot();
  if (previous.GetLength() && _CurrentBundledAmplifierIndex() < 0)
  {
    const auto path = std::filesystem::u8path(previous.Get());
    if (path.extension() == ".dess")
      mLastCustomAmpPath = path;
  }
  const std::size_t target = mBundledAmplifierProductRootIndices[static_cast<std::size_t>(next)];
  if (_RestoreLastAmpPreset(mBundledAmplifierPaths[target])) return;
  if (!GetParam(kPlayerMode)->Bool())
    _SetParameterValueFromMainThread(kPlayerMode, 1.0);
  if (!GetParam(kNAMActive)->Bool())
    _SetParameterValueFromMainThread(kNAMActive, 1.0);
  _ApplyBundledFactoryDefaults(mBundledAmplifierPaths[target], true);
  _SetPlayerManifestPathAndRequest(mBundledAmplifierPaths[target].c_str(), true);
  mDessPresetPath.clear();
  mDessPresetLabel = "Model Default";
  mDessPresetModified = false;
  _UpdateDessPresetUI();
}

void NeuralAmpModeler::_UpdateDessPresetUI()
{
  auto* ui = GetUI();
  if (ui == nullptr)
    return;
  auto* raw = ui->GetControlWithTag(kCtrlTagPlayerManifest);
  if (raw == nullptr)
    return;

  std::string directory = mDessPresetDirectory.string();
  if (!directory.empty() && directory.back() != std::filesystem::path::preferred_separator)
    directory.push_back(std::filesystem::path::preferred_separator);
  const bool hasDefault = _BundledFactoryDefaults(std::filesystem::u8path(_GetPlayerManifestPathSnapshot().Get())) != nullptr;
  raw->As<DessAmpModelSelectorControl>()->SetPresetState(
    mDessPresetLabel.c_str(), mDessPresetPath.u8string().c_str(), directory.c_str(), hasDefault, mDessPresetModified);
}

void NeuralAmpModeler::_ShowTuner(bool open)
{
  if (auto* ui = GetUI())
    if (auto* panel = ui->GetControlWithTag(kCtrlTagTunerPanel))
    {
      mTuner.Enable(open);
      panel->As<DessAmpTunerPanel>()->Clear();
      panel->Hide(!open);
      ui->SetAllControlsDirty();
    }
}

void NeuralAmpModeler::OnUIClose()
{
  _SaveStandaloneRig(true);
  _FinishRigChange(false);
  _ShowTuner(false);
  mTuner.Enable(false);
  Plugin::OnUIClose();
}

void NeuralAmpModeler::OnUIOpen()
{
  Plugin::OnUIOpen();
  _SyncDessPresetScopeToCurrentAmplifier();
  _UpdatePlayerModeUI();
  _UpdateDessPresetUI();

  // Map the historical five-state host parameter onto the four distinct
  // visible models. State 3 remains a legacy alias for the red capture.
  if (auto* ampModelControl = GetUI()->GetControlWithTag(kCtrlTagAmpModel))
  {
    const int modelIndex = GetParam(kAmpModel)->Int();
    ampModelControl->SetValueFromDelegate(GetVisibleAmpModelValue(modelIndex));
  }

  // Note: Model file browser control was removed in favor of amp model selector
  const WDL_String irPath = _GetIRPathSnapshot();
  if (irPath.GetLength() && GetUI()->GetControlWithTag(kCtrlTagIRFileBrowser))
  {
    SendControlMsgFromDelegate(kCtrlTagIRFileBrowser, kMsgTagLoadedIR, irPath.GetLength(), irPath.Get());
    if (mIR.load(std::memory_order_acquire) == nullptr && mPendingIR.load(std::memory_order_acquire) == nullptr
        && mHandledIRLoadRequest.load(std::memory_order_acquire) == mIRLoadRequest.load(std::memory_order_acquire))
      SendControlMsgFromDelegate(kCtrlTagIRFileBrowser, kMsgTagLoadFailed);
  }

  if (mModel.load(std::memory_order_acquire) != nullptr)
  {
    _UpdateControlsFromModel();
    _UpdatePlayerControls();
  }

  const WDL_String playerManifestPath = _GetPlayerManifestPathSnapshot();
  if (playerManifestPath.GetLength() && GetUI()->GetControlWithTag(kCtrlTagPlayerManifest))
    SendControlMsgFromDelegate(
      kCtrlTagPlayerManifest, kMsgTagLoadedPlayerManifest, playerManifestPath.GetLength(), playerManifestPath.Get());

  const WDL_String boostPath = _GetBoostNAMPathSnapshot();
  if (boostPath.GetLength() && GetUI()->GetControlWithTag(kCtrlTagBoostModelFileBrowser))
  {
    SendControlMsgFromDelegate(
      kCtrlTagBoostModelFileBrowser, kMsgTagLoadedBoostModel, boostPath.GetLength(), boostPath.Get());
    if (mBoostModel.load(std::memory_order_acquire) == nullptr
        && mPendingBoostModel.load(std::memory_order_acquire) == nullptr)
      SendControlMsgFromDelegate(kCtrlTagBoostModelFileBrowser, kMsgTagLoadFailed);
  }
}

void NeuralAmpModeler::OnParamChange(int paramIdx)
{
  if (paramIdx >= kFirstPlayerConditionParam && paramIdx < kFirstPlayerConditionParam + kNumPlayerConditionParams)
  {
    const auto conditionIndex = static_cast<std::size_t>(paramIdx - kFirstPlayerConditionParam);
    mPlayerConditionTargets[conditionIndex].store(
      std::clamp(GetParam(paramIdx)->Value() / 100.0, 0.0, 1.0), std::memory_order_release);
    return;
  }

  switch (paramIdx)
  {
    // Load model when amp model changes
    case kAmpModel:
    {
      int ampModelInt = GetParam(kAmpModel)->Int();
      mAmpModelIdx = ampModelInt; // Update atomic index for audio thread
      mAmpRetryNeeded.store(false, std::memory_order_release);

      // Parameter callbacks may execute on the render thread. Loading is
      // coalesced and performed from OnIdle(); this callback remains wait-free.
      mAmpLoadRequest.fetch_add(1, std::memory_order_release);

      break;
    }
    case kGain:
    {
      double gainVal = GetParam(kGain)->Value();
      // Direct mapping: knob 2 = param 0.2, knob 8 = param 0.8 (matches training data)
      // Clamp to training range [0.2, 0.8] to avoid extrapolation
      mTargetGain.store(std::clamp(gainVal / 10.0, 0.2, 0.8), std::memory_order_relaxed);
      break;
    }
    case kPlayerMode:
    {
      mPlayerModeTarget.store(GetParam(kPlayerMode)->Bool(), std::memory_order_release);
      mAmpRetryNeeded.store(false, std::memory_order_release);
      mApplyPlayerDefaultsRequest.store(0U, std::memory_order_release);
      mAmpLoadRequest.fetch_add(1, std::memory_order_release);
      break;
    }
    case kBoostModel:
    {
      mBoostModelIdx.store(GetParam(kBoostModel)->Int(), std::memory_order_release);
      if (mBoostActiveTarget.load(std::memory_order_acquire))
        mBoostLoadRequest.fetch_add(1, std::memory_order_release);
      break;
    }
    case kBoostActive:
    {
      mBoostActiveTarget.store(GetParam(kBoostActive)->Bool(), std::memory_order_release);
      mBoostLoadRequest.fetch_add(1, std::memory_order_release);
      break;
    }
    case kNAMActive:
    {
      const bool active = GetParam(kNAMActive)->Bool();
      mAmpActiveTarget.store(active, std::memory_order_release);
      // Ordinary bypass automation never reloads. Only a user attempt to
      // recover from a prior verified load failure schedules another load.
      if (active && mAmpRetryNeeded.exchange(false, std::memory_order_acq_rel))
        mAmpLoadRequest.fetch_add(1, std::memory_order_release);
      _SetInputGain();
      _SetOutputGain();
      break;
    }
    case kIRToggle:
    {
      mIRActiveTarget.store(GetParam(kIRToggle)->Bool(), std::memory_order_release);
      mIRLoadRequest.fetch_add(1, std::memory_order_release);
      break;
    }

    // Changes to the input gain
    case kCalibrateInput:
    case kInputCalibrationLevel:
    case kInputLevel: _SetInputGain(); break;
    // Changes to the output gain
    case kOutputLevel: _SetOutputGain(); break;
    case kOutputMode:
      _SetOutputGain();
      break;
    // Tone stack:
    case kToneBass: mToneBassTarget.store(GetParam(paramIdx)->Value(), std::memory_order_release); break;
    case kToneMid: mToneMidTarget.store(GetParam(paramIdx)->Value(), std::memory_order_release); break;
    case kToneTreble: mToneTrebleTarget.store(GetParam(paramIdx)->Value(), std::memory_order_release); break;
    default: break;
  }
}

void NeuralAmpModeler::OnParamChangeUI(int paramIdx, EParamSource source)
{
  if (!_IsRigParam(paramIdx) && (source == EParamSource::kUI || source == EParamSource::kHost))
  {
    mDessPresetModified = true;
    _UpdateDessPresetUI();
  }
  if (auto pGraphics = GetUI())
  {
    bool active = GetParam(paramIdx)->Bool();

    switch (paramIdx)
    {
      case kNoiseGateActive:
        // Keep threshold editable while bypassed so the next engage is
        // predictable, and so the persistent strip and PRE mirror agree.
        break;
      case kEQActive:
        pGraphics->ForControlInGroup("EQ_KNOBS", [active](IControl* pControl) { pControl->SetDisabled(!active); });
        break;
      case kBoostActive:
        // Drive settings remain editable while bypassed.
        break;
      case kIRToggle:
        if (auto* cabinet = pGraphics->GetControlWithTag(kCtrlTagIRFileBrowser))
          cabinet->SetDirty(false);
        break;
      case kPlayerMode: _UpdatePlayerModeUI(); break;
      case kAmpModel:
        // The historical selector remains host-compatible but is intentionally
        // absent from the independent conditioned-player UI.
        break;

      default: break;
    }

    const int morphConditionIndex = mPlayerMorphConditionIndex.load(std::memory_order_acquire);
    if (morphConditionIndex >= 0 && morphConditionIndex < kNumPlayerConditionParams
        && paramIdx == kFirstPlayerConditionParam + morphConditionIndex)
    {
      if (auto* head = pGraphics->GetControlWithTag(kCtrlTagPlayerHead))
        head->As<DessAmpHeadControl>()->SetMorphValue(GetParam(paramIdx)->GetNormalized());
    }
  }
}

bool NeuralAmpModeler::OnMessage(int msgTag, int ctrlTag, int dataSize, const void* pData)
{
  switch (msgTag)
  {
    case kMsgTagClearModel:
      mAmpActiveTarget.store(false, std::memory_order_release);
      _SetParameterValueFromMainThread(kNAMActive, 0.0);
      mHandledAmpLoadRequest.store(
        mAmpLoadRequest.fetch_add(1, std::memory_order_acq_rel) + 1, std::memory_order_release);
      delete mPendingModel.exchange(nullptr, std::memory_order_acq_rel);
      mShouldRemoveModel = true;
      return true;
    case kMsgTagClearIR:
      mIRActiveTarget.store(false, std::memory_order_release);
      _SetParameterValueFromMainThread(kIRToggle, 0.0);
      {
        std::lock_guard<std::mutex> lock(mPathMutex);
        mIRPath.Set("");
        mHandledIRLoadRequest.store(mIRLoadRequest.load(std::memory_order_acquire), std::memory_order_release);
      }
      delete mPendingIR.exchange(nullptr, std::memory_order_acq_rel);
      mShouldRemoveIR = true;
      return true;
    case kMsgTagClearBoostModel:
      mBoostActiveTarget.store(false, std::memory_order_release);
      _SetParameterValueFromMainThread(kBoostActive, 0.0);
      mHandledBoostLoadRequest.store(mBoostLoadRequest.load(std::memory_order_acquire), std::memory_order_release);
      delete mPendingBoostModel.exchange(nullptr, std::memory_order_acq_rel);
      mShouldRemoveBoostModel = true;
      return true;
    case kMsgTagClearPlayerManifest:
      _SetPlayerManifestPath("");
      mApplyPlayerDefaultsRequest.store(0U, std::memory_order_release);
      _SetParameterValueFromMainThread(kPlayerMode, 0.0);
      _UpdatePlayerModeUI();
      return true;
    case kMsgTagHighlightColor:
    {
      if (pData == nullptr || dataSize <= 0 || dataSize > 32)
        return false;
      std::string colorCode(static_cast<const char*>(pData), static_cast<size_t>(dataSize));
      if (!colorCode.empty() && colorCode.back() == '\0')
        colorCode.pop_back();
      if (colorCode.empty())
        return false;
      mHighLightColor.Set(colorCode.c_str());

      if (GetUI())
      {
        GetUI()->ForStandardControlsFunc([&](IControl* pControl) {
          if (auto* pVectorBase = pControl->As<IVectorBase>())
          {
            IColor color = IColor::FromColorCodeStr(mHighLightColor.Get());

            pVectorBase->SetColor(kX1, color);
            pVectorBase->SetColor(kPR, color.WithOpacity(0.3f));
            pVectorBase->SetColor(kFR, color.WithOpacity(0.4f));
            pVectorBase->SetColor(kX3, color.WithContrast(0.1f));
          }
          pControl->GetUI()->SetAllControlsDirty();
        });
      }

      return true;
    }
    default: return false;
  }
}

// Private methods ============================================================

WDL_String NeuralAmpModeler::_GetNAMPathSnapshot() const
{
  std::lock_guard<std::mutex> lock(mPathMutex);
  return mNAMPath;
}

WDL_String NeuralAmpModeler::_GetPlayerManifestPathSnapshot() const
{
  std::lock_guard<std::mutex> lock(mPathMutex);
  return mPlayerManifestPath;
}

WDL_String NeuralAmpModeler::_GetIRPathSnapshot() const
{
  std::lock_guard<std::mutex> lock(mPathMutex);
  return mIRPath;
}

WDL_String NeuralAmpModeler::_GetBoostNAMPathSnapshot() const
{
  std::lock_guard<std::mutex> lock(mPathMutex);
  return mBoostNAMPath;
}

void NeuralAmpModeler::_SetNAMPath(const char* path)
{
  std::lock_guard<std::mutex> lock(mPathMutex);
  mNAMPath.Set(path == nullptr ? "" : path);
}

void NeuralAmpModeler::_SetPlayerManifestPath(const char* path)
{
  std::lock_guard<std::mutex> lock(mPathMutex);
  mPlayerManifestPath.Set(path == nullptr ? "" : path);
}

void NeuralAmpModeler::_SetPlayerManifestPathAndRequest(const char* path, const bool applyManifestDefaults)
{
  std::lock_guard<std::mutex> lock(mPathMutex);
  mPlayerManifestPath.Set(path == nullptr ? "" : path);
  const unsigned int request = mAmpLoadRequest.fetch_add(1, std::memory_order_acq_rel) + 1U;
  mApplyPlayerDefaultsRequest.store(applyManifestDefaults ? request : 0U, std::memory_order_release);
}

void NeuralAmpModeler::_SetIRPath(const char* path)
{
  std::lock_guard<std::mutex> lock(mPathMutex);
  mIRPath.Set(path == nullptr ? "" : path);
}

void NeuralAmpModeler::_SetIRPathAndRequest(const char* path)
{
  std::lock_guard<std::mutex> lock(mPathMutex);
  mIRPath.Set(path == nullptr ? "" : path);
  mIRLoadRequest.fetch_add(1, std::memory_order_release);
}

void NeuralAmpModeler::_RequestIRReload()
{
  std::lock_guard<std::mutex> lock(mPathMutex);
  mIRLoadRequest.fetch_add(1, std::memory_order_release);
}

void NeuralAmpModeler::_SetBoostNAMPath(const char* path)
{
  std::lock_guard<std::mutex> lock(mPathMutex);
  mBoostNAMPath.Set(path == nullptr ? "" : path);
}

void NeuralAmpModeler::_ApplyDSPStaging(const bool ampActiveForBlock)
{
  // Never destroy DSP objects on the render thread. A swap/clear is deferred
  // while its single retirement slot is awaiting main-thread reclamation.
  if (mRetiredModel.load(std::memory_order_acquire) == nullptr
      && mShouldRemoveModel.exchange(false, std::memory_order_acq_rel))
  {
    ResamplingNAM* old = mModel.exchange(nullptr, std::memory_order_acq_rel);
    if (old != nullptr)
      mRetiredModel.store(old, std::memory_order_release);
    mModelCleared = true;
  }
  if (mRetiredIR.load(std::memory_order_acquire) == nullptr
      && mShouldRemoveIR.exchange(false, std::memory_order_acq_rel))
  {
    auto* old = mIR.exchange(nullptr, std::memory_order_acq_rel);
    if (old != nullptr)
      mRetiredIR.store(old, std::memory_order_release);
  }
  if (mRetiredBoostModel.load(std::memory_order_acquire) == nullptr
      && mShouldRemoveBoostModel.exchange(false, std::memory_order_acq_rel))
  {
    auto* old = mBoostModel.exchange(nullptr, std::memory_order_acq_rel);
    if (old != nullptr)
      mRetiredBoostModel.store(old, std::memory_order_release);
    mBoostModelCleared = true;
  }

  const std::uint64_t audioEpoch = mAudioConfigEpoch.load(std::memory_order_acquire);
  if (mRetiredModel.load(std::memory_order_acquire) == nullptr)
  {
    auto* current = mModel.load(std::memory_order_acquire);
    const bool currentModelUsable = current != nullptr && current->GetPreparedEpoch() == audioEpoch;
    // A model chosen while AMP is bypassed is fully prepared off the render
    // thread, but adoption waits until AMP is re-enabled. Retaining the live
    // object keeps one continuous identity-resampler history and prevents a
    // bypass-only selection from causing a phase reset or dropout.
    const bool deferBypassedReplacement = !ampActiveForBlock && currentModelUsable;
    if (!deferBypassedReplacement)
    {
      if (auto* pending = mPendingModel.exchange(nullptr, std::memory_order_acq_rel))
      {
        if (pending->GetPreparedEpoch() != audioEpoch
            || pending->GetPreparedRequestToken() != mAmpLoadRequest.load(std::memory_order_acquire)
            || pending->GetPreparedSelection() != mAmpModelIdx.load(std::memory_order_acquire)
            || pending->IsPreparedPlayerMode() != mPlayerModeTarget.load(std::memory_order_acquire))
          mRetiredModel.store(pending, std::memory_order_release);
        else
        {
          auto* old = mModel.exchange(pending, std::memory_order_acq_rel);
          if (old != nullptr)
            mRetiredModel.store(old, std::memory_order_release);
          _PublishModelMetadata(*pending);
          // Publish model-dependent level changes before this same render block
          // reads mInputGain/mOutputGain. Deferring these calls to OnIdle would
          // process one block using metadata from the previous model.
          _SetInputGain();
          _SetOutputGain();
          mModelCleared = false;
          mNewModelLoadedInDSP = true;
        }
      }
    }
  }
  if (mRetiredBoostModel.load(std::memory_order_acquire) == nullptr)
  {
    if (auto* pending = mPendingBoostModel.exchange(nullptr, std::memory_order_acq_rel))
    {
      if (pending->GetPreparedEpoch() != audioEpoch
          || pending->GetPreparedRequestToken() != mBoostLoadRequest.load(std::memory_order_acquire)
          || pending->GetPreparedSelection() != mBoostModelIdx.load(std::memory_order_acquire)
          || !mBoostActiveTarget.load(std::memory_order_acquire))
        mRetiredBoostModel.store(pending, std::memory_order_release);
      else
      {
        auto* old = mBoostModel.exchange(pending, std::memory_order_acq_rel);
        if (old != nullptr)
          mRetiredBoostModel.store(old, std::memory_order_release);
        mBoostModelCleared = false;
        mNewBoostModelLoadedInDSP = true;
      }
    }
  }
  if (mRetiredIR.load(std::memory_order_acquire) == nullptr)
  {
    if (auto* pending = mPendingIR.exchange(nullptr, std::memory_order_acq_rel))
    {
      if (pending->GetPreparedEpoch() != audioEpoch
          || pending->GetPreparedRequestToken() != mIRLoadRequest.load(std::memory_order_acquire))
        mRetiredIR.store(pending, std::memory_order_release);
      else
      {
        auto* old = mIR.exchange(pending, std::memory_order_acq_rel);
        if (old != nullptr)
          mRetiredIR.store(old, std::memory_order_release);
        mNewIRLoadedInDSP = true;
      }
    }
  }
}

void NeuralAmpModeler::_PublishModelMetadata(ResamplingNAM& model) noexcept
{
  mModelMetadataEpoch.store(model.GetPreparedEpoch(), std::memory_order_relaxed);
  mModelMetadataSelection.store(model.GetPreparedSelection(), std::memory_order_relaxed);
  mModelMetadataLatency.store(model.GetLatency(), std::memory_order_relaxed);
  mModelMetadataHasInputLevel.store(model.HasInputLevel(), std::memory_order_relaxed);
  mModelMetadataHasOutputLevel.store(model.HasOutputLevel(), std::memory_order_relaxed);
  mModelMetadataHasLoudness.store(model.HasLoudness(), std::memory_order_relaxed);
  mModelMetadataInputLevel.store(model.HasInputLevel() ? model.GetInputLevel() : 0.0, std::memory_order_relaxed);
  mModelMetadataOutputLevel.store(model.HasOutputLevel() ? model.GetOutputLevel() : 0.0, std::memory_order_relaxed);
  mModelMetadataLoudness.store(model.HasLoudness() ? model.GetLoudness() : 0.0, std::memory_order_relaxed);
  mModelMetadataRequest.store(model.GetPreparedRequestToken(), std::memory_order_release);
}

void NeuralAmpModeler::_FallbackDSP(iplug::sample** inputs, iplug::sample** outputs, const size_t numChannels,
                                    const size_t numFrames)
{
  for (auto c = 0; c < numChannels; c++)
    for (auto s = 0; s < numFrames; s++)
      outputs[c][s] = inputs[c][s];
}

std::optional<double> NeuralAmpModeler::_ModelInputReferenceDBU() const
{
  // Acquire the metadata publication before reading its relaxed fields. The
  // metadata describes the currently adopted object, which remains the DSP
  // source while a replacement selection is loading or parked in bypass.
  const unsigned int metadataRequest = mModelMetadataRequest.load(std::memory_order_acquire);
  const bool haveCurrentModel =
    mAmpActiveTarget.load(std::memory_order_acquire) && mModel.load(std::memory_order_acquire) != nullptr
    && metadataRequest == mAmpLoadRequest.load(std::memory_order_acquire)
    && mModelMetadataSelection.load(std::memory_order_relaxed) == mAmpModelIdx.load(std::memory_order_acquire)
    && mModelMetadataEpoch.load(std::memory_order_relaxed) == mAudioConfigEpoch.load(std::memory_order_acquire);
  if (!haveCurrentModel) return std::nullopt;
  // Use the documented 0 dBu fallback for models without a measured reference.
  return mModelMetadataHasInputLevel.load(std::memory_order_relaxed)
    ? mModelMetadataInputLevel.load(std::memory_order_relaxed) : 0.0;
}

void NeuralAmpModeler::_SetInputGain()
{
  double inputGainDB = GetParam(kInputLevel)->Value();
  if (GetParam(kCalibrateInput)->Bool())
    if (const auto reference = _ModelInputReferenceDBU())
      inputGainDB += GetParam(kInputCalibrationLevel)->Value() - *reference;
  mInputGain.store(DBToAmp(inputGainDB), std::memory_order_release);
}

void NeuralAmpModeler::_SetOutputGain()
{
  double gainDB = GetParam(kOutputLevel)->Value();
  const auto metadataRequest = mModelMetadataRequest.load(std::memory_order_acquire);
  const bool haveCurrentMetadata =
    mAmpActiveTarget.load(std::memory_order_acquire) && mModel.load(std::memory_order_acquire) != nullptr
    && metadataRequest == mAmpLoadRequest.load(std::memory_order_acquire)
    && mModelMetadataSelection.load(std::memory_order_relaxed) == mAmpModelIdx.load(std::memory_order_acquire)
    && mModelMetadataEpoch.load(std::memory_order_relaxed) == mAudioConfigEpoch.load(std::memory_order_acquire);
  if (haveCurrentMetadata)
  {
    const int outputMode = GetParam(kOutputMode)->Int();
    switch (outputMode)
    {
      case 1: // Normalized
        if (mModelMetadataHasLoudness.load(std::memory_order_relaxed))
        {
          const double targetLoudness = -18.0;
          gainDB += targetLoudness - mModelMetadataLoudness.load(std::memory_order_relaxed);
        }
        break;
      case 2: // Calibrated
        if (mModelMetadataHasOutputLevel.load(std::memory_order_relaxed))
          gainDB +=
            mModelMetadataOutputLevel.load(std::memory_order_relaxed) - GetParam(kInputCalibrationLevel)->Value();
        break;
      case 0: // Raw
      default: break;
    }
  }
  mOutputGain.store(DBToAmp(gainDB), std::memory_order_release);
}

std::string NeuralAmpModeler::_GetAmpModelName(int ampModelValue)
{
  // State 3 is the historical unshipped DessBlock-red slot. It previously
  // fell back to DessTortion-red, so preserve that sound for saved sessions.
  switch (ampModelValue)
  {
    case 0: return "DessTortion-blue";
    case 1:
    case 3: return "DessTortion-red";
    case 2: return "DessBlock-green";
    case 4: return "SickDess";
    default: return "DessTortion-blue";
  }
}

std::string NeuralAmpModeler::_LoadModelForGain(const std::string& ampModel, int gainValue,
                                                const unsigned int requestToken, const int requestedAmpIndex)
{
  // Clamp gain value to valid range
  if (gainValue < 0)
    gainValue = 0;
  if (gainValue > 10)
    gainValue = 10;

  if (ampModel == "Custom")
    return ""; // Custom model handled separately

  // Handle DessTortion-blue (Index 2)
  if (ampModel == "DessTortion-blue")
  {
    // Load static model: DessParametric.nam (remapped)
    // It handles gain via mCurrentParams
    WDL_String resourcePath;
    PlayerResourcePath(resourcePath, GetBundleID());
    WDL_String modelPath;
    modelPath.Set(resourcePath.Get());
    modelPath.Append("/models/DessTortion-blue/DessTortion-blue.nam");

    if (!std::filesystem::exists(modelPath.Get()))
    {
      std::string errorMsg = "DessTortion-blue model not found: " + std::string(modelPath.Get());
      std::cerr << errorMsg << std::endl;
      return errorMsg;
    }
    return _StageModel(modelPath, requestToken, requestedAmpIndex);
  }

  // Handle DessTortion-red (Index 1 in new enum)
  if (ampModel == "DessTortion-red")
  {
    WDL_String resourcePath;
    PlayerResourcePath(resourcePath, GetBundleID());
    WDL_String modelPath;
    modelPath.Set(resourcePath.Get());
    modelPath.Append("/models/DessTortion-red/DessTortion-red.nam");

    if (!std::filesystem::exists(modelPath.Get()))
    {
      std::string errorMsg = "DessTortion-red model not found: " + std::string(modelPath.Get());
      std::cerr << errorMsg << std::endl;
      return errorMsg;
    }
    return _StageModel(modelPath, requestToken, requestedAmpIndex);
  }

  // Handle DessBlock-green (Index 2 in the enum)
  if (ampModel == "DessBlock-green")
  {
    WDL_String resourcePath;
    PlayerResourcePath(resourcePath, GetBundleID());
    WDL_String modelPath;
    modelPath.Set(resourcePath.Get());
    modelPath.Append("/models/DessBlock-green/model.nam");

    if (!std::filesystem::exists(modelPath.Get()))
    {
      return "DessBlock-green model not found: " + std::string(modelPath.Get());
    }
    return _StageModel(modelPath, requestToken, requestedAmpIndex);
  }

  if (ampModel == "SickDess")
  {
    WDL_String resourcePath;
    PlayerResourcePath(resourcePath, GetBundleID());
    WDL_String modelPath;
    modelPath.Set(resourcePath.Get());
    modelPath.Append("/models/SickDess/SickDess.nam");

    if (!std::filesystem::exists(modelPath.Get()))
    {
      return "SickDess model not found: " + std::string(modelPath.Get());
    }
    return _StageModel(modelPath, requestToken, requestedAmpIndex);
  }

  // Legacy code for non-parametric models (should not be reached with new enum)
  // Lock gain for DessTortion (Only 5 and 6 available)
  if (ampModel == "DessTortion")
  {
    if (gainValue <= 5)
      gainValue = 5;
    else
      gainValue = 6;
  }

  // Build model filename: {ampModel}-{gainValue} (e.g., "DessBlock-1", "DessCut-5")
  std::stringstream ss;
  ss << ampModel << "-" << gainValue;
  std::string modelBaseName = ss.str();

  // Get resource path - iPlug2 stores resources in the bundle
  WDL_String resourcePath;
  PlayerResourcePath(resourcePath, GetBundleID());

  // Build path: models/{ampModel}/{ampModel}-{gainValue}.nam
  WDL_String modelPath;
  modelPath.Set(resourcePath.Get());
  modelPath.Append("/models/");
  modelPath.Append(ampModel.c_str());
  modelPath.Append("/");
  modelPath.Append(modelBaseName.c_str());
  modelPath.Append(".nam");

  // Check if .nam file exists
  if (!std::filesystem::exists(modelPath.Get()))
  {
    // Try directory format (config.json + weights.npy)
    modelPath.Set(resourcePath.Get());
    modelPath.Append("/models/");
    modelPath.Append(ampModel.c_str());
    modelPath.Append("/");
    modelPath.Append(modelBaseName.c_str());
    modelPath.Append("/config.json");
    if (!std::filesystem::exists(modelPath.Get()))
    {
      std::string errorMsg = "Model file not found: " + modelBaseName;
      std::cerr << errorMsg << std::endl;
      return errorMsg;
    }
    // Remove /config.json to get directory path
    modelPath.remove_filepart();
  }

  return _StageModel(modelPath, requestToken, requestedAmpIndex);
}

std::string NeuralAmpModeler::_StagePlayerManifest(const WDL_String& manifestPath, const unsigned int requestToken,
                                                   const int requestedAmpIndex)
{
  const std::uint64_t preparedEpoch = mAudioConfigEpoch.load(std::memory_order_acquire);
  if ((preparedEpoch & 1U) != 0U)
    return "";
  const double sampleRate = mAudioSampleRate.load(std::memory_order_acquire);
  const int maxBlockSize = mAudioMaxBlockSize.load(std::memory_order_acquire);
  try
  {
    auto imported = std::move(mImportedAmp);
    const auto path = std::filesystem::u8path(manifestPath.Get());
    const bool useImport = imported && imported->manifest.manifest_path == path;
    auto manifest = std::make_shared<nam::player::Manifest>(
      useImport ? std::move(imported->manifest) : nam::player::LoadManifest(path, kNumPlayerConditionParams));
    // Factory voicing is a product setting, not a new trained artifact. Keep
    // immutable package bytes intact; only this instance's defaults change.
    if (const auto* defaults = _BundledFactoryDefaults(std::filesystem::u8path(manifestPath.Get())))
      for (std::size_t index = 0; index < manifest->controls.size(); ++index)
        manifest->controls[index].default_value = manifest->controls[index].ModelValueFromNormalized(
          defaults->at(kFirstPlayerConditionParam + index) / 100.0);
    std::unique_ptr<nam::DSP> model = useImport ? std::move(imported->model) : nam::player::LoadPlayerDSP(*manifest);
    const double modelSampleRate = model->GetExpectedSampleRate();
    if (std::isfinite(modelSampleRate) && modelSampleRate > 0.0
        && std::abs(modelSampleRate - manifest->sample_rate) > 1.0e-6)
      throw std::invalid_argument("Manifest sample_rate disagrees with the loaded model");

    const bool applyDefaults = requestToken == mApplyPlayerDefaultsRequest.load(std::memory_order_acquire);
    const int playerParameterCount = manifest->PlayerParameterCount();
    std::vector<double> modelParameters(static_cast<std::size_t>(playerParameterCount), 0.5);
    std::array<double, kNumPlayerConditionParams> normalizedValues{};
    for (int index = 0; index < playerParameterCount; ++index)
    {
      const auto& control = manifest->controls[static_cast<std::size_t>(index)];
      const double normalized =
        applyDefaults ? control.NormalizedDefault()
                      : mPlayerConditionTargets[static_cast<std::size_t>(index)].load(std::memory_order_acquire);
      normalizedValues[static_cast<std::size_t>(index)] = normalized;
      modelParameters[static_cast<std::size_t>(index)] = control.ModelValueFromNormalized(normalized);
    }
    model->SetDefaultParams(
      modelParameters.empty() ? nullptr : modelParameters.data(), static_cast<int>(modelParameters.size()));

    auto temp = std::make_unique<ResamplingNAM>(std::move(model), sampleRate);
    temp->SetPlayerManifest(manifest);
    temp->SetPreparedPlayerMode(true);
    temp->Reset(sampleRate, maxBlockSize);
    temp->SetPreparedEpoch(preparedEpoch);
    temp->SetPreparedRequest(requestToken, requestedAmpIndex);
    temp->SetPreparedPath(manifest->manifest_path.u8string().c_str());

    if (preparedEpoch != mAudioConfigEpoch.load(std::memory_order_acquire)
        || requestToken != mAmpLoadRequest.load(std::memory_order_acquire)
        || requestedAmpIndex != mAmpModelIdx.load(std::memory_order_acquire)
        || !mPlayerModeTarget.load(std::memory_order_acquire))
      return "";

    if (applyDefaults)
    {
      for (int index = 0; index < kNumPlayerConditionParams; ++index)
      {
        const double normalized =
          index < playerParameterCount ? normalizedValues[static_cast<std::size_t>(index)] : 0.5;
        mPlayerConditionTargets[static_cast<std::size_t>(index)].store(normalized, std::memory_order_release);
        mPendingPlayerDefaults[static_cast<std::size_t>(index)].store(normalized, std::memory_order_release);
      }
      mPendingPlayerDefaultsRequest.store(requestToken, std::memory_order_release);
      mApplyPlayerDefaultsRequest.store(0U, std::memory_order_release);
    }

    // The render thread remains the sole owner of live-DSP adoption, but the
    // editor may already present the validated package and accept parameter
    // edits when a standalone audio device has not produced its first block.
    mStagedPlayerManifestForUI = manifest;
    mStagedPlayerManifestRequest = requestToken;
    mStagedPlayerManifestSelection = requestedAmpIndex;
    mStagedPlayerManifestEpoch = preparedEpoch;
    mStagedPlayerManifestUIChanged.store(true, std::memory_order_release);
    delete mPendingModel.exchange(temp.release(), std::memory_order_acq_rel);
  }
  catch (const std::exception& error)
  {
    std::cerr << "Failed to load player artifact" << std::endl;
    std::cerr << error.what() << std::endl;
    return error.what();
  }
  return "";
}

std::string NeuralAmpModeler::_StageModel(const WDL_String& modelPath, const unsigned int requestToken,
                                          const int requestedAmpIndex)
{
  const std::uint64_t preparedEpoch = mAudioConfigEpoch.load(std::memory_order_acquire);
  if ((preparedEpoch & 1U) != 0U)
    return "";
  const double sampleRate = mAudioSampleRate.load(std::memory_order_acquire);
  const int maxBlockSize = mAudioMaxBlockSize.load(std::memory_order_acquire);
  try
  {
    auto dspPath = std::filesystem::u8path(modelPath.Get());
    std::unique_ptr<nam::DSP> model = nam::get_dsp(dspPath);
    std::unique_ptr<ResamplingNAM> temp = std::make_unique<ResamplingNAM>(std::move(model), sampleRate);
    temp->SetPreparedPlayerMode(false);
    const int globalParams = temp->NumGlobalParams();
    if (globalParams < 0 || globalParams > 1)
      throw std::invalid_argument("DessMetal amp slots support fixed snapshots or one Gain condition");
    if (globalParams == 1)
    {
      const double defaultCondition = mTargetGain.load(std::memory_order_relaxed);
      temp->SetDefaultParams(&defaultCondition, 1);
    }
    temp->Reset(sampleRate, maxBlockSize);
    temp->SetPreparedEpoch(preparedEpoch);
    temp->SetPreparedRequest(requestToken, requestedAmpIndex);
    temp->SetPreparedPath(modelPath.Get());

    if (preparedEpoch != mAudioConfigEpoch.load(std::memory_order_acquire)
        || requestToken != mAmpLoadRequest.load(std::memory_order_acquire)
        || requestedAmpIndex != mAmpModelIdx.load(std::memory_order_acquire)
        || mPlayerModeTarget.load(std::memory_order_acquire))
      return "";

    delete mPendingModel.exchange(temp.release(), std::memory_order_acq_rel);
  }
  catch (const std::exception& e)
  {
    std::cerr << "Failed to read DSP module" << std::endl;
    std::cerr << e.what() << std::endl;
    return e.what();
  }
  return "";
}

dsp::wav::LoadReturnCode NeuralAmpModeler::_StageIR(const WDL_String& irPath, const unsigned int requestToken)
{
  const std::uint64_t preparedEpoch = mAudioConfigEpoch.load(std::memory_order_acquire);
  if ((preparedEpoch & 1U) != 0U)
    return dsp::wav::LoadReturnCode::SUCCESS;
  const double sampleRate = mAudioSampleRate.load(std::memory_order_acquire);
  const int maxBlockSize = mAudioMaxBlockSize.load(std::memory_order_acquire);
  dsp::wav::LoadReturnCode wavState = dsp::wav::LoadReturnCode::ERROR_OTHER;
  try
  {
    auto irPathU8 = std::filesystem::u8path(irPath.Get());
    auto temp = std::make_unique<dsp::ImpulseResponse>(irPathU8.string().c_str(), sampleRate);
    wavState = temp->GetWavState();
    if (wavState == dsp::wav::LoadReturnCode::SUCCESS)
    {
      temp->PrepareBuffers(kNumChannelsInternal, maxBlockSize);
      temp->SetPreparedEpoch(preparedEpoch);
      temp->SetPreparedRequestToken(requestToken);
      if (preparedEpoch != mAudioConfigEpoch.load(std::memory_order_acquire)
          || requestToken != mIRLoadRequest.load(std::memory_order_acquire))
        return dsp::wav::LoadReturnCode::SUCCESS;
      delete mPendingIR.exchange(temp.release(), std::memory_order_acq_rel);
    }
  }
  catch (const std::exception& e)
  {
    wavState = dsp::wav::LoadReturnCode::ERROR_OTHER;
    std::cerr << "Caught unhandled exception while attempting to load IR:" << std::endl;
    std::cerr << e.what() << std::endl;
  }

  return wavState;
}


void NeuralAmpModeler::_InitToneStack()
{
  // If you want to customize the tone stack, then put it here!
  mToneStack = std::make_unique<dsp::tone_stack::BasicNamToneStack>();
}
void NeuralAmpModeler::_ProcessInput(iplug::sample** inputs, const size_t nFrames, const size_t nChansIn,
                                     const size_t nChansOut)
{
  if (nChansOut != 1 || mInputArray.empty())
    return;

  // Every block starts from silence so zero-input configurations cannot reuse
  // samples from the previous callback.
  std::fill(mInputArray[0].begin(), mInputArray[0].begin() + nFrames, 0.0);
  if (nChansIn == 0 || inputs == nullptr)
    return;

  // On the standalone, we can probably assume that the user has plugged into only one input and they expect it to be
  // carried straight through. Don't apply any division over nChansIn because we're just "catching anything out there."
  // However, in a DAW, it's probably something providing stereo, and we want to take the average in order to avoid
  // doubling the loudness. (This would change w/ double mono processing)
  double gain = mInputGain.load(std::memory_order_acquire);
#ifndef APP_API
  gain /= (float)nChansIn;
#endif
  for (size_t c = 0; c < nChansIn; c++)
    for (size_t s = 0; s < nFrames; s++)
      if (c == 0)
        mInputArray[0][s] = gain * inputs[c][s];
      else
        mInputArray[0][s] += gain * inputs[c][s];
}

void NeuralAmpModeler::_ProcessOutput(iplug::sample** inputs, iplug::sample** outputs, const size_t nFrames,
                                      const size_t nChansIn, const size_t nChansOut)
{
  const double gain = mOutputGain.load(std::memory_order_acquire);
  if (nChansIn != 1 || inputs == nullptr)
  {
    for (size_t cout = 0; cout < nChansOut; ++cout)
      if (outputs != nullptr && outputs[cout] != nullptr)
        std::fill(outputs[cout], outputs[cout] + nFrames, 0.0);
    return;
  }
  // Broadcast the internal mono stream to all output channels.
  const size_t cin = 0;
  for (auto cout = 0; cout < nChansOut; cout++)
    for (auto s = 0; s < nFrames; s++)
    {
      const double scaledSample = gain * inputs[cin][s];
      const double finiteSample = std::isfinite(scaledSample) ? scaledSample : 0.0;
#ifdef APP_API // Ensure valid output to interface
      outputs[cout][s] = std::clamp(finiteSample, -1.0, 1.0);
#else // In a DAW, other things may come next and should be able to handle large
      // values.
      outputs[cout][s] = finiteSample;
#endif
    }
}

bool NeuralAmpModeler::_IsRigParam(int param)
{
  return std::find(kRigParams.begin(), kRigParams.end(), param) != kRigParams.end();
}

std::array<double, NeuralAmpModeler::kRigParams.size()> NeuralAmpModeler::_RigValues() const
{
  std::array<double, kRigParams.size()> values{};
  for (size_t i = 0; i < values.size(); ++i) values[i] = GetParam(kRigParams[i])->Value();
  return values;
}

void NeuralAmpModeler::_LoadStandaloneRig()
{
#if defined(OS_IOS)
  // The iOS app hosts its own AUv3 code in process. Other AU hosts keep
  // separate instance state in their projects and must never write app prefs.
  const auto bundle = CFBundleGetMainBundle();
  const auto id = bundle ? CFBundleGetIdentifier(bundle) : nullptr;
  if (!id || !CFEqual(id, CFSTR("com.dessdynamics.dessamp"))) return;
#elif !defined(APP_API)
  return;
#endif
  mRigSettingsFile = mDessPresetRootDirectory.parent_path() / "rig-settings.json";
  mLastRigValues = _RigValues();
  std::error_code ec;
  if (!std::filesystem::is_regular_file(mRigSettingsFile, ec)
      || std::filesystem::file_size(mRigSettingsFile, ec) > 4096 || ec) return;
  try
  {
    std::ifstream input(mRigSettingsFile);
    const auto doc = nlohmann::json::parse(input);
    if (doc.at("version") != 1) return;
    auto values = mLastRigValues;
    for (size_t i = 0; i < values.size(); ++i)
    {
      const auto* param = GetParam(kRigParams[i]);
      const auto value = doc.at("rig").at(param->GetName()).get<double>();
      if (!std::isfinite(value) || value < param->GetMin() || value > param->GetMax()) return;
      values[i] = value;
    }
    for (size_t i = 0; i < values.size(); ++i) GetParam(kRigParams[i])->Set(values[i]);
    mLastRigValues = _RigValues();
    _SetInputGain();
    _SetOutputGain();
  }
  catch (const std::exception& e) { std::cerr << "Ignoring invalid DessAmp rig settings: " << e.what() << '\n'; }
}

// Called by the UI timer and editor close, never by the render callback.
void NeuralAmpModeler::_SaveStandaloneRig(bool flush)
{
  if (mRigSettingsFile.empty()) return;
  const auto now = std::chrono::steady_clock::now();
  if (!flush && now < mNextRigSave) return;
  mNextRigSave = now + std::chrono::milliseconds(150);
  const auto values = _RigValues();
  if (values == mLastRigValues) return;
  nlohmann::json doc{{"version", 1}};
  for (size_t i = 0; i < values.size(); ++i) doc["rig"][GetParam(kRigParams[i])->GetName()] = values[i];
  std::error_code ec;
  std::filesystem::create_directories(mRigSettingsFile.parent_path(), ec);
  if (ec) return;
  auto temp = mRigSettingsFile;
  temp += ".tmp-" + std::to_string(now.time_since_epoch().count());
  std::ofstream output(temp, std::ios::trunc);
  output << doc.dump() << '\n';
  output.close();
  if (output) std::filesystem::rename(temp, mRigSettingsFile, ec);
  if (!output || ec) std::filesystem::remove(temp, ec);
  else mLastRigValues = values;
}

void NeuralAmpModeler::_SetParameterValueFromMainThread(const int paramIdx, const double normalizedValue)
{
  // OnIdle is a non-render thread. Bracket programmatic corrections as a
  // complete host gesture so AU/VST3 generic controls and automation receive
  // the same value used by the DSP and custom editor.
  BeginInformHostOfParamChangeFromUI(paramIdx);
  SetParameterValue(paramIdx, normalizedValue);
  EndInformHostOfParamChangeFromUI(paramIdx);
  SendParameterValueFromDelegate(paramIdx, normalizedValue, true);
}

void NeuralAmpModeler::_UpdateControlsFromModel()
{
  if (auto* graphics = GetUI())
    if (auto* settings = graphics->GetControlWithTag(kCtrlTagSettingsBox))
      settings->As<DessAmpSettingsPanelControl>()->SetModelInputReference(
        _ModelInputReferenceDBU(), mModelMetadataHasInputLevel.load(std::memory_order_relaxed));
}

void NeuralAmpModeler::_UpdatePlayerControls()
{
  auto* graphics = GetUI();
  if (graphics == nullptr)
    return;

  const bool playerMode = mPlayerModeTarget.load(std::memory_order_acquire);
  nam::player::Manifest activeManifest;
  bool haveManifest = false;
  bool waitingForReplacement = false;
  const bool hasSelection = _GetPlayerManifestPathSnapshot().GetLength() > 0;
  if (playerMode)
  {
    std::lock_guard<std::mutex> lock(mDSPLoadMutex);
    if (auto* model = mModel.load(std::memory_order_acquire))
    {
      if (model->IsPreparedPlayerMode() && model->GetPlayerManifest() != nullptr
          && model->GetPreparedRequestToken() == mAmpLoadRequest.load(std::memory_order_acquire)
          && model->GetPreparedSelection() == mAmpModelIdx.load(std::memory_order_acquire)
          && model->GetPreparedEpoch() == mAudioConfigEpoch.load(std::memory_order_acquire))
      {
        activeManifest = *model->GetPlayerManifest();
        haveManifest = true;
      }
    }
    if (!haveManifest && mStagedPlayerManifestForUI != nullptr
        && mStagedPlayerManifestRequest == mAmpLoadRequest.load(std::memory_order_acquire)
        && mStagedPlayerManifestSelection == mAmpModelIdx.load(std::memory_order_acquire)
        && mStagedPlayerManifestEpoch == mAudioConfigEpoch.load(std::memory_order_acquire))
    {
      activeManifest = *mStagedPlayerManifestForUI;
      haveManifest = true;
    }
    // A selection invalidates the old token before the next package has
    // finished loading. Keep the last validated face on screen until the
    // replacement is ready; it must never become the CUSTOM chooser.
    if (!haveManifest && hasSelection && mStagedPlayerManifestForUI &&
        !(mHandledAmpLoadRequest.load() == mAmpLoadRequest.load() && mAmpRetryNeeded.load()))
    {
      activeManifest = *mStagedPlayerManifestForUI;
      haveManifest = true;
      waitingForReplacement = true;
    }
  }

  const int conditionCount = playerMode && haveManifest
                               ? std::clamp(activeManifest.PlayerParameterCount(), 0, kNumPlayerConditionParams)
                               : 0;

  const auto normalizeArtifactCode = [](const std::string& rawID) {
    std::string code;
    for (const unsigned char character : rawID)
    {
      if (std::isalnum(character))
        code.push_back(static_cast<char>(std::toupper(character)));
      if (code.size() == 8U)
        break;
    }
    return code;
  };
  const std::string artifactCode = playerMode && haveManifest ? normalizeArtifactCode(activeManifest.artifact_id) : "";
  const auto shellIdentity = playerMode && haveManifest
                               ? dessamp::player::ResolveHeadShellIdentity(
                                   activeManifest.head_shell_style, activeManifest.morph.enabled, artifactCode)
                               : dessamp::player::HeadShellIdentity::Generic;
  const bool forgeShell = shellIdentity == dessamp::player::HeadShellIdentity::Forge;
  const bool fireShell = shellIdentity == dessamp::player::HeadShellIdentity::Fire;
  const bool redlineShell = shellIdentity == dessamp::player::HeadShellIdentity::Redline;
  const auto morphPresentation = dessamp::player::ResolveMorphPresentation(
    shellIdentity, activeManifest.morph.enabled, activeManifest.global_condition_size,
    activeManifest.morph.from_label, activeManifest.morph.to_label, activeManifest.controls);
  const bool customHead = haveManifest && std::none_of(mBundledAmplifierPaths.begin(), mBundledAmplifierPaths.end(),
    [&](const std::string& path) { return activeManifest.manifest_path == std::filesystem::u8path(path); });
  const bool redlineVariantSelector = redlineShell && !customHead && activeManifest.product_family == "redline"
                                      && !_BundledAmplifierVariantPath("redline", "us").empty()
                                      && !_BundledAmplifierVariantPath("redline", "brit").empty();
  const std::string resolvedShellStyle = forgeShell
                                           ? "forge-streamlined-v1"
                                           : fireShell ? "fire-chrome-v1"
                                                       : playerMode && haveManifest ? activeManifest.head_shell_style
                                                                                   : "studio-black-v1";

  if (auto* head = graphics->GetControlWithTag(kCtrlTagPlayerHead))
  {
    auto* headControl = head->As<DessAmpHeadControl>();
    headControl->SetShellStyle(resolvedShellStyle.c_str());
    const int morphConditionIndex = playerMode && haveManifest && morphPresentation.IsEnabled()
                                      ? morphPresentation.conditionIndex
                                      : -1;
    mPlayerMorphConditionIndex.store(morphConditionIndex, std::memory_order_release);
    const bool showMorph = morphConditionIndex >= 0 && morphConditionIndex < conditionCount;
    headControl->SetMorphPresentation(
      showMorph, showMorph ? GetParam(kFirstPlayerConditionParam + morphConditionIndex)->GetNormalized() : 0.5,
      showMorph ? morphPresentation.fromLabel.data() : "GER",
      showMorph ? morphPresentation.toLabel.data() : "BRIT");
    headControl->SetSubtitle(
      customHead ? "C U S T O M" : forgeShell ? "F O R G E" : fireShell ? "F I R E" : redlineShell ? "R E D L I N E" : "");
  }
  else
    mPlayerMorphConditionIndex.store(-1, std::memory_order_release);

  const auto normalizeControlID = [](std::string id) {
    std::transform(
      id.begin(), id.end(), id.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return id;
  };
  const auto isLeadOnlyHiddenControl = [&normalizeControlID](const std::string& rawID) {
    const std::string id = normalizeControlID(rawID);
    return id == "amp_gain_1" || id == "gain_1" || id == "gain1" || id == "clean_gain" || id == "clean_lead"
           || id == "clean_lead_channel" || id == "channel";
  };
  const auto visualRankForID = [&normalizeControlID](const std::string& rawID, const int fallback) {
    const std::string id = normalizeControlID(rawID);
    if (id == "amp_gain_2" || id == "gain_2" || id == "gain2" || id == "lead_gain")
      return 0;
    if (id == "bright" || id == "amp_bright")
      return 10;
    if (id == "bottom" || id == "amp_bottom" || id == "depth" || id == "resonance")
      return 20;
    if (id == "amp_bass" || id == "bass")
      return 30;
    if (id == "amp_middle" || id == "middle" || id == "mid")
      return 40;
    if (id == "amp_treble" || id == "treble")
      return 50;
    if (id == "amp_mvc" || id == "mvc" || id == "master" || id == "lead_volume")
      return 55;
    if (id == "morph")
      return 60;
    if (id == "mid_boost" || id == "midboost" || id == "amp_mid_boost")
      return 70;
    if (id == "amp_presence" || id == "presence")
      return 80;
    return 100 + fallback;
  };
  const auto displayLabelForID = [&normalizeControlID](const nam::player::Control& definition) {
    const std::string id = normalizeControlID(definition.id);
    if (id == "amp_gain_2" || id == "gain_2" || id == "gain2" || id == "lead_gain")
      return std::string("Lead Gain");
    if (id == "amp_bass" || id == "bass")
      return std::string("Bass");
    if (id == "amp_middle" || id == "middle" || id == "mid")
      return std::string("Middle");
    if (id == "amp_treble" || id == "treble")
      return std::string("Treble");
    if (id == "amp_mvc" || id == "mvc" || id == "master" || id == "lead_volume")
      return std::string("Lead Volume");
    if (id == "amp_presence" || id == "presence")
      return std::string("Presence");
    return definition.label;
  };

  struct VisiblePlayerControl
  {
    int modelIndex = 0;
    int visualRank = 0;
  };
  std::vector<VisiblePlayerControl> visibleControls;
  if (activeManifest.controls.size() >= static_cast<std::size_t>(conditionCount))
  {
    visibleControls.reserve(static_cast<std::size_t>(conditionCount));
    for (int modelIndex = 0; modelIndex < conditionCount; ++modelIndex)
    {
      const auto& definition = activeManifest.controls[static_cast<std::size_t>(modelIndex)];
      if (!isLeadOnlyHiddenControl(definition.id))
        visibleControls.push_back({modelIndex, visualRankForID(definition.id, modelIndex)});
    }
    std::stable_sort(visibleControls.begin(), visibleControls.end(), [](const auto& left, const auto& right) {
      return left.visualRank < right.visualRank;
    });
  }

  const auto boundsForVisiblePosition = [graphics, fireShell, redlineShell,
                                         count = visibleControls.size() + (redlineVariantSelector ? 1U : 0U)](
                                          const std::size_t position, const bool isSwitch) {
    const DessAmpUI::ResponsiveLayout layout(graphics->GetBounds());
    if (layout.phone)
      return layout.PhoneCondition(count, position, fireShell);
    // REDLINE has one learned Gain control plus the code-owned Voice selector.
    // Keep that pair as a compact, symmetrical centre group while preserving
    // each control's full draw rectangle and direct hit target. Other shells
    // retain the full-width responsive row used by their larger control sets.
    // Gain's full hit bounds extend 50 points from centre and Voice extends 34.
    // Keep at least six points between those rectangles, even on the narrow
    // phone layout, while pulling the pair into one deliberately tight group.
    const float rowHalfSpan = redlineShell ? std::max(90.f, layout.head.W() * 0.085f)
                                           : layout.head.W() * 0.445f;
    const float rowLeft = layout.head.MW() - rowHalfSpan;
    const float rowRight = layout.head.MW() + rowHalfSpan;
    const float cellWidth = count > 0U ? (rowRight - rowLeft) / static_cast<float>(count) : rowRight - rowLeft;
    const float center = count <= 1U ? (rowLeft + rowRight) * 0.5f
                                    : rowLeft + cellWidth * (static_cast<float>(position) + 0.5f);
    const float top = layout.head.T + layout.head.H() * (fireShell ? 0.65f : 0.63f);
    // FIRE's chrome shell has a taller lower control panel than the FORGE and
    // REDLINE shells. Translate the complete control bounds (rather than only
    // increasing their top edge) so the knobs are vertically centred without
    // shrinking their responsive diameter or hit target on iPhone.
    const float fireRowOffset = fireShell ? std::clamp(layout.head.H() * 0.03f, 10.f, 14.f) : 0.f;
    const auto bounds = isSwitch ? IRECT(center - 34.f, top + 8.f, center + 34.f, layout.head.B - 42.f)
                                 : IRECT(center - 50.f, top, center + 50.f, layout.head.B - 24.f);
    return bounds.GetTranslated(0.f, fireRowOffset);
  };

  // Put the physical shell back before applying the model-specific visibility
  // decisions. This also makes AMP restoration after visiting PRE deterministic.
  if (mPlayerPage == kAmpPage)
    graphics->ForControlInGroup("AMP_STAGE", [](IControl* control) { control->Hide(false); });

  for (int index = 0; index < kNumPlayerConditionParams; ++index)
  {
    auto* raw = graphics->GetControlWithTag(kFirstPlayerConditionCtrlTag + index);
    if (raw != nullptr)
    {
      auto* control = raw->As<DessAmpKnobControl>();
      control->SetIgnoreMouse(waitingForReplacement);
      control->SetDisabled(true);
      control->Hide(true);
    }

    auto* toggleRaw = graphics->GetControlWithTag(kFirstPlayerConditionToggleCtrlTag + index);
    if (toggleRaw != nullptr)
    {
      auto* toggle = toggleRaw->As<DessAmpToggleControl>();
      toggle->SetPresentation("Condition", false);
      toggle->SetTooltip("Load an amplifier to enable this switch");
      toggle->SetIgnoreMouse(waitingForReplacement);
      toggle->SetDisabled(true);
      toggle->Hide(true);
    }
  }

  for (int slot = 0; slot < 3; ++slot)
  {
    if (auto* placeholder = graphics->GetControlWithTag(kFirstPlayerSwitchPlaceholderCtrlTag + slot))
    {
      placeholder->As<DessAmpReservedToggleControl>()->SetUnavailable();
      placeholder->Hide(true);
    }
  }

  auto* variantRaw = graphics->GetControlWithTag(kCtrlTagPlayerVariantSelector);
  auto* variantSelector = variantRaw == nullptr ? nullptr : variantRaw->As<DessAmpVariantSelectorControl>();
  if (variantSelector != nullptr)
  {
    variantSelector->SetIgnoreMouse(waitingForReplacement);
    variantSelector->SetVariants("US", "BRIT", activeManifest.product_variant.c_str(), redlineVariantSelector);
    if (redlineVariantSelector)
      variantSelector->SetTargetAndDrawRECTs(boundsForVisiblePosition(visibleControls.size(), false));
    variantSelector->Hide(mPlayerPage != kAmpPage || !redlineVariantSelector);
  }

  auto* selectorRaw = graphics->GetControlWithTag(kCtrlTagPlayerManifest);
  auto* selector = selectorRaw == nullptr ? nullptr : selectorRaw->As<DessAmpModelSelectorControl>();
  if (auto* strip = graphics->GetControlWithTag(kCtrlTagPlayerFactoryStrip))
    strip->As<DessAmpFactoryStrip>()->Select(_AmplifierStripIndex());
  auto* emptyState = graphics->GetControlWithTag(kCtrlTagPlayerMode);

  const bool ready = playerMode && haveManifest
                     && activeManifest.controls.size() >= static_cast<std::size_t>(conditionCount);
  if (selector) selector->SetIgnoreMouse(waitingForReplacement || !ready);
  if (!ready)
  {
    const bool unavailable = hasSelection && mAmpRetryNeeded.load();
    if (selector)
    {
      if (!hasSelection) selector->SetEmpty();
      else if (unavailable) selector->SetError();
      else selector->SetLoading();
    }
    if (emptyState != nullptr)
      emptyState->Hide(hasSelection || mPlayerPage != kAmpPage);
    graphics->ForControlInGroup("PLAYER_FOOTER", [hasSelection, unavailable](IControl* control) {
      if (auto* text = control->As<ITextControl>())
      {
        text->SetStr(!hasSelection ? "NO AMPLIFIER" : unavailable ? "AMPLIFIER UNAVAILABLE" : "LOADING AMPLIFIER");
        text->SetDirty(false);
      }
    });
    if (mPlayerPage != kAmpPage)
      graphics->ForControlInGroup("AMP_STAGE", [](IControl* control) { control->Hide(true); });
    graphics->SetAllControlsDirty();
    return;
  }

  // Show the owner's manifest name for custom heads. Factory identities stay stable.
  const std::string displayName = customHead ? activeManifest.name :
                                  forgeShell ? "DESS FORGE" :
                                  fireShell ? "DESS FIRE" :
                                  redlineShell ? "DESS REDLINE" :
                                  artifactCode.empty() ? "DESS HEAD" : "DESS HEAD  /  " + artifactCode;

  const int visibleConditionCount = static_cast<int>(visibleControls.size());

  if (selector != nullptr)
  {
    const std::string runtimeLabel = morphPresentation.IsEnabled()
                                       ? "LIVE MORPH"
                                       : conditionCount == 0 ? "SNAPSHOT"
                                       : activeManifest.runtime_quality.mode == "slimmable" ? "SLIMMABLE"
                                                                                           : "CONDITIONED";
    const std::string redlineVoice = activeManifest.product_variant == "brit" ? "BRIT" : "US";
    const std::string catalogLabel = redlineShell ? redlineVoice + "  /  " + runtimeLabel : runtimeLabel;
    selector->SetReady(displayName.c_str(), visibleConditionCount, catalogLabel.c_str());
  }
  if (emptyState != nullptr)
    emptyState->Hide(true);

  for (std::size_t position = 0; position < visibleControls.size(); ++position)
  {
    const int modelIndex = visibleControls[position].modelIndex;
    const auto& definition = activeManifest.controls[static_cast<std::size_t>(modelIndex)];
    if (definition.IsSwitch())
    {
      auto* raw = graphics->GetControlWithTag(kFirstPlayerConditionToggleCtrlTag + modelIndex);
      if (raw == nullptr)
        continue;
      auto* control = raw->As<DessAmpToggleControl>();
      control->SetTargetAndDrawRECTs(boundsForVisiblePosition(position, true));
      control->SetPresentation(definition.label.c_str(), definition.NormalizedDefault() > 0.5);
      control->SetTooltip((definition.label + ": off / on").c_str());
      control->SetDisabled(false);
      control->Hide(mPlayerPage != kAmpPage);
      continue;
    }

    auto* raw = graphics->GetControlWithTag(kFirstPlayerConditionCtrlTag + modelIndex);
    if (raw == nullptr)
      continue;
    auto* control = raw->As<DessAmpKnobControl>();
    control->SetTargetAndDrawRECTs(boundsForVisiblePosition(position, false));
    const std::string normalizedID = normalizeControlID(definition.id);
    const bool isMorph = normalizedID == "morph";
    const bool percentAsAmpDial = definition.unit == "%" && !isMorph;
    const bool detentAsAmpDial = definition.unit == "detent";
    const bool clockAsAmpDial = definition.unit == "clock h";
    const bool manifestAmpDial = definition.unit.empty();
    const double displayScale = percentAsAmpDial ? 0.1 : 1.0;
    // Physical capture projects record clock positions as hours. The Fireball
    // convention is the ordinary ten-point face from 07:00 (0) to 17:00 (10).
    const double displayOffset = clockAsAmpDial ? -7.0 : 0.0;
    const double displayMinimum = definition.DisplayMinimum() * displayScale + displayOffset;
    const double displayMaximum = definition.DisplayMaximum() * displayScale + displayOffset;
    const double displayDefault =
      definition.DisplayValueFromNormalized(definition.NormalizedDefault()) * displayScale + displayOffset;
    const bool constrainedAmpDial = displayMinimum >= 0.0 && displayMaximum <= 10.0
                                    && (percentAsAmpDial || detentAsAmpDial || clockAsAmpDial || manifestAmpDial);
    const double dialMinimumNormalized = constrainedAmpDial ? displayMinimum * 0.1 : 0.0;
    const double dialMaximumNormalized = constrainedAmpDial ? displayMaximum * 0.1 : 1.0;
    const std::string label = displayLabelForID(definition);
    const bool ampDialUnit = percentAsAmpDial || detentAsAmpDial || clockAsAmpDial;
    const char* displayUnit = ampDialUnit ? "" : definition.unit.c_str();
    control->SetPresentation(label.c_str(), displayMinimum, displayMaximum, displayDefault, displayUnit,
                             dialMinimumNormalized, dialMaximumNormalized);
    if (isMorph && modelIndex == morphPresentation.conditionIndex)
      control->SetEndpointLabels(morphPresentation.fromLabel.data(), morphPresentation.toLabel.data());
    std::ostringstream tooltip;
    tooltip << label << ": " << displayMinimum << " to " << displayMaximum;
    if (!ampDialUnit && !definition.unit.empty())
      tooltip << " " << definition.unit;
    if (constrainedAmpDial && (displayMinimum > 0.0 || displayMaximum < 10.0))
      tooltip << " (trained range on 0-10 dial)";
    control->SetTooltip(tooltip.str().c_str());
    control->SetDisabled(false);
    control->Hide(mPlayerPage != kAmpPage);
  }

  std::ostringstream footer;
  footer << displayName << "  /  " << visibleConditionCount << (visibleConditionCount == 1 ? " CONTROL" : " CONTROLS")
         << "  /  " << static_cast<int>(std::lround(activeManifest.sample_rate)) << " HZ";
  if (redlineShell && !activeManifest.product_variant.empty())
    footer << "  /  " << (activeManifest.product_variant == "brit" ? "BRIT" : "US");
  graphics->ForControlInGroup("PLAYER_FOOTER", [&](IControl* control) {
    if (auto* text = control->As<ITextControl>())
    {
      text->SetStr(footer.str().c_str());
      text->SetDirty(false);
    }
  });

  if (mPlayerPage != kAmpPage)
    graphics->ForControlInGroup("AMP_STAGE", [](IControl* control) { control->Hide(true); });
  graphics->SetAllControlsDirty();
}

void NeuralAmpModeler::_UpdatePlayerModeUI()
{
  auto* graphics = GetUI();
  if (graphics == nullptr)
    return;
  // Player mode is intentionally not a visual choice in DESS AMP. A legacy
  // state may still carry the hidden compatibility value, but the editor always
  // presents the portable-amplifier workflow and enables it on the next load.
  _UpdatePlayerControls();
  const DessAmpUI::ResponsiveLayout layout(graphics->GetBounds());
  if (auto* chrome = graphics->GetControlWithTag(kCtrlTagPlayerGlassChrome))
    chrome->As<DessAmpGlassChromeControl>()->SetPage(mPlayerPage);
  const bool hideRig = layout.phone && mPlayerPage != kRigPage;
  for (int tag : {kCtrlTagPlayerInputKnob, kCtrlTagPlayerGateKnob, kCtrlTagPlayerGateToggle,
                  kCtrlTagInputMeter, kCtrlTagOutputMeter, kCtrlTagPlayerOutputKnob})
    if (auto* control = graphics->GetControlWithTag(tag))
      control->Hide(hideRig);
  if (auto* selector = graphics->GetControlWithTag(kCtrlTagPlayerManifest))
  {
    const bool quick = layout.phone && mPlayerPage != kRigPage;
    selector->As<DessAmpModelSelectorControl>()->SetQuick(quick);
    selector->SetTargetAndDrawRECTs(quick ? layout.quickPreset : layout.model);
  }
  graphics->ForControlInGroup("PRE_STAGE", [this](IControl* control) { control->Hide(mPlayerPage != kPrePage); });
  graphics->ForControlInGroup("CAB_STAGE", [this](IControl* control) { control->Hide(mPlayerPage != kCabPage); });
  graphics->SetAllControlsDirty();
}

void NeuralAmpModeler::_ToggleAutoInputMatch()
{
  using State = dessamp::AutoInputMatch::State;
  const auto state = mAutoInputMatch.GetState();
  if (state == State::Requested || state == State::Measuring || state == State::Complete)
  {
    mAutoInputMatch.Cancel();
    mAutoInputStatus = "Match cancelled.";
    return;
  }
  if (!mAudioConfigInitialized.load() || !_ModelInputReferenceDBU())
  {
    mAutoInputStatus = "Enable an amp and audio processing first.";
    return;
  }
  mAutoInputAmpRequest = mAmpLoadRequest.load();
  mAutoInputBefore = GetParam(kInputLevel)->Value();
  mAutoCalibrationBefore = GetParam(kInputCalibrationLevel)->Value();
  mAutoCalibrationEnabled = GetParam(kCalibrateInput)->Bool();
  mAutoInputStarted = std::chrono::steady_clock::now();
  mAutoInputStatus = "Listening for five seconds.";
  mAutoInputMatch.Start();
}

void NeuralAmpModeler::_SetMatchedInput(bool enableCalibration, double trim)
{
  const double previousTrim = GetParam(kInputLevel)->Value();
  if (trim == previousTrim && enableCalibration == GetParam(kCalibrateInput)->Bool()) return;
  // Apply a cut before a boost so an audio block between the two host gestures
  // cannot see more gain than either the old or the new total.
  if (trim < previousTrim)
    _SetParameterValueFromMainThread(kInputLevel, GetParam(kInputLevel)->ToNormalized(trim));
  if (GetParam(kCalibrateInput)->Bool() != enableCalibration)
    _SetParameterValueFromMainThread(kCalibrateInput, enableCalibration ? 1.0 : 0.0);
  if (trim > previousTrim)
    _SetParameterValueFromMainThread(kInputLevel, GetParam(kInputLevel)->ToNormalized(trim));
}

void NeuralAmpModeler::_ResetInput()
{
  mAutoInputMatch.Cancel();
  mMatchedInput.reset();
  _SetMatchedInput(false, 0.0);
  mAutoInputStatus = "Reset. Interface level kept.";
}

void NeuralAmpModeler::_UpdateAutoInputMatch()
{
  using State = dessamp::AutoInputMatch::State;
  const auto state = mAutoInputMatch.GetState();
  const auto reference = _ModelInputReferenceDBU();
  if (mMatchedInput && (mMatchedInput->ampRequest != mAmpLoadRequest.load() || !GetParam(kCalibrateInput)->Bool()
      || mMatchedInput->trimDB != GetParam(kInputLevel)->Value()
      || mMatchedInput->headroomDBU != GetParam(kInputCalibrationLevel)->Value()))
  {
    mMatchedInput.reset();
    mAutoInputStatus.clear();
  }
  if (state == State::Interrupted)
  {
    mAutoInputMatch.Cancel();
    mAutoInputStatus = "Audio rate changed. Match again.";
  }
  bool active = state == State::Requested || state == State::Measuring;
  if (active || state == State::Complete)
  {
    if (mAutoInputAmpRequest != mAmpLoadRequest.load() || !reference
        || mAutoInputBefore != GetParam(kInputLevel)->Value()
        || mAutoCalibrationBefore != GetParam(kInputCalibrationLevel)->Value()
        || mAutoCalibrationEnabled != GetParam(kCalibrateInput)->Bool())
    {
      mAutoInputMatch.Cancel();
      mAutoInputStatus = "Settings changed. Match again.";
      active = false;
    }
    else if (state == State::Complete)
    {
      const double peak = mAutoInputMatch.Peak();
      if (mAutoInputMatch.Clipped())
        mAutoInputStatus = "Clipping. Lower interface gain and retry.";
      else if (peak < 0.0031622776601683794) // -50 dBFS, reject silence/very low input
        mAutoInputStatus = "Input too quiet. Check routing.";
      else
      {
        const double requested = dessamp::AutoInputMatch::TrimDB(peak, mAutoCalibrationBefore - *reference);
        const double trim = std::round(requested * 10.0) / 10.0;
        if (trim < GetParam(kInputLevel)->GetMin() || trim > GetParam(kInputLevel)->GetMax())
          mAutoInputStatus = "Outside trim range. Check interface gain.";
        else
        {
          _SetMatchedInput(true, trim);
          mMatchedInput = MatchedInput{mAutoInputAmpRequest, GetParam(kInputLevel)->Value(), mAutoCalibrationBefore};
          mAutoInputStatus = "Matched to -12 dBFS peak.";
        }
      }
      mAutoInputMatch.Cancel(); // consume once; no continuous gain following
      active = false;
    }
    else if (std::chrono::steady_clock::now() - mAutoInputStarted > std::chrono::seconds(12))
    {
      mAutoInputMatch.Cancel();
      mAutoInputStatus = "No audio. Enable input monitoring.";
      active = false;
    }
    else
      mAutoInputStatus = "Listening... " + std::to_string(mAutoInputMatch.Progress()) + "%";
  }
  if (auto* graphics = GetUI())
    if (auto* settings = graphics->GetControlWithTag(kCtrlTagSettingsBox))
    {
      auto* panel = settings->As<DessAmpSettingsPanelControl>();
      panel->SetModelInputReference(reference, mModelMetadataHasInputLevel.load(std::memory_order_relaxed));
      panel->SetAutoInputStatus(mAutoInputStatus, active, mMatchedInput.has_value());
    }
}

void NeuralAmpModeler::_UpdateLatency()
{
  int latency = 0;
  // Latency follows the adopted processing object, not the newest requested
  // selection. A prepared replacement can wait in the handoff slot without
  // changing either the live signal path or host PDC.
  mModelMetadataRequest.load(std::memory_order_acquire);
  if (mModel.load(std::memory_order_acquire) != nullptr
      && mModelMetadataEpoch.load(std::memory_order_relaxed) == mAudioConfigEpoch.load(std::memory_order_acquire))
    latency += mModelMetadataLatency.load(std::memory_order_relaxed);
  {
    // The loader mutex prevents reclamation while inspecting the adopted
    // Boost. Render-thread swaps may retire it, but cannot destroy it here.
    std::lock_guard<std::mutex> lock(mDSPLoadMutex);
    auto* boost = mBoostModel.load(std::memory_order_acquire);
    if (mBoostActiveTarget.load(std::memory_order_acquire) && boost != nullptr
        && boost->GetPreparedEpoch() == mAudioConfigEpoch.load(std::memory_order_acquire))
      latency += boost->GetLatency();
  }

  // Feels weird to have to do this.
  if (GetLatency() != latency)
  {
    SetLatency(latency);
  }
}

void NeuralAmpModeler::_UpdateMeters(sample** inputPointer, sample** outputPointer, const size_t nFrames,
                                     const size_t nChansIn, const size_t nChansOut)
{
  // Right now, we didn't specify MAXNC when we initialized these, so it's 1.
  if (nChansIn > 0 && inputPointer && inputPointer[0])
    mInputSender.ProcessBlock(inputPointer, (int)nFrames, kCtrlTagInputMeter, 1);
  if (nChansOut > 0 && outputPointer && outputPointer[0])
    mOutputSender.ProcessBlock(outputPointer, (int)nFrames, kCtrlTagOutputMeter, 1);
}


void NeuralAmpModeler::_InitBoost()
{
  mBoostModel.store(nullptr, std::memory_order_release);
  _SetBoostNAMPath("");
}

std::string NeuralAmpModeler::_StageBoostModel(const WDL_String& modelPath, const unsigned int requestToken,
                                               const int requestedBoostIndex)
{
  const std::uint64_t preparedEpoch = mAudioConfigEpoch.load(std::memory_order_acquire);
  if ((preparedEpoch & 1U) != 0U)
    return "";
  const double sampleRate = mAudioSampleRate.load(std::memory_order_acquire);
  const int maxBlockSize = mAudioMaxBlockSize.load(std::memory_order_acquire);
  try
  {
    auto dspPath = std::filesystem::u8path(modelPath.Get());
    std::unique_ptr<nam::DSP> model = nam::get_dsp(dspPath);
    const int controls = model->NumGlobalParams();
    if (controls != 0 && (requestedBoostIndex != kVerdantDriveIndex || controls != 1))
      return "Unsupported drive control layout";
    std::unique_ptr<ResamplingNAM> temp = std::make_unique<ResamplingNAM>(std::move(model), sampleRate);
    temp->Reset(sampleRate, maxBlockSize);
    temp->SetPreparedEpoch(preparedEpoch);
    temp->SetPreparedRequest(requestToken, requestedBoostIndex);
    temp->SetPreparedPath(modelPath.Get());
    if (preparedEpoch != mAudioConfigEpoch.load(std::memory_order_acquire)
        || requestToken != mBoostLoadRequest.load(std::memory_order_acquire)
        || requestedBoostIndex != mBoostModelIdx.load(std::memory_order_acquire)
        || !mBoostActiveTarget.load(std::memory_order_acquire))
      return "";
    delete mPendingBoostModel.exchange(temp.release(), std::memory_order_acq_rel);
  }
  catch (const std::exception& e)
  {
    return e.what();
  }
  return "";
}

std::string NeuralAmpModeler::_LoadBoostModel(const std::string& modelName, const unsigned int requestToken,
                                              const int requestedBoostIndex)
{
  WDL_String resourcePath;
  PlayerResourcePath(resourcePath, GetBundleID());

  if (resourcePath.GetLength() == 0)
    return "Failed to get resource path";

  WDL_String path;
  path.Set(resourcePath.Get());
  path.Append("/models/DessDrive/");
  path.Append(modelName.c_str());
  path.Append(".nam");

  return _StageBoostModel(path, requestToken, requestedBoostIndex);
}

std::string NeuralAmpModeler::_GetBoostModelName(int index)
{
  switch (index)
  {
    case 0: return "OD808";
    case 1: return "SD1";
    case 2: return "TS9";
    case 3: return "aesahaettr";
    default: return "OD808";
  }
}

// HACK
#include "Unserialization.cpp"
