// Device-independent visual QA host: loads the exact supplied AU, never CoreAudio hardware.
#define main DessAmpIntegrationTestMain
#include "test_au_player.mm"
#undef main
#import <Cocoa/Cocoa.h>
#import <AudioUnit/AUCocoaUIView.h>
#include <fstream>
#include "factory_test_profile.h"

int main(int argc, char** argv)
{
  @autoreleasepool {
    try {
      if (argc != 5) throw std::runtime_error("usage: au_editor_host binary profile.json slot screenshot.png");
      [NSApplication sharedApplication];
      [NSApp setActivationPolicy:NSApplicationActivationPolicyRegular];
      void* handle = nullptr;
      auto unit = CreateUnit(RegisterComponentBinary(argv[1], handle));
      std::ifstream input(argv[2]);
      const auto profile = nlohmann::json::parse(input);
      const int slot = std::stoi(argv[3]);
      const auto& entry = profile.at("amplifiers").at(slot - 1);
      const auto params = FactoryTestParameters<kParameterCount>(entry);
      const auto resources = std::filesystem::path(argv[1]).parent_path().parent_path() / "Resources";
      const auto model = (resources / "PrivateValidation/Amplifiers" / ("amp-0" + std::to_string(slot) + ".dess")).string();
      const auto cabinet = (resources / "IRs" / entry.value("cabinet", std::string("Desfire.wav"))).string();
      if (slot != 1) {
        auto state = MakeStateDictionary(model, cabinet, params); ApplyState(unit, state); CFRelease(state);
      }
      RenderRealtime(unit, 48000);
      VerifyHostParameters(unit, params);
      InputState inputState;
      ConfigureUnit(unit, inputState);
      RequireStatus(AudioUnitInitialize(unit), "initialize offline visual host");
      __block double sampleTime = 0;
      [NSTimer scheduledTimerWithTimeInterval:0.016 repeats:YES block:^(NSTimer*) {
        std::array<Float32, kBlockSize> samples{};
        for (int index = 0; index < 12; ++index) {
          AudioBufferList output{};
          output.mNumberBuffers = 1;
          output.mBuffers[0] = {1, kBlockSize * sizeof(Float32), samples.data()};
          AudioTimeStamp stamp{}; stamp.mFlags = kAudioTimeStampSampleTimeValid; stamp.mSampleTime = sampleTime;
          AudioUnitRenderActionFlags flags = 0;
          AudioUnitRender(unit, &flags, &stamp, 0, kBlockSize, &output);
          sampleTime += kBlockSize;
        }
      }];
      UInt32 size = 0;
      RequireStatus(AudioUnitGetPropertyInfo(unit, kAudioUnitProperty_CocoaUI, kAudioUnitScope_Global, 0, &size, nullptr), "Cocoa UI size");
      std::vector<unsigned char> infoBytes(size);
      RequireStatus(AudioUnitGetProperty(unit, kAudioUnitProperty_CocoaUI, kAudioUnitScope_Global, 0, infoBytes.data(), &size), "Cocoa UI");
      auto* info = reinterpret_cast<AudioUnitCocoaViewInfo*>(infoBytes.data());
      NSBundle* bundle = [NSBundle bundleWithURL:(__bridge NSURL*)info->mCocoaAUViewBundleLocation];
      [bundle load];
      Class viewClass = [bundle classNamed:(__bridge NSString*)info->mCocoaAUViewClass[0]];
      id<AUCocoaUIBase> factory = [[viewClass alloc] init];
      NSView* view = [factory uiViewForAudioUnit:unit withSize:NSMakeSize(1200, 760)];
      if (!view) throw std::runtime_error("No exact AU editor");
      NSWindow* window = [[NSWindow alloc] initWithContentRect:view.frame styleMask:NSWindowStyleMaskBorderless backing:NSBackingStoreBuffered defer:NO];
      window.title = @"DESS AMP — exact AU visual QA (no audio device)";
      [window setContentView:view]; [window center]; [window makeKeyAndOrderFront:nil]; [NSApp activateIgnoringOtherApps:YES];
      // Capture the isolated window with screencapture; Metal-backed views cannot
      // be captured by NSView's CPU bitmap cache. The final argument is reserved.
      [NSApp run];
      AudioComponentInstanceDispose(unit);
    } catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
  }
  return 0;
}
