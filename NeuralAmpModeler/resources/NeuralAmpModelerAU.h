#include <TargetConditionals.h>
#if TARGET_OS_IOS == 1
  #import <UIKit/UIKit.h>
#else
  #import <Cocoa/Cocoa.h>
#endif

// Compatibility filename retained by the inherited iPlug2 iOS project. The
// public runtime identity is DessAmp and the Objective-C prefix is vDessAmp.
#define IPLUG_AUVIEWCONTROLLER IPlugAUViewController_vDessAmp
#define IPLUG_AUAUDIOUNIT IPlugAUAudioUnit_vDessAmp
#import <NeuralAmpModelerAU/IPlugAUAudioUnit.h>
#import <NeuralAmpModelerAU/IPlugAUViewController.h>

FOUNDATION_EXPORT double NeuralAmpModelerAUVersionNumber;
FOUNDATION_EXPORT const unsigned char NeuralAmpModelerAUVersionString[];
