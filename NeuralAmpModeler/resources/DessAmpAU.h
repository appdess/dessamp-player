
#include <TargetConditionals.h>
#if TARGET_OS_IOS == 1
  #import <UIKit/UIKit.h>
#else
  #import <Cocoa/Cocoa.h>
#endif

#define IPLUG_AUVIEWCONTROLLER IPlugAUViewController_vDessAmp
#define IPLUG_AUAUDIOUNIT IPlugAUAudioUnit_vDessAmp
#import <DessAmpAU/IPlugAUAudioUnit.h>
#import <DessAmpAU/IPlugAUViewController.h>

//! Project version number for DessAmpAU.
FOUNDATION_EXPORT double DessAmpAUVersionNumber;

//! Project version string for DessAmpAU.
FOUNDATION_EXPORT const unsigned char DessAmpAUVersionString[];

@class IPlugAUViewController_vDessAmp;
