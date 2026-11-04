 /*
 ==============================================================================
 
 This file is part of the iPlug 2 library. Copyright (C) the iPlug 2 developers. 
 
 See LICENSE.txt for  more info.
 
 ==============================================================================
*/

#import "AppViewController.h"
#import "IPlugAUPlayer.h"
#import "IPlugAUAudioUnit.h"

#include "config.h"

#import "IPlugAUViewController.h"
#import <CoreAudioKit/CoreAudioKit.h>

#if !__has_feature(objc_arc)
#error This file must be compiled with Arc. Use -fobjc-arc flag
#endif

@interface AppViewController ()
{
  IPlugAUPlayer* player;
  IPLUG_AUVIEWCONTROLLER* pluginVC;
  IBOutlet UIView* auView;
}
@end

@implementation AppViewController

- (BOOL) prefersStatusBarHidden
{
  return YES;
}

- (void) viewDidLoad
{
  [super viewDidLoad];

#if PLUG_HAS_UI
  NSString* storyBoardName = [NSString stringWithFormat:@"%s-iOS-MainInterface", PLUG_NAME];
  UIStoryboard* storyboard = [UIStoryboard storyboardWithName:storyBoardName bundle: nil];
  pluginVC = [storyboard instantiateViewControllerWithIdentifier:@"main"];
  [self addChildViewController:pluginVC];
#endif
  
  AudioComponentDescription desc;

#if PLUG_TYPE==0
#if PLUG_DOES_MIDI_IN
  desc.componentType = kAudioUnitType_MusicEffect;
#else
  desc.componentType = kAudioUnitType_Effect;
#endif
#elif PLUG_TYPE==1
  desc.componentType = kAudioUnitType_MusicDevice;
#elif PLUG_TYPE==2
  desc.componentType = 'aumi';
#endif

  desc.componentSubType = PLUG_UNIQUE_ID;
  desc.componentManufacturer = PLUG_MFR_ID;
  desc.componentFlags = 0;
  desc.componentFlagsMask = 0;

  [AUAudioUnit registerSubclass: IPLUG_AUAUDIOUNIT.class asComponentDescription:desc name:@"Local AUv3" version: UINT32_MAX];

  player = [[IPlugAUPlayer alloc] initWithComponentType:desc.componentType];

  [player loadAudioUnitWithComponentDescription:desc completion:^{
    // AVAudioUnit instantiation is asynchronous and can finish after this
    // container has already appeared. In that ordering, the child view
    // controller's earlier viewWillAppear: sees a nil audio unit and never
    // opens the plug-in editor, leaving the standalone app black. Complete all
    // UIKit work on the main queue, finish containment, and open the editor
    // explicitly once both the audio unit and its parent view exist.
    dispatch_async(dispatch_get_main_queue(), ^{
      IPLUG_AUAUDIOUNIT* audioUnit = (IPLUG_AUAUDIOUNIT*) self->player.currentAudioUnit;

      if (audioUnit == nil)
      {
        NSLog(@"Unable to instantiate the DessAmp AUv3 editor");
        return;
      }

      self->pluginVC.audioUnit = audioUnit;
      [self->pluginVC audioUnitInitialized];
      [self embedPlugInView];
      [self.view layoutIfNeeded];
      [self->pluginVC.view layoutIfNeeded];
      if ([audioUnit openWindow:self->pluginVC.view] == nil)
        NSLog(@"Unable to open the DessAmp AUv3 editor view");

      [audioUnit hostResized:self->pluginVC.view.bounds.size];
    });
  }];
  
  [[NSNotificationCenter defaultCenter] addObserver:self selector:@selector(receiveNotification:) name:@"LaunchBTMidiDialog" object:nil];
}

- (void) receiveNotification:(NSNotification*) notification
{
  if ([notification.name isEqualToString:@"LaunchBTMidiDialog"])
  {
    NSDictionary* dict = notification.userInfo;
    NSNumber* x = (NSNumber*) dict[@"x"];
    NSNumber* y = (NSNumber*) dict[@"y"];
   
    CABTMIDICentralViewController* vc = [[CABTMIDICentralViewController alloc] init];
    UINavigationController* nc = [[UINavigationController alloc] initWithRootViewController:vc];
    nc.modalPresentationStyle = UIModalPresentationPopover;
    
    UIPopoverPresentationController* ppc = nc.popoverPresentationController;
    ppc.permittedArrowDirections = UIPopoverArrowDirectionAny;
    ppc.sourceView = self.view;
    ppc.sourceRect = CGRectMake([x floatValue], [y floatValue], 1., 1.);
    
    [self presentViewController:nc animated:YES completion:nil];
  }
}

- (void) embedPlugInView
{
#if PLUG_HAS_UI
  UIView* view = pluginVC.view;
  view.frame = auView.bounds;
  [auView addSubview: view];

  view.translatesAutoresizingMaskIntoConstraints = NO;

  NSArray* constraints = [NSLayoutConstraint constraintsWithVisualFormat: @"H:|[view]|" options:0 metrics:nil views:NSDictionaryOfVariableBindings(view)];
  [auView addConstraints: constraints];

  constraints = [NSLayoutConstraint constraintsWithVisualFormat: @"V:|[view]|" options:0 metrics:nil views:NSDictionaryOfVariableBindings(view)];
  [auView addConstraints: constraints];

  [pluginVC didMoveToParentViewController:self];
#endif
}

- (UIRectEdge) preferredScreenEdgesDeferringSystemGestures
{
  return UIRectEdgeAll;
}
@end
