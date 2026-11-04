#define PLUG_NAME "DessAmp"
#define PLUG_MFR "Alexander Dess"
#define PLUG_VERSION_HEX 0x000200
#define PLUG_VERSION_STR "0.2.0"
#define PLUG_UNIQUE_ID 'DsAP'
#define PLUG_MFR_ID 'AdMs'
#define PLUG_URL_STR "https://github.com/appdess/dessamp-player"
#define PLUG_EMAIL_STR "alex@dess.metal"
#define PLUG_COPYRIGHT_STR "Copyright 2026 Alexander Dess; Original content Copyright Steven Atkinson"
#define PLUG_CLASS_NAME NeuralAmpModeler
#define BUNDLE_NAME "DessAmp"
#define BUNDLE_MFR "AlexanderDess"
#define BUNDLE_DOMAIN "com"

#define SHARED_RESOURCES_SUBPATH "DessAmp"

#ifdef APP_API
  #define PLUG_CHANNEL_IO "1-2"
#else
  #define PLUG_CHANNEL_IO "1-1 1-2 2-2"
#endif

#define PLUG_LATENCY 0
#define PLUG_TYPE 0
#define PLUG_DOES_MIDI_IN 0
#define PLUG_DOES_MIDI_OUT 0
#define PLUG_DOES_MPE 0
#define PLUG_DOES_STATE_CHUNKS 1
#define PLUG_HAS_UI 1
#define PLUG_WIDTH 1280
#define PLUG_HEIGHT 800
#define PLUG_FPS 60
#define PLUG_SHARED_RESOURCES 0
#ifdef OS_IOS
  // Required for device rotation and AUv3 host view-configuration changes.
  // The editor owns responsive geometry; the host supplies the viewport.
  #define PLUG_HOST_RESIZE 1
  // A portrait phone is narrower than the default half-desktop minimum (640 pt).
  #define PLUG_MIN_WIDTH 320
  #define PLUG_MIN_HEIGHT 240
#else
  #define PLUG_HOST_RESIZE 0
#endif
#define PLUG_MAX_WIDTH PLUG_WIDTH * 4
#define PLUG_MAX_HEIGHT PLUG_HEIGHT * 4

#define AUV2_ENTRY DessAmp_Entry
#define AUV2_ENTRY_STR "DessAmp_Entry"
#define AUV2_FACTORY DessAmp_Factory
#define AUV2_VIEW_CLASS DessAmp_View
#define AUV2_VIEW_CLASS_STR "DessAmp_View"

#define AAX_TYPE_IDS 'ITP1'
#define AAX_TYPE_IDS_AUDIOSUITE 'ITA1'
#define AAX_PLUG_MFR_STR "Acme"
#define AAX_PLUG_NAME_STR "DessAmp\nIPEF"
#define AAX_PLUG_CATEGORY_STR "Effect"
#define AAX_DOES_AUDIOSUITE 1

#define VST3_SUBCATEGORY "Fx"

#define APP_NUM_CHANNELS 2
#define APP_N_VECTOR_WAIT 0
#define APP_MULT 1
#define APP_COPY_AUV3 0
#define APP_SIGNAL_VECTOR_SIZE 64

#define ROBOTO_FN "Roboto-Regular.ttf"
#define ROBOTO_SEMIBOLD_FN "Roboto-SemiBold.ttf"
#define MICHROMA_FN "Michroma-Regular.ttf"

#ifdef OS_IOS
  // iOS keeps resources beside the executable, so a directory named DessAmp
  // would collide with the application binary itself.
  #define DESSAMP_HEAD_SHELL_FN "amp-head-shell@2x.png"
  #define DESSAMP_DUAL_RED_HEAD_SHELL_FN "amp-head-shell-dual-red-modern-v1@2x.png"
  #define DESSAMP_FORGE_HEAD_SHELL_FN "amp-head-shell-forge-streamlined-v1@2x.png"
  #define DESSAMP_FIRE_HEAD_SHELL_FN "amp-head-shell-fire-chrome-v1@2x.png"
  #define DESSAMP_KNOB_FACE_FN "amp-knob-face@2x.png"
  #define DESSAMP_BADGE_FN "dess-badge@2x.png"
  #define DESSAMP_WORDMARK_FN "dess-amp-wordmark@2x.png"
  #define DESSAMP_OXIDE_FN "studio-oxide@2x.png"
  #define DESSAMP_AMBER_FN "studio-amber@2x.png"
  #define DESSAMP_VERDANT_FN "studio-verdant@2x.png"
  #define DESSAMP_NOCTURNE_FN "studio-nocturne@2x.png"
  #define DESSAMP_CAB_FN "studio-cab@2x.png"
#else
  #define DESSAMP_HEAD_SHELL_FN "DessAmp/amp-head-shell@2x.png"
  #define DESSAMP_DUAL_RED_HEAD_SHELL_FN "DessAmp/amp-head-shell-dual-red-modern-v1@2x.png"
  #define DESSAMP_FORGE_HEAD_SHELL_FN "DessAmp/amp-head-shell-forge-streamlined-v1@2x.png"
  #define DESSAMP_FIRE_HEAD_SHELL_FN "DessAmp/amp-head-shell-fire-chrome-v1@2x.png"
  #define DESSAMP_KNOB_FACE_FN "DessAmp/amp-knob-face@2x.png"
  #define DESSAMP_BADGE_FN "DessAmp/dess-badge@2x.png"
  #define DESSAMP_WORDMARK_FN "DessAmp/dess-amp-wordmark@2x.png"
  #define DESSAMP_OXIDE_FN "DessAmp/studio-oxide@2x.png"
  #define DESSAMP_AMBER_FN "DessAmp/studio-amber@2x.png"
  #define DESSAMP_VERDANT_FN "DessAmp/studio-verdant@2x.png"
  #define DESSAMP_NOCTURNE_FN "DessAmp/studio-nocturne@2x.png"
  #define DESSAMP_CAB_FN "DessAmp/studio-cab@2x.png"
#endif
