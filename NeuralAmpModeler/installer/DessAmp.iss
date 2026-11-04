#include "..\build-win\installer-settings.iss"
[Setup]
AppName=DessAmp
AppContact=alex@dess.metal
AppCopyright=Copyright (C) 2026 Alexander Dess
AppPublisher=Alexander Dess
AppVersion=0.2.0
VersionInfoVersion=0.2.0
DefaultDirName={autopf}\DessAmp
DefaultGroupName=DessAmp
Compression=lzma2
SolidCompression=yes
OutputDir=..\build-win\installer
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
OutputBaseFilename={#InstallerName}
LicenseFile=license.rtf
SetupLogging=yes

[Types]
Name: "full"; Description: "Full installation"
Name: "custom"; Description: "Custom installation"; Flags: iscustom

[Messages]
WelcomeLabel1=Welcome to the {#InstallerName}
SetupWindowTitle={#InstallerName}

[Components]
Name: "app"; Description: "Standalone application"; Types: full custom
Name: "vst3_64"; Description: "64-bit VST3 plug-in"; Types: full custom

[Files]
Source: "..\build-win\DessAmp_x64.exe"; DestDir: "{app}"; Components: app; Flags: ignoreversion
Source: "..\build-win\DessAmp.vst3\Contents\x86_64-win\DessAmp.vst3"; DestDir: "{commoncf64}\VST3\DessAmp.vst3\Contents\x86_64-win"; Components: vst3_64; Flags: ignoreversion
Source: "..\build-win\THIRD_PARTY_LICENSES\*"; DestDir: "{app}\THIRD_PARTY_LICENSES"; Flags: ignoreversion
#ifdef PrivateListening
#include "..\build-win\listening-files.iss"
#else
#include "..\build-win\factory-cabinets.iss"
#endif
Source: "{#ReadmeName}"; DestDir: "{app}"; DestName: "readme.rtf"; Flags: isreadme

[Icons]
Name: "{group}\DessAmp"; Filename: "{app}\DessAmp_x64.exe"; Components: app
Name: "{group}\Uninstall DessAmp"; Filename: "{uninstallexe}"
