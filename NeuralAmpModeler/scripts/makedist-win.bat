@echo off
setlocal
set "MODE=%~1"
if /I "%MODE%"=="listening" (
  set "MODE=listening"
  if /I not "%~2"=="installer" goto :usage
) else if /I "%MODE%"=="full" (set "MODE=full") else if /I "%MODE%"=="demo" (set "MODE=demo") else goto :usage
if /I "%~2"=="installer" (set "ZIP=0") else if /I "%~2"=="zip" (set "ZIP=1") else goto :usage
if not "%~3"=="" goto :usage
pushd "%~dp0.." || exit /b 1
where MSBuild.exe >nul 2>&1 || (echo Run this from a Visual Studio Developer Command Prompt with v143 installed. & goto :failed)
if "%ZIP%"=="0" (
  if not defined ISCC set "ISCC=%ProgramFiles(x86)%\Inno Setup 6\ISCC.exe"
)
python scripts\prepare_resources-win.py --distribution %MODE% || goto :failed
python scripts\update_installer-win.py %MODE% || goto :failed
msbuild NeuralAmpModeler.sln "/t:NeuralAmpModeler-app:Rebuild;NeuralAmpModeler-vst3:Rebuild" /p:Configuration=Release /p:Platform=x64 /nologo /verbosity:minimal || goto :failed
if not exist "build-win\DessAmp_x64.exe" goto :failed
if not exist "build-win\DessAmp.vst3\Contents\x86_64-win\DessAmp.vst3" goto :failed
if "%ZIP%"=="0" (
  "%ISCC%" installer\DessAmp.iss || goto :failed
)
python scripts\makezip-win.py %MODE% %ZIP% || goto :failed
popd
exit /b 0

:failed
echo DessAmp Windows distribution failed. 1>&2
popd
exit /b 1

:usage
echo Usage: %~nx0 full^|demo installer^|zip, or listening installer
exit /b 2
