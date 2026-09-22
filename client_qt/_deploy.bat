@echo off
setlocal

REM NOTE: keep this file pure ASCII. cmd.exe reads .bat line by line using the
REM SYSTEM code page (GBK on this machine), so UTF-8 Chinese comments turn into
REM garbage that gets executed as commands and breaks the whole script.

REM ---------------------------------------------------------------------------
REM Step 11: build + package the Qt client into a self-contained folder
REM
REM Why is one .exe not enough?
REM   A Qt app looks for Qt5Core.dll / Qt5Gui.dll / Qt5Widgets.dll /
REM   Qt5Network.dll plus the platform plugin platforms\qwindows.dll at runtime.
REM   Copy just the exe to another machine and double-click does nothing.
REM   windeployqt is the official tool that copies those dependencies over.
REM
REM Toolchain locations are MACHINE CONFIG, not build logic: they live in
REM local_env.bat next to this script (not published). You can also just set
REM the two variables in your shell before running this file:
REM   LIBS_VS_DIR   e.g. C:\Program Files\Microsoft Visual Studio\2022\Community
REM   LIBS_QT_DIR   e.g. C:\Qt\5.15.2\msvc2019_64
REM ---------------------------------------------------------------------------

if exist "%~dp0local_env.bat" call "%~dp0local_env.bat"

if not defined LIBS_VS_DIR (
    echo RESULT=LIBS_VS_DIR_NOT_SET
    echo Create local_env.bat next to this script containing:
    echo   set LIBS_VS_DIR=C:\Program Files\Microsoft Visual Studio\2022\Community
    echo or set the variable in your shell before running _deploy.bat
    exit /b 1
)
if not defined LIBS_QT_DIR (
    echo RESULT=LIBS_QT_DIR_NOT_SET
    echo Create local_env.bat next to this script containing:
    echo   set LIBS_QT_DIR=C:\Qt\5.15.2\msvc2019_64
    echo or set the variable in your shell before running _deploy.bat
    exit /b 1
)

set VSDIR=%LIBS_VS_DIR%
set QTDIR=%LIBS_QT_DIR%
set CMK=%VSDIR%\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe
set NINJA=%VSDIR%\Common7\IDE\CommonExtensions\Microsoft\CMake\Ninja\ninja.exe
set SRC=%~dp0
set DIST=%SRC%dist

if not exist "%VSDIR%\VC\Auxiliary\Build\vcvars64.bat" (
    echo RESULT=VS_NOT_FOUND  "%VSDIR%"
    exit /b 1
)
if not exist "%QTDIR%\bin\windeployqt.exe" (
    echo RESULT=QT_NOT_FOUND  "%QTDIR%"
    exit /b 1
)

echo [1/4] vcvars64.bat
call "%VSDIR%\VC\Auxiliary\Build\vcvars64.bat" >nul 2>&1
if errorlevel 1 ( echo RESULT=VCVARS_FAILED & exit /b 1 )

cd /d "%SRC%"

echo [2/4] cmake configure + build
"%CMK%" -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_MAKE_PROGRAM="%NINJA%" -DCMAKE_PREFIX_PATH="%QTDIR%" >nul
if errorlevel 1 ( echo RESULT=CONFIGURE_FAILED & exit /b 2 )
"%CMK%" --build build
if errorlevel 1 ( echo RESULT=BUILD_FAILED & exit /b 3 )

echo [3/4] prepare dist folder
if exist "%DIST%" rmdir /s /q "%DIST%"
mkdir "%DIST%"
copy /y "build\libs_agent_client.exe" "%DIST%\" >nul
if errorlevel 1 ( echo RESULT=COPY_FAILED & exit /b 4 )

echo [4/4] windeployqt
REM --no-translations    skip the .qm language packs (tens of MB, unused here)
REM --no-system-d3d-compiler / --no-opengl-sw
REM                     skip the D3D compiler and software OpenGL fallback;
REM                     those are for machines with no GPU driver.
"%QTDIR%\bin\windeployqt.exe" --release --no-translations --no-system-d3d-compiler --no-opengl-sw "%DIST%\libs_agent_client.exe"
if errorlevel 1 ( echo RESULT=DEPLOY_FAILED & exit /b 5 )

echo RESULT=DEPLOY_OK
