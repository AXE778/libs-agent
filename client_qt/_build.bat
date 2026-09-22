@echo off
setlocal

REM NOTE: keep this file pure ASCII (cmd.exe reads .bat with the system code page).

REM ---------------------------------------------------------------------------
REM Build only -- no windeployqt, no dist folder. Much faster than _deploy.bat,
REM use this while writing code; use _deploy.bat when you want a runnable folder.
REM
REM Toolchain paths come from local_env.bat (or from the environment).
REM ---------------------------------------------------------------------------

if exist "%~dp0local_env.bat" call "%~dp0local_env.bat"

if not defined LIBS_VS_DIR (
    echo RESULT=LIBS_VS_DIR_NOT_SET  (see _deploy.bat header for how to set it)
    exit /b 1
)
if not defined LIBS_QT_DIR (
    echo RESULT=LIBS_QT_DIR_NOT_SET  (see _deploy.bat header for how to set it)
    exit /b 1
)

set VSDIR=%LIBS_VS_DIR%
set QTDIR=%LIBS_QT_DIR%
set CMK=%VSDIR%\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe
set NINJA=%VSDIR%\Common7\IDE\CommonExtensions\Microsoft\CMake\Ninja\ninja.exe

call "%VSDIR%\VC\Auxiliary\Build\vcvars64.bat" >nul 2>&1
if errorlevel 1 ( echo RESULT=VCVARS_FAILED & exit /b 1 )

cd /d "%~dp0"

"%CMK%" -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_MAKE_PROGRAM="%NINJA%" -DCMAKE_PREFIX_PATH="%QTDIR%" >nul
if errorlevel 1 ( echo RESULT=CONFIGURE_FAILED & exit /b 2 )

"%CMK%" --build build
if errorlevel 1 ( echo RESULT=BUILD_FAILED & exit /b 3 )

echo RESULT=BUILD_OK
