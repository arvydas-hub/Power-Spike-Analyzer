@echo off
setlocal EnableExtensions
echo [BUILD] Starting the build and sign process...
echo.

rem --- Step 1: Activate Virtual Environment ---
set "VENV_ACTIVATE=.venv\Scripts\activate.bat"
if not exist "%VENV_ACTIVATE%" (
    echo [BUILD] ERROR: Virtual environment not found at "%VENV_ACTIVATE%"
    exit /b 1
)
echo [BUILD] Activating virtual environment...
call "%VENV_ACTIVATE%"
echo.

rem --- Step 2: Clean Up Old Builds (Optional but Recommended) ---
echo [BUILD] Cleaning up old build artifacts...
if exist "dist" rd /s /q "dist"
if exist "build" rd /s /q "build"
if exist "main.spec" del "main.spec"
echo.

rem --- Step 3: Compile the Executable with PyInstaller ---
echo [BUILD] Compiling main.py into an executable...
rem --onefile: Bundle into a single .exe
rem --windowed: Hide the black console window for this GUI app
pyinstaller --onefile --windowed main.py

rem --- Check if compilation was successful ---
if errorlevel 1 (
    echo [BUILD] ERROR: PyInstaller failed with error code %ERRORLEVEL%.
    exit /b %ERRORLEVEL%
)
if not exist "dist\main.exe" (
    echo [BUILD] ERROR: Expected executable not found at dist\main.exe.
    exit /b 1
)
echo [BUILD] Compilation successful.
echo.

rem --- Step 4: Sign the New Executable ---
echo [BUILD] Calling the signing script...
call sign.bat "dist\main.exe"

rem --- Final check ---
if errorlevel 1 (
    echo [BUILD] ERROR: The signing script failed.
    exit /b %ERRORLEVEL%
)

echo.
echo [BUILD] Process complete! The signed executable is in the 'dist' folder.
exit /b 0