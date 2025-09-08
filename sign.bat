@echo off
setlocal EnableExtensions

rem ===== robust code signer (hard-coded path + thumbprint) =====
set "TARGET=%~1"
if not defined TARGET set "TARGET=dist\main.exe"

rem -- tools / cert (adjust if you move SDK or rotate cert) --
set "SIGNTOOL=C:\Program Files (x86)\Windows Kits\10\bin\10.0.26100.0\x64\signtool.exe"
set "THUMB=B3D14E7A69CE39EAAE2579DA26F88DB691127319"
set "TSA=http://timestamp.digicert.com"

echo [sign] Target     : "%TARGET%"
echo [sign] Signtool   : "%SIGNTOOL%"
echo [sign] Thumbprint : %THUMB%
echo [sign] TSA        : %TSA%

if not exist "%TARGET%" (
  echo [sign] ERROR: target not found: "%TARGET%"
  exit /b 1
)
if not exist "%SIGNTOOL%" (
  echo [sign] ERROR: signtool not found at "%SIGNTOOL%"
  exit /b 2
)

"%SIGNTOOL%" sign /fd SHA256 /sha1 %THUMB% /sm /tr "%TSA%" /td SHA256 /as "%TARGET%"
if errorlevel 1 (
  echo [sign] ERROR: signtool sign failed with %ERRORLEVEL%
  exit /b %ERRORLEVEL%
)

echo [sign] Verifying...
for /f "usebackq delims=" %%S in (`
  powershell -NoProfile -Command "(Get-AuthenticodeSignature '%TARGET%').Status"
`) do set "SIGSTATUS=%%S"

set "SIGSTATUS=%SIGSTATUS: =%"
echo [sign] Authenticode Status: %SIGSTATUS%
if /I not "%SIGSTATUS%"=="Valid" (
  powershell -NoProfile -Command "Get-AuthenticodeSignature '%TARGET%' | fl Status,StatusMessage,SignerCertificate,TimeStamperCertificate"
  echo [sign] ERROR: Signature status is %SIGSTATUS%
  exit /b 5
)

echo [sign] OK: signed and verified.
exit /b 0
