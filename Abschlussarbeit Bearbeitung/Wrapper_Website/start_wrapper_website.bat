@echo off
title Wrapper Website - Flask Starter

echo Starte Flask Server...
echo.

REM Project
set BASEDIR=%~dp0

REM Flask start application
start "" cmd /k "python app.py"

REM wait for flask to start
timeout /t 2 >nul

REM start in browser
start http://127.0.0.1:5000

echo Website has been started!
echo Windows can be closed!
