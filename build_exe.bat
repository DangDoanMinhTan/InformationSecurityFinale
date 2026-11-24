@echo off
cd /d "%~dp0"
py -m pip install -r requirements.txt pyinstaller
py -m PyInstaller --noconfirm --clean --windowed --onefile --name IrisRecognition gui.py
echo.
echo Done. Your program is dist\IrisRecognition.exe
pause
