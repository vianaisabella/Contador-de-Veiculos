@echo off
REM ============================================================
REM Gera o executavel em dist\ContadorVeiculos\
REM
REM Use um ambiente separado, SEM torch/ultralytics, para o .exe
REM ficar pequeno:
REM     python -m venv venv-exe
REM     venv-exe\Scripts\activate
REM     pip install -r requirements.txt pyinstaller
REM
REM Antes, gere o modelo (no ambiente de desenvolvimento):
REM     python scripts\exportar_modelo.py
REM ============================================================

if not exist modelos (
    echo Pasta modelos\ nao encontrada. Rode antes: python scripts\exportar_modelo.py
    pause
    exit /b 1
)

pyinstaller --noconfirm --windowed --name ContadorVeiculos ^
    --add-data "modelos;modelos" ^
    --collect-all openvino ^
    --exclude-module torch --exclude-module torchvision --exclude-module ultralytics ^
    --exclude-module matplotlib --exclude-module pandas ^
    Main.py

echo.
echo Pronto: dist\ContadorVeiculos\ContadorVeiculos.exe
echo Copie a pasta dist\ContadorVeiculos inteira para a outra maquina.
pause
