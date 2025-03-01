@echo off
chcp 65001

pushd %~dp0

SET "SCRIPT_DIR=%CD:\=/%"

SET "EXPORT_DIR=%SCRIPT_DIR%/Output"

SET "F5_TTS_PROJECT_DIR=%SCRIPT_DIR%/F5-TTS"
SET "DOWNLOAD_DIR=%SCRIPT_DIR%/Downloads"
SET "EXPORT_PY_FILE=%SCRIPT_DIR%/Export_ONNX/F5_TTS/Export_F5.py"

SET "F5_TTS_MODEL_DIR=%DOWNLOAD_DIR%/F5-TTS"
SET "VOCOS_DIR=%DOWNLOAD_DIR%/vocos-mel-24khz"


@rem This for export_F5.py

SET "F5_project_path=%F5_TTS_PROJECT_DIR%"


@echo Work DIR: %SCRIPT_DIR%


if NOT EXIST "%DOWNLOAD_DIR%" (
    mkdir "%DOWNLOAD_DIR%"
)

if NOT EXIST "%EXPORT_DIR%" (
    mkdir "%EXPORT_DIR%"
)

if NOT EXIST "%F5_TTS_PROJECT_DIR%" (
    @echo Clone F5-TTS project ...
    @echo.
    git clone clone https://github.com/SWivid/F5-TTS.git
    
    cd "%F5_TTS_PROJECT_DIR%"
    pip install -e .
)

if NOT EXIST "%F5_TTS_MODEL_DIR%" (
    git clone https://huggingface.co/SWivid/F5-TTS "%F5_TTS_MODEL_DIR%"
)

if NOT EXIST "%VOCOS_DIR%" (
    git clone https://huggingface.co/charactr/vocos-mel-24khz "%VOCOS_DIR%"
)

pushd %SCRIPT_DIR%

pip install onnxruntime-tools==1.7.0

@rem change code for windows
@echo modify code ...
@echo.
call:ReplaceString "%EXPORT_PY_FILE%" "/home/DakeQQ/Downloads/F5-TTS-main" "%F5_TTS_PROJECT_DIR%"
call:ReplaceString "%EXPORT_PY_FILE%" "/home/DakeQQ/Downloads/F5TTS_Base" "%F5_TTS_MODEL_DIR%/F5TTS_Base"
call:ReplaceString "%EXPORT_PY_FILE%" "/home/DakeQQ/Downloads/vocos-mel-24khz" "%VOCOS_DIR%"
call:ReplaceString "%EXPORT_PY_FILE%" "/home/DakeQQ/Downloads/F5_ONNX/" "%EXPORT_DIR%/"


cd "%SCRIPT_DIR%/Export_ONNX/F5_TTS"
python ./Export_F5.py

@rem restore code

git checkout .

exit


:ReplaceString
set ff=%~1
set old_str=%~2
set new_str=%~3
powershell -Command "(gc -Encoding UTF8 %ff%) -replace '%old_str%', '%new_str%' | Set-Content -Encoding UTF8 %ff%"
GOTO:EOF
