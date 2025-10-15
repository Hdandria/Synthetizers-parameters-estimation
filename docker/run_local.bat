@echo off
REM Run training locally with Docker Compose

REM Load environment variables from .env file
if exist .env (
    echo Loading environment variables from .env file...
    for /f "usebackq tokens=1,2 delims==" %%a in (".env") do (
        if not "%%a"=="" if not "%%a:~0,1%"=="#" (
            set "%%a=%%b"
        )
    )
    echo ✓ Environment variables loaded from .env
) else (
    echo Warning: .env file not found
)

REM Default experiment
set EXPERIMENT=%1
if "%EXPERIMENT%"=="" set EXPERIMENT=surge/base

echo Running training with experiment: %EXPERIMENT%
echo Using Docker Compose...

REM Run the training service
docker-compose run --rm train-service python src/train.py experiment=%EXPERIMENT% paths=docker

if %ERRORLEVEL% EQU 0 (
    echo Training completed!
) else (
    echo Training failed!
    exit /b 1
)
