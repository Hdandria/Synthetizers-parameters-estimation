@echo off
REM Build Docker image for synthesizer parameter estimation

REM Load environment variables from .env file (for reference)
if exist .env (
    echo Environment file .env found
) else (
    echo Warning: .env file not found
)

echo Building Docker image for synthesizer parameter estimation...

REM Build the image
docker build -t synth-param-estimation:latest .

if %ERRORLEVEL% EQU 0 (
    echo Build completed successfully!
    echo Image: synth-param-estimation:latest
    echo.
    echo Image size:
    docker images synth-param-estimation:latest --format "table {{.Repository}}\t{{.Tag}}\t{{.Size}}"
) else (
    echo Build failed!
    exit /b 1
)
