@echo off
REM Enhanced RAG Pipeline - Docker Deployment Script
REM PanScience Innovations LLM Specialist Assignment

echo.
echo ========================================
echo   Enhanced RAG Pipeline Deployment
echo   PanScience Innovations LLM Assignment
echo ========================================
echo.

REM Check if Docker is running
echo Checking Docker status...
docker info >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: Docker is not running or not installed!
    echo Please start Docker Desktop and try again.
    pause
    exit /b 1
)

REM Check if .env file exists
if not exist ".env" (
    echo Creating .env file...
    echo GOOGLE_API_KEY=your_gemini_api_key_here > .env
    echo.
    echo WARNING: Please edit .env file with your actual Google API key!
    echo.
    pause
)

REM Build and start the application
echo Building Docker image...
docker-compose build

if %errorlevel% neq 0 (
    echo ERROR: Docker build failed!
    pause
    exit /b 1
)

echo.
echo Starting RAG Pipeline...
docker-compose up -d

if %errorlevel% neq 0 (
    echo ERROR: Failed to start containers!
    pause
    exit /b 1
)

echo.
echo ========================================
echo   Deployment Successful!
echo ========================================
echo.
echo Application is running at: http://localhost:8000
echo Web Interface: http://localhost:8000/static/index.html
echo.
echo Useful commands:
echo   View logs: docker-compose logs -f
echo   Stop: docker-compose down
echo   Restart: docker-compose restart
echo.
pause
