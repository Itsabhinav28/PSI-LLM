@echo off
echo Starting Enhanced RAG Pipeline Servers...
echo.

echo Starting Simple HTTP Server on Port 8080...
start "HTTP Server" cmd /k "cd src\api\static && python -m http.server 8080"

echo Starting FastAPI Server on Port 8000...
start "FastAPI Server" cmd /k "venv\Scripts\activate && python -m uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload"

echo.
echo Servers are starting...
echo.
echo 1. Web Interface: http://localhost:8080
echo 2. API Backend: http://localhost:8000
echo 3. API Docs: http://localhost:8000/docs
echo.
echo Press any key to close this window...
pause > nul
