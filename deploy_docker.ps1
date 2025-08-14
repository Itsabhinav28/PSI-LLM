# Enhanced RAG Pipeline - Docker Deployment Script (PowerShell)
# PanScience Innovations LLM Specialist Assignment

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  Enhanced RAG Pipeline Deployment" -ForegroundColor Cyan
Write-Host "  PanScience Innovations LLM Assignment" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Check if Docker is running
Write-Host "Checking Docker status..." -ForegroundColor Yellow
try {
    docker info | Out-Null
    Write-Host "✅ Docker is running" -ForegroundColor Green
} catch {
    Write-Host "❌ ERROR: Docker is not running or not installed!" -ForegroundColor Red
    Write-Host "Please start Docker Desktop and try again." -ForegroundColor Red
    Read-Host "Press Enter to continue"
    exit 1
}

# Check if .env file exists
if (-not (Test-Path ".env")) {
    Write-Host "Creating .env file..." -ForegroundColor Yellow
    "GOOGLE_API_KEY=your_gemini_api_key_here" | Out-File -FilePath ".env" -Encoding UTF8
    Write-Host ""
    Write-Host "⚠️  WARNING: Please edit .env file with your actual Google API key!" -ForegroundColor Yellow
    Write-Host ""
    Read-Host "Press Enter to continue"
}

# Build and start the application
Write-Host "Building Docker image..." -ForegroundColor Yellow
try {
    docker-compose build
    Write-Host "✅ Docker build successful" -ForegroundColor Green
} catch {
    Write-Host "❌ ERROR: Docker build failed!" -ForegroundColor Red
    Read-Host "Press Enter to continue"
    exit 1
}

Write-Host ""
Write-Host "Starting RAG Pipeline..." -ForegroundColor Yellow
try {
    docker-compose up -d
    Write-Host "✅ RAG Pipeline started successfully" -ForegroundColor Green
} catch {
    Write-Host "❌ ERROR: Failed to start containers!" -ForegroundColor Red
    Read-Host "Press Enter to continue"
    exit 1
}

Write-Host ""
Write-Host "========================================" -ForegroundColor Green
Write-Host "   Deployment Successful!" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Green
Write-Host ""
Write-Host "🌐 Application is running at: http://localhost:8000" -ForegroundColor Green
Write-Host "💻 Web Interface: http://localhost:8000/static/index.html" -ForegroundColor Green
Write-Host ""
Write-Host "📋 Useful commands:" -ForegroundColor Cyan
Write-Host "   View logs: docker-compose logs -f" -ForegroundColor White
Write-Host "   Stop: docker-compose down" -ForegroundColor White
Write-Host "   Restart: docker-compose restart" -ForegroundColor White
Write-Host "   Status: docker-compose ps" -ForegroundColor White
Write-Host ""
Write-Host "🚀 Your Enhanced RAG Pipeline is now running!" -ForegroundColor Green
Write-Host ""

Read-Host "Press Enter to continue"
