@echo off
echo ====================================================
echo  ForensicLens v2.1 - Docker Startup
echo ====================================================
echo.
echo Checking Docker Desktop is running...
docker info >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo.
    echo [ERROR] Docker Desktop is not running!
    echo Please start Docker Desktop and wait for it to be ready.
    echo Then run this script again.
    pause
    exit /b 1
)
echo [OK] Docker is running.
echo.
echo Building and starting ForensicLens...
echo (First build takes ~5-10 minutes to download dependencies)
echo.
docker compose up --build -d
if %ERRORLEVEL% neq 0 (
    echo.
    echo [ERROR] Build failed! Check the output above for errors.
    pause
    exit /b 1
)
echo.
echo ====================================================
echo  ForensicLens is starting up!
echo ====================================================
echo.
echo  The model is loading (~30-60 seconds)...
echo  Once ready, open your browser to:
echo.
echo     http://localhost:8000
echo.
echo  To view logs:     docker compose logs -f
echo  To stop:          docker compose down
echo ====================================================
echo.

REM Wait for health check
echo Waiting for server to be ready...
set /a retries=0
:waitloop
if %retries% geq 24 (
    echo.
    echo [WARN] Server is still starting. Check: docker compose logs
    goto :done
)
timeout /t 5 /nobreak >nul
curl -s -o nul -w "%%{http_code}" http://localhost:8000/health 2>nul | findstr "200" >nul
if %ERRORLEVEL% equ 0 (
    echo.
    echo [OK] Server is ready! Opening browser...
    start http://localhost:8000
    goto :done
)
set /a retries=%retries%+1
echo   Still loading... (%retries%/24)
goto :waitloop

:done
echo.
pause
