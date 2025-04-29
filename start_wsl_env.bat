@echo off
REM Get the current directory
set "CURRENT_DIR=%cd%"

REM Start WSL, move into current folder, activate conda env
wsl bash -c "cd \"$(wslpath '%CURRENT_DIR%')\" && conda activate myproject && exec bash"
