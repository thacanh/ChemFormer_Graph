@echo off
echo ========================================
echo CHEMFORMER RETROSYNTHESIS DEMO
echo ========================================
echo.

echo Activating chemformer environment...
call conda activate chemformer

echo Chay demo retrosynthesis...
echo.

python demo_retrosynthesis.py

echo.
echo ========================================
echo Demo da hoan tat!
echo ========================================
pause
