@echo off
setlocal
set /a "count=0"

:loop
if %count%==100 goto end

python -u "c:\RBE470x-projects\team02\project2\variant1.py"
set /a "count+=1"
goto loop

:end
endlocal
echo Finished running the script 100 times.