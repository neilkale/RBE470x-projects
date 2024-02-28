@echo off
setlocal
set /a "count=0"

:loop
if %count%==20 goto end

python -u "c:\RBE470x-projects\team02\project2\variant4.py"
set /a "count+=1"
goto loop

:end
endlocal
echo Finished running the script 20 times.