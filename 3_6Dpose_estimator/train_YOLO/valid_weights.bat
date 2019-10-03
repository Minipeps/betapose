@echo off

set objects[1] = ape
set objects[2] = bvise
set objects[3] = bowl
set objects[4] = camera
set objects[5] = can
set objects[6] = cat
set objects[7] = cup
set objects[8] = driller
set objects[9] = duck
set objects[10] = eggbo
set objects[11] = glue
set objects[12] = holepuncher
set objects[13] = iron
set objects[14] = lamp
set objects[15] = phone

if %1% LSS 10 (
    set obj=0%1
) else (
    set obj=%1
)

echo Evaluating training of %objects[1]%...

FOR %%f in (backup_linemod_gt\%obj%\*.weights) DO (
    echo Evaluating %%~nxf
    darknet.exe detector map ../data_linemod_gt/%obj%/%objects[%1%]%.data cfg/yolo-linemod-single.cfg backup_linemod_gt/%obj%/%%~nxf > backup_linemod_gt/%obj%/%%~nf.log
)