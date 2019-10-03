@echo off

for /l %%i in (7,1,14) do (
    if [[%%i NEQ 7]] python src/train.py --trainBatch 8 --expID seq%%i_batchSize_8 --optMethod adam --nThreads 12 >> seq%%i_batchSize_8.log
)