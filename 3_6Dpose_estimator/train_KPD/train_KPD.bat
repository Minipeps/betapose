:: 1. Without DPG
python src/train.py --trainBatch 8 --expID seq2_batchSize_8 --optMethod adam --nThreads 12
:: 2. With DPG
REM python src/train.py --trainBatch 8 --expID seq2_dpg_batchSize_8 --optMethod adam --loadModel ../exp/coco/seq2_batchSize_8/model_485.pkl --nThreads 12 --addDPG
