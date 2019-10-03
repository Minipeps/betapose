:: Train on LineMod dataset without DPG
REM python src/train.py --expID seq1_batchSize_4 --nClasses 50 --optMethod adam --trainBatch 4 --validBatch 4 --nThreads 12
:: Train on LineMod dataset with DPG
python src/train.py --expID seq1_dpg_batchSize_4 --nClasses 50 --optMethod adam --trainBatch 4 --validBatch 4 --nThreads 12 --loadModel ../exp/coco/seq1_batchSize_4/model_490.pkl --addDPG
