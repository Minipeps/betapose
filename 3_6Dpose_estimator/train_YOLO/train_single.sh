#CUDA_VISIBLE_DEVICES=3 ./darknet detector train data_linemod_gt/02/bvise.data cfg/yolo-linemod-single.cfg backup_linemod_no_neg/02/yolo-linemod-single_5000.weights
CUDA_VISIBLE_DEVICES=0 ./darknet detector train data_linemod_gt/02/bvise.data cfg/yolo-linemod-single.cfg backup/darknet53.conv.74
