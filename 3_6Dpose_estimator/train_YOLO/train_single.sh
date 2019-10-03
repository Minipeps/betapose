#CUDA_VISIBLE_DEVICES=3 ./darknet detector train data_linemod_gt/02/bvise.data cfg/yolo-linemod-single.cfg backup_linemod_no_neg/02/yolo-linemod-single_5000.weights

./darknet_cudnn0 detector train ../data_linemod_gt/05/can.data cfg/yolo-linemod-single.cfg darknet53.conv.74 -map