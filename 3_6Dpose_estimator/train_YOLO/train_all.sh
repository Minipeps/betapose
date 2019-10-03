CUDA_VISIBLE_DEVICES=0 ./darknet detector train data_linemod_gt/04/camera.data cfg/yolo-linemod-single.cfg backup/darknet53.conv.74
sh valid_yolo.sh 04 camera
CUDA_VISIBLE_DEVICES=0 ./darknet detector train data_linemod_gt/05/can.data cfg/yolo-linemod-single.cfg backup/darknet53.conv.74
sh valid_yolo.sh 05 can
CUDA_VISIBLE_DEVICES=0 ./darknet detector train data_linemod_gt/06/cat.data cfg/yolo-linemod-single.cfg backup/darknet53.conv.74
sh valid_yolo.sh 06 cat
CUDA_VISIBLE_DEVICES=0 ./darknet detector train data_linemod_gt/08/driller.data cfg/yolo-linemod-single.cfg backup/darknet53.conv.74
sh valid_yolo.sh 08 driller
CUDA_VISIBLE_DEVICES=0 ./darknet detector train data_linemod_gt/09/duck.data cfg/yolo-linemod-single.cfg backup/darknet53.conv.74
sh valid_yolo.sh 09 duck
CUDA_VISIBLE_DEVICES=0 ./darknet detector train data_linemod_gt/10/eggbo.data cfg/yolo-linemod-single.cfg backup/darknet53.conv.74
sh valid_yolo.sh 10 eggbo
CUDA_VISIBLE_DEVICES=0 ./darknet detector train data_linemod_gt/11/glue.data cfg/yolo-linemod-single.cfg backup/darknet53.conv.74
sh valid_yolo.sh 11 glue
CUDA_VISIBLE_DEVICES=0 ./darknet detector train data_linemod_gt/12/holepuncher.data cfg/yolo-linemod-single.cfg backup/darknet53.conv.74
sh valid_yolo.sh 12 holepuncher
CUDA_VISIBLE_DEVICES=0 ./darknet detector train data_linemod_gt/13/iron.data cfg/yolo-linemod-single.cfg backup/darknet53.conv.74
sh valid_yolo.sh 13 iron
CUDA_VISIBLE_DEVICES=0 ./darknet detector train data_linemod_gt/14/lamp.data cfg/yolo-linemod-single.cfg backup/darknet53.conv.74
sh valid_yolo.sh 14 lamp
CUDA_VISIBLE_DEVICES=0 ./darknet detector train data_linemod_gt/15/phone.data cfg/yolo-linemod-single.cfg backup/darknet53.conv.74
sh valid_yolo.sh 15 phone

