obj=$1
name=$2

echo Evaluating training of ${obj}...

for f in backup_linemod_gt/${obj}/*.weights; do
    echo Evaluating ${f##*/}
    CUDA_VISIBLE_DEVICES=0 ./darknet detector map ../data_linemod_gt/${obj}/$2.data cfg/yolo-linemod-single.cfg ${f} > ${f%.weights}.log
done
