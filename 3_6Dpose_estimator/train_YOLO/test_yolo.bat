:: Test with training cfg
darknet.exe detector test ../data_linemod_gt/04/camera.data cfg/yolo-linemod-single.cfg ../models/yolo/04.weights -thresh 0.25
:: Test with yolov3 cfg
REM darknet.exe detector test ../data_linemod_gt/01/ape.data ../yolo/cfg/yolov3-single.cfg ../models/yolo/01.weights -thresh 0.25
