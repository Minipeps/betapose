@echo off

FOR /L %%n IN (1,1,15) DO (
    echo Annotating keypoints for obj %%n...
    python annotate_keypoint.py --obj_id %%n --total_kp_number 50 --output_base ../3_6Dpose_estimator/data/ --sixd_base ../LineMod
)