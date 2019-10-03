# Changelog

* Updated code to run on Python 3.7, PyTorch 1.0.1, CUDA 10 and CUDNN 7.5 (_tested on Windows_)

* (TODO) Added missing keypoints files for class 3 and 7

* Added batch scripts (for automating tasks on Windows):
  * `annotate_all.bat`: automate annotation tasks (II) for each LineMod class
  * `train_single.bat`: launch training on darknet for the designated class (modify batch file to change class)

* Modified `yolo-linemod-single.cfg`:
  * Set batch_size to 32 (instead of 64) because of hardware limitation
  * Limited max_batches to 10000 (instead of 500200) for easier training time management

* **(TODO)** Added pre-trained weights for each class for quick evaluation

* **(TODO)** Updated `README.md`
