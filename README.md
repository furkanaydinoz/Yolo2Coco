# Yolo2Coco



## About

This repo is for converting Yolov8 PyTorch Txt format data to Coco format. The repo was created because open source labeling tools such as Label-Studio (https://github.com/HumanSignal/label-studio) or Labelme do not support directly importing the corresponding Yolo format. 

Once the relevant output has been created, <label-studio-converter> (https://github.com/HumanSignal/label-studio-converter) can be used for Label-Studio. In this way, the output of the model can be imported into Label-Studio and a kind of automatic labeling process can be performed. 

The system, which is currently developed for Yolov8 (https://github.com/ultralytics/ultralytics) segmentation datasets, will also be developed for Yolo object detection.

For any questions, please contact us at furkanaydinoz01@gmail.com.


## Usage

* python yolo2coco.py --model_path <path/to/model> --image_path <path/to/image/folder> --json_save_path  <path/to/json> --classes_file_path  <path/to/classes/file> --convert_type from-model or from-txt --labels_path  <path/to/label/folder> 
