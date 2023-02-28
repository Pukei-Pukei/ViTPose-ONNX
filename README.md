# ViTPose-ONNX
Easy inference for [ViTPose](https://github.com/ViTAE-Transformer/ViTPose) using ONNX  
<p align="center">
<img src="https://user-images.githubusercontent.com/105025612/221503731-ba87c70b-2422-4e53-a101-ad1bdd9bd3d4.gif">
</p>

## Requirements
```
pip install -r requirements.txt
```
As you can see in 'requirements.txt', it requires only 5 libraries below  
 - matplotlib  
 - numpy  
 - onnxruntime-gpu  
 - opencv-python  
 - yacs  

## Usage
### Install
```
git clone https://github.com/Pukei-Pukei/ViTPose-ONNX.git
cd ViTPose-ONNX
pip install -r requirements.txt
```
### Run
Download [vitpose-b-multi-coco.onnx](https://drive.google.com/drive/folders/1v7tStPJqV4x9vgEW9l_mnwbEuw87exiq?usp=share_link) and [yolov6m.onnx](https://drive.google.com/file/d/1lZ251Y_oG0yNwgFW067HWKsSQAbiLdln/view?usp=share_link), then put them in ViTPose-ONNX folder  
Run the commands below to start inference
```
python run.py -img <path_to_image>
```
```
python run.py -vid <path_to_video>
```
```
python run.py -wc <webcam ID or URL>
```
```
python run.py -cfg <config path> -vid <path_to_video>
```  
### Example
```
python run.py -cfg configs/custom_config.py -vid dance.mp4 -s
```
'-s' for save option

## Options

    --yolov6-path, -yolo PATH       :   Path to YOLOv6 onnx file
    --vitpose-path, -pose PATH      :   Path to ViTPose onnx file

    --image-path, -img PATH         :   Image path 
    --video-path, -vid PATH         :   Videos path 
    --webcam, -wc PATH              :   Webcam id or webcam URL 

    --no-background, -nobg          :   Background will be black screen
    --no-bbox, -nobx                :   Don't draw bboxes
    --no-skeleton, -nosk            :   Don't draw skeletons
    --dynamic-drawing, -dd          :   Turn on dynamic drawing, keypoint 
                                        radius and skeleton width change 
                                        dynamically with bbox size
    --result-scale, -rs SIZE        :   Set a coefficient to scale a size 
                                        of result, set None for not 
                                        processing

    --save, -s                      :   Save drawing result
    --save-prediction, -sp          :   Save the predictions(bbox, pose), 
                                        Numpy is needed to read the save 
                                        file

    --conf-thres, -conf THRES       :   Set confidence threshold for 
                                        non-maximum suppression
    --iou-thres, -iou THRES         :   Set IoU threshold for 
                                        non-maximum suppression
    --max-detection, -max MAX       :   Set max detection for non-maximum 
                                        suppression
    --key-conf-thres, -kconf THRES  :   Set keypoint confidence threshold
    --no-pad                        :   Don't use additional padding

    --cpu, -cpu                     :   Use cpu instead of gpu
    --pose-batch-size, -pbs SIZE    :   Set pose batch size
    --yolo-batch-size, -ybs SIZE    :   Set yolo batch size, 
                                        it works only in video

    --config, -cfg                  :   Config path. use config for easy 
                                        usage of options. default config 
                                        path is 'configs/base_config.py'



## Download ONNX file

|Model   |ONNX       |Original Weight for PyTorch|
|:------:|:---------:|:-------------:|
|[ViTPose-B](https://github.com/ViTAE-Transformer/ViTPose#results-from-this-repo-on-ms-coco-val-set-single-task-training)|[GoogleDrive](https://drive.google.com/drive/folders/1v7tStPJqV4x9vgEW9l_mnwbEuw87exiq?usp=share_link)|[Onedrive](https://1drv.ms/u/s!AimBgYV7JjTlgSrlMB093JzJtqq-?e=Jr5S3R)|
|[YOLOv6-M](https://github.com/meituan/YOLOv6#benchmark)|[GoogleDrive](https://drive.google.com/file/d/1lZ251Y_oG0yNwgFW067HWKsSQAbiLdln/view?usp=share_link)|[Download](https://github.com/meituan/YOLOv6/releases/download/0.3.0/yolov6m.pt)|

If you want other versions, refer to [Tutorial](https://pytorch.org/tutorials/advanced/super_resolution_with_onnxruntime.html) and get your own ONNX


## Acknowledgements

- [ViTPose](https://github.com/ViTAE-Transformer/ViTPose)

- [YOLOv6](https://github.com/meituan/YOLOv6)

- [simple-HRNet](https://github.com/stefanopini/simple-HRNet)

- [(Faster) Non-Maximum Suppression in Python](https://pyimagesearch.com/2015/02/16/faster-non-maximum-suppression-python/)

