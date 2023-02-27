from yacs.config import CfgNode as CN


"""
yolov6_path     (str): path to YOLOv6 onnx file
vitpose_path    (str): path to ViTPose onnx file

image_path      (str): path to image file
video_path      (str): path to video file
webcam          (str or int): webcam URL or ID, set None for not using

no_background   (bool): draw a black screen instead of the original image, if True
no_bbox         (bool): skip drawing bboxes, if True
no_skeleton     (bool): skip drawing skeletons, if True
dynamic_drawing (bool): keypoint radius and skeleton width change dynamic with bbox size, if True
smooth_net      (bool): reduce jitter in keypoints predicted by using SmoothNet. not implemented yet
result_scale    (float): set a coefficient to scale a size of result, set None for not processing

save            (bool): save the result, if True
save_prediction (bool): save the predictions(bbox, pose), if True.
                        Numpy is needed to read the save file
set_fps         (int): set a fps of result to be saved, 
                       set None to use original fps for video( or 60fps for webcam)

conf_thres      (float): set a bbox confidence threshold for non-maximum suppression
iou_thres       (float): set a bbox iou threshold for non-maximum suppression
max_detection   (int): set the maximum amount of bbox
key_conf_thres  (float): set a keypoint confidence threshold
no_pad          (bool): do not use additional padding. if True
cpu             (bool): use CPU to inference, if True
pose batch size (int): set pose batch size
yolo batch size (int): set yolo batch size, it works only in video
"""


_C = CN()

_C.yolov6_path = 'yolov6m.onnx'
_C.vitpose_path = 'vitpose-b-multi-coco.onnx'

_C.image_path = ''
_C.video_path = ''
_C.webcam = None

_C.no_background = False
_C.no_bbox = True
_C.no_skeleton = False
_C.dynamic_drawing = True
_C.smooth_net = False
_C.result_scale = None

_C.save = True
_C.save_prediction = False
_C.set_fps = None

_C.conf_thres = 0.25
_C.iou_thres = 0.45
_C.max_detection = 100
_C.key_conf_thres = 0.15
_C.no_pad = False
_C.cpu = False
_C.pose_batch_size = 1
_C.yolo_batch_size = 1


cfg = _C