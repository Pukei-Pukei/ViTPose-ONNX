import os

import onnxruntime as ort

from utils.inference import inference_image, inference_video, inference_webcam
from utils import get_config


def main(cfg):
    YOLOV6_PATH = cfg.yolov6_path
    VITPOSE_PATH = cfg.vitpose_path
    IMG_PATH = cfg.image_path
    VID_PATH = cfg.video_path
    WEBCAM = cfg.webcam

    assert (IMG_PATH or VID_PATH or (WEBCAM is not None)), "Argument -img or -vid or -wc should be provided"

    if cfg.cpu:
        EP_list = ['CPUExecutionProvider']
    else:
        EP_list = ['CUDAExecutionProvider', 'CPUExecutionProvider']

    yolov6_sess = ort.InferenceSession(YOLOV6_PATH, providers=EP_list)
    vitpose_sess = ort.InferenceSession(VITPOSE_PATH, providers=EP_list)
    # TODO : implement smooth_net feature
    # if cfg.smooth_net:
    #     smooth_net = ort.InferenceSession('smoothnet-32.onnx', providers=EP_list)

    os.system("") # make terminal to be able to use ANSI escape

    # Inference image
    if IMG_PATH:
        inference_image(IMG_PATH, yolov6_sess, vitpose_sess, cfg)

    #Inference video from file
    if VID_PATH:
        inference_video(VID_PATH, yolov6_sess, vitpose_sess, cfg)

    # Inference video from webcam
    if WEBCAM is not None:
        inference_webcam(WEBCAM, yolov6_sess, vitpose_sess, cfg)


if __name__ == "__main__":
    cfg = get_config()

    main(cfg)