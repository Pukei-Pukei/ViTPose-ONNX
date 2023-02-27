import sys
import argparse

from yacs.config import CfgNode as CN


def get_config():
    parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS)
    parser.add_argument('--yolov6_path', '-yolo', help='yolov6 path')
    parser.add_argument('--vitpose_path', '-pose', help='vitpose path')

    parser.add_argument('--image-path', '-img', help='image path')
    parser.add_argument('--video-path', '-vid', help='videos path')
    parser.add_argument('--webcam', '-wc', help='webcam id or webcam URL')

    parser.add_argument('--no-background', '-nobg', action='store_true', help="draw only skeletons or bboxes, background will be black")
    parser.add_argument('--no-bbox', '-nobx', action='store_true', help="dont't draw bboxes")
    parser.add_argument('--no-skeleton', '-nosk', action='store_true', help="don't draw skeletons")
    parser.add_argument('--dynamic-drawing', '-dd', action='store_true', help='turn on dynamic drawing')
    parser.add_argument('--smooth-net', '-sn', action='store_true', help='use smooth-net for jitter filtering')
    parser.add_argument('--result-scale', '-rs', type=float, help='set scale to result')

    parser.add_argument('--save', '-s', action='store_true', help='save drawing result')
    parser.add_argument('--save-prediction', '-sp', action='store_true', help='save prediction')
    parser.add_argument('--set-fps', '-fps', type=int, help='set fps for result video')

    parser.add_argument('--conf-thres', '-conf',type=float, help='set conf thres for nms')
    parser.add_argument('--iou-thres', '-iou', type=float, help='set iou thres for nms')
    parser.add_argument('--max-detection', '-max',type=int, help='set max detection for nms')
    parser.add_argument('--key-conf-thres', '-kconf',type=float, help='set keypoint conf thres')
    parser.add_argument('--no-pad', action='store_true', help="don't use additional padding")
    parser.add_argument('--cpu', '-cpu', action='store_true', help="use cpu instead of gpu")
    parser.add_argument('--pose-batch-size', '-pbs',type=int, help='set pose batch size')
    parser.add_argument('--yolo-batch-size', '-ybs',type=int, help='set yolo batch size')

    parser.add_argument('--config', '-cfg', default=None, help='config path')

    args = parser.parse_args()
    args = vars(args)

    if args['config'] is None:
        cfg = cfg = CN._load_cfg_py_source('configs/base_config.py')
    else:
        cfg = CN._load_cfg_py_source(args.config)

    for key, value in args.items():
        cfg[key] = value

    return cfg


def print_fps(fps):
    bar = int(fps*2) // 5
    sub_bar = " ▎▍▋▊"[int(fps*2)%5] # ▉▊▋▍▎
    sys.stdout.write("\033[K")
    print(f'- fps:{fps:06.1f} : ' + "▉"*bar + sub_bar, end='\r')