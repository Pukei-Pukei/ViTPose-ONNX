import numpy as np
import cv2

from utils.nms import nms


def letterbox(im, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleup=True, stride=32, return_int=False):
    '''Resize and pad image while meeting stride-multiple constraints.'''
    shape = im.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)
    elif isinstance(new_shape, list) and len(new_shape) == 1:
       new_shape = (new_shape[0], new_shape[0])

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better val mAP)
        r = min(r, 1.0)

    # Compute padding
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding

    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding

    dw /= 2  # divide padding into 2 sides
    dh /= 2
    
    if shape[::-1] != new_unpad:  # resize
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    if not return_int:
        return im, r, (dw, dh)
    else:
        return im, r, (left, top)


def plot_box_and_label(image, lw, box, label='', color=(128, 128, 128), txt_color=(255, 255, 255), font=cv2.FONT_HERSHEY_COMPLEX):
    # Add one xyxy box to image with label
    p1, p2 = (int(box[0]), int(box[1])), (int(box[2]), int(box[3]))
    cv2.rectangle(image, p1, p2, color, thickness=lw, lineType=cv2.LINE_AA)
    if label:
        lw = 1
        tf = max(lw - 1, 1)  # font thickness
        w, h = cv2.getTextSize(label, 0, fontScale=lw / 3, thickness=tf)[0]  # text width, height
        outside = p1[1] - h - 3 >= 0  # label fits outside box
        p2 = p1[0] + w, p1[1] - h - 3 if outside else p1[1] + h + 3
        cv2.rectangle(image, p1, p2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(image, label, (p1[0], p1[1] - 2 if outside else p1[1] + h + 2), font, lw / 3, txt_color,
                    thickness=tf, lineType=cv2.LINE_AA)


def xywh2xyxy(x):
    '''Convert boxes with shape [n, 4] from [x, y, w, h] to [x1, y1, x2, y2] where x1y1 is top-left, x2y2=bottom-right.'''
    y = np.copy(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
    return y


def xyxy2xywh(x):
    '''Convert nx4 boxes from [x1, y1, x2, y2] to [x, y, w, h] where xy1=top-left, xy2=bottom-right.'''
    y = np.copy(x)
    y[..., 0] = (x[..., 0] + x[..., 2]) / 2  # x center
    y[..., 1] = (x[..., 1] + x[..., 3]) / 2  # y center
    y[..., 2] = x[..., 2] - x[..., 0]  # width
    y[..., 3] = x[..., 3] - x[..., 1]  # height
    return y


def non_max_suppression(prediction, conf_thres=0.25, iou_thres=0.45):
    """Runs Non-Maximum Suppression (NMS) on inference results.
    This code is borrowed from: https://github.com/ultralytics/yolov5/blob/47233e1698b89fc437a4fb9463c815e9171be955/utils/general.py#L775
    Args:
        prediction: (numpy.ndarray), with shape [N, 5 + num_classes], N is the number of bboxes.
        conf_thres: (float) confidence threshold.
        iou_thres: (float) iou threshold.
    Returns:
         output: (numpy.ndarray), list of detections, each item is a tensor with shape (num_boxes, 6), 6 is for [xyxy, conf, cls].
    """

    pred_candidates = np.logical_and(prediction[..., 4] > conf_thres, np.amax(prediction[..., 5:], axis=-1)[0] > conf_thres)  # candidates
    # Check the parameters.
    assert 0 <= conf_thres <= 1, f'conf_thresh must be in 0.0 to 1.0, however {conf_thres} is provided.'
    assert 0 <= iou_thres <= 1, f'iou_thres must be in 0.0 to 1.0, however {iou_thres} is provided.'

    # Function settings.
    max_wh = 4096  # maximum box width and height
    max_nms = 30000  # maximum number of boxes put into torchvision.ops.nms()

    output = [np.zeros((0, 6))] * prediction.shape[0]
    for img_idx, x in enumerate(prediction):  # image index, image inference
        x = x[pred_candidates[img_idx]]  # confidence

        # If no box remains, skip the next process.
        if not x.shape[0]:
            continue

        # confidence multiply the objectness
        x[:, 5:] *= x[:, 4:5]  # conf = obj_conf * cls_conf

        # (center x, center y, width, height) to (x1, y1, x2, y2)
        box = xywh2xyxy(x[:, :4])

        class_idx = x[:, 5:].argmax(1, keepdims=True)
        conf = np.take_along_axis(x[:, 5:], class_idx, axis=1)
        x = np.concatenate((box, conf, class_idx.astype('float32')), 1)[conf.flatten() > conf_thres]

        # Check shape
        num_box = x.shape[0]  # number of boxes
        if not num_box:  # no boxes kept.
            continue
        elif num_box > max_nms:  # excess max boxes' number.
            # sort by confidence
            x = x[np.flip(x[:, 4].argsort(), -1)[:max_nms]]

        # Batched NMS
        boxes, scores = x[:, :4], x[:, 4]  # boxes (offset by class), scores
        keep_box_idx = nms(boxes, scores, iou_thres)

        output[img_idx] = x[keep_box_idx]

    return output


def rescale(ori_shape, boxes, target_shape):
    '''Rescale the output to the original image shape'''
    ratio = min(ori_shape[0] / target_shape[0], ori_shape[1] / target_shape[1])
    padding = (ori_shape[1] - target_shape[1] * ratio) / 2, (ori_shape[0] - target_shape[0] * ratio) / 2

    boxes[:, [0, 2]] -= padding[0]
    boxes[:, [1, 3]] -= padding[1]
    boxes[:, :4] /= ratio

    boxes[:, 0] = boxes[:, 0].clip(0, target_shape[1])  # x1
    boxes[:, 1] = boxes[:, 1].clip(0, target_shape[0])  # y1
    boxes[:, 2] = boxes[:, 2].clip(0, target_shape[1])  # x2
    boxes[:, 3] = boxes[:, 3].clip(0, target_shape[0])  # y2

    return boxes


def preprocess_with_bboxes(original_img, bboxes, detection_img_size, pose_img_size, cfg):
    """
        Args:
        original_img: (numpy.ndarray) (H, W, C)
        bboxes: (numpy.ndarray), normalized bboxes with shape [N, 5 + num_classes], N is the number of bboxes.
        detection_img_size: (tuple), (H, W)
        pose_img_size: (tuple), (H, W)
    Returns:
         img_list: (list of numpy.ndarray)
         xyxy_list: (list of numpy.ndarray)
         conf_list: (list of numpy.ndarray)
    """

    if len(bboxes):
        bboxes[:, :4] = rescale(detection_img_size, bboxes[:, :4], original_img.shape)

    img_list = []
    xyxy_list = []
    conf_list = []
 
    # bboxes = np.flip(bboxes, axis=0)
    for i, (*xyxy, conf, cls) in enumerate(bboxes):
        if i >= cfg.max_detection:
            break

        # pad to offset the wrong effect in PatchEmbed in vit.py
        # in my opinion, a little bit smaller image is better than a little bit truncated image
        # padding=4 for base conf
        if not cfg.no_pad:
            padding = 4
            xyxy[2] += padding * (xyxy[2]-xyxy[0]) / pose_img_size[1]
            xyxy[3] += padding * (xyxy[3]-xyxy[1]) / pose_img_size[0]
            xyxy[2] = np.clip(xyxy[2], xyxy[0], original_img.shape[1])
            xyxy[3] = np.clip(xyxy[3], xyxy[1], original_img.shape[0])
        
        # crop image
        l, t, r, b = map(int, np.round(xyxy))
        img = original_img[t:b, l:r, :]

        # resize image
        img = cv2.resize(img, pose_img_size[::-1], interpolation = cv2.INTER_LINEAR)

        # normalization
        img = img / 255.0
        mean_std = np.array([[0.485, 0.456, 0.406], [0.229, 0.224, 0.225]])
        img = (img - mean_std[0]) / mean_std[1]

        # convert to torch tensor format
        # img = img.transpose(2, 0, 1).astype('float32') # HWC to CHW

        img_list.append(img)
        xyxy_list.append(xyxy)
        conf_list.append(conf)

    return img_list, xyxy_list, conf_list