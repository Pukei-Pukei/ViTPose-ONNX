import cv2
import matplotlib.pyplot as plt
import numpy as np


__all__ = ["joints_dict", "draw_points_and_skeleton"]


def joints_dict():
    joints = {
        "coco": {
            "keypoints": {
                0: "nose",
                1: "left_eye",
                2: "right_eye",
                3: "left_ear",
                4: "right_ear",
                5: "left_shoulder",
                6: "right_shoulder",
                7: "left_elbow",
                8: "right_elbow",
                9: "left_wrist",
                10: "right_wrist",
                11: "left_hip",
                12: "right_hip",
                13: "left_knee",
                14: "right_knee",
                15: "left_ankle",
                16: "right_ankle"
            },
            "skeleton": [
                # # [16, 14], [14, 12], [17, 15], [15, 13], [12, 13], [6, 12], [7, 13], [6, 7], [6, 8],
                # # [7, 9], [8, 10], [9, 11], [2, 3], [1, 2], [1, 3], [2, 4], [3, 5], [4, 6], [5, 7]
                # [15, 13], [13, 11], [16, 14], [14, 12], [11, 12], [5, 11], [6, 12], [5, 6], [5, 7],
                # [6, 8], [7, 9], [8, 10], [1, 2], [0, 1], [0, 2], [1, 3], [2, 4], [3, 5], [4, 6]
                [15, 13], [13, 11], [16, 14], [14, 12], [11, 12], [5, 11], [6, 12], [5, 6], [5, 7],
                [6, 8], [7, 9], [8, 10], [1, 2], [0, 1], [0, 2], [1, 3], [2, 4],  # [3, 5], [4, 6]
                [0, 5], [0, 6]
            ]
        },
        "mpii": {
            "keypoints": {
                0: "right_ankle",
                1: "right_knee",
                2: "right_hip",
                3: "left_hip",
                4: "left_knee",
                5: "left_ankle",
                6: "pelvis",
                7: "thorax",
                8: "upper_neck",
                9: "head top",
                10: "right_wrist",
                11: "right_elbow",
                12: "right_shoulder",
                13: "left_shoulder",
                14: "left_elbow",
                15: "left_wrist"
            },
            "skeleton": [
                # [5, 4], [4, 3], [0, 1], [1, 2], [3, 2], [13, 3], [12, 2], [13, 12], [13, 14],
                # [12, 11], [14, 15], [11, 10], # [2, 3], [1, 2], [1, 3], [2, 4], [3, 5], [4, 6], [5, 7]
                [5, 4], [4, 3], [0, 1], [1, 2], [3, 2], [3, 6], [2, 6], [6, 7], [7, 8], [8, 9],
                [13, 7], [12, 7], [13, 14], [12, 11], [14, 15], [11, 10],
            ]
        },
    }
    return joints


def draw_points(image, points, color_palette='tab20', palette_samples=16, confidence_threshold=0.5, xywh=None):
    """
    Draws `points` on `image`.

    Args:
        image: image in opencv format
        points: list of points to be drawn.
            Shape: (nof_points, 3)
            Format: each point should contain (y, x, confidence)
        color_palette: name of a matplotlib color palette
            Default: 'tab20'
        palette_samples: number of different colors sampled from the `color_palette`
            Default: 16
        confidence_threshold: only points with a confidence higher than this threshold will be drawn. Range: [0, 1]
            Default: 0.5

    Returns:
        A new image with overlaid points

    """
    try:
        colors = np.round(
            np.array(plt.get_cmap(color_palette).colors) * 255
        ).astype(np.uint8)[:, ::-1].tolist()
    except AttributeError:  # if palette has not pre-defined colors
        colors = np.round(
            np.array(plt.get_cmap(color_palette)(np.linspace(0, 1, palette_samples))) * 255
        ).astype(np.uint8)[:, -2::-1].tolist()

    if xywh is not None:
        circle_size = 7
        circle_size = int(np.clip(np.sqrt(sum(xywh[2:]) / 3000)*circle_size, 1, circle_size*2))
    else:
        circle_size = max(1, min(image.shape[:2]) // 160)
    

    for i, pt in enumerate(points):
        if pt[2] > confidence_threshold:
            image = cv2.circle(image, (int(pt[1]), int(pt[0])), circle_size, tuple(colors[i % len(colors)]), -1)

    return image


def draw_skeleton(image, points, skeleton, color_palette='Set2', palette_samples=8, person_index=0,
                  confidence_threshold=0.5, xywh=None):
    """
    Draws a `skeleton` on `image`.

    Args:
        image: image in opencv format
        points: list of points to be drawn.
            Shape: (nof_points, 3)
            Format: each point should contain (y, x, confidence)
        skeleton: list of joints to be drawn
            Shape: (nof_joints, 2)
            Format: each joint should contain (point_a, point_b) where `point_a` and `point_b` are an index in `points`
        color_palette: name of a matplotlib color palette
            Default: 'Set2'
        palette_samples: number of different colors sampled from the `color_palette`
            Default: 8
        person_index: index of the person in `image`
            Default: 0
        confidence_threshold: only points with a confidence higher than this threshold will be drawn. Range: [0, 1]
            Default: 0.5

    Returns:
        A new image with overlaid joints

    """
    try:
        colors = np.round(
            np.array(plt.get_cmap(color_palette).colors) * 255
        ).astype(np.uint8)[:, ::-1].tolist()
    except AttributeError:  # if palette has not pre-defined colors
        colors = np.round(
            np.array(plt.get_cmap(color_palette)(np.linspace(0, 1, palette_samples))) * 255
        ).astype(np.uint8)[:, -2::-1].tolist()

    if xywh is not None:
        lw = 2
        lw = int(np.clip((sum(xywh[2:]) / 300)**(1/6)*lw, 1, lw*2))
    else:
        lw = 2

    for i, joint in enumerate(skeleton):
        pt1, pt2 = points[joint]
        if pt1[2] > confidence_threshold and pt2[2] > confidence_threshold:
            image = cv2.line(
                image, (int(pt1[1]), int(pt1[0])), (int(pt2[1]), int(pt2[0])),
                tuple(colors[person_index % len(colors)]), lw
            )

    return image


def draw_points_and_skeleton(image, points, skeleton, points_color_palette='tab20', points_palette_samples=16,
                             skeleton_color_palette='Set2', skeleton_palette_samples=8, person_index=0,
                             confidence_threshold=0.5, xywh=None):
    """
    Draws `points` and `skeleton` on `image`.

    Args:
        image: image in opencv format
        points: list of points to be drawn.
            Shape: (nof_points, 3)
            Format: each point should contain (y, x, confidence)
        skeleton: list of joints to be drawn
            Shape: (nof_joints, 2)
            Format: each joint should contain (point_a, point_b) where `point_a` and `point_b` are an index in `points`
        points_color_palette: name of a matplotlib color palette
            Default: 'tab20'
        points_palette_samples: number of different colors sampled from the `color_palette`
            Default: 16
        skeleton_color_palette: name of a matplotlib color palette
            Default: 'Set2'
        skeleton_palette_samples: number of different colors sampled from the `color_palette`
            Default: 8
        person_index: index of the person in `image`
            Default: 0
        confidence_threshold: only points with a confidence higher than this threshold will be drawn. Range: [0, 1]
            Default: 0.5

    Returns:
        A new image with overlaid joints

    """
    image = draw_skeleton(image, points, skeleton, color_palette=skeleton_color_palette,
                          palette_samples=skeleton_palette_samples, person_index=person_index,
                          confidence_threshold=confidence_threshold, xywh=xywh)
    image = draw_points(image, points, color_palette=points_color_palette, palette_samples=points_palette_samples,
                        confidence_threshold=confidence_threshold, xywh=xywh)
    return image