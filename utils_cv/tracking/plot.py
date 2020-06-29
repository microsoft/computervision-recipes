# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import os.path as osp
from typing import Dict, List, Tuple
import cv2
import numpy as np

from .bbox import TrackingBbox


def draw_boxes(
    im: np.ndarray,
    cur_tracks: List[TrackingBbox],
    color_map: Dict[int, Tuple[int, int, int]],
) -> np.ndarray:
    """ 
    Overlay bbox and id labels onto the frame

    Args:
        im: raw frame
        cur_tracks: list of bboxes in the current frame
        color_map: dictionary mapping ids to bbox colors
    """

    cur_ids = [bb.track_id for bb in cur_tracks]
    tracks = dict(zip(cur_ids, cur_tracks))
    for label, bb in tracks.items():
        left = round(bb.left)
        top = round(bb.top)
        right = round(bb.right)
        bottom = round(bb.bottom)

        # box text and bar
        color = color_map[label]
        label = str(label)

        # last two args of getTextSize() are font_scale and thickness
        t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 1, 1)[0]
        cv2.rectangle(im, (left, top), (right, bottom), color, 3)
        cv2.putText(
            im,
            "id_" + label,
            (left, top + t_size[1] - 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            color,
            3,
        )

    return im


def assign_colors(id_list: List[int],) -> Dict[int, Tuple[int, int, int]]:
    """ 
    Produce corresponding unique color palettes for unique ids
    
    Args:
        id_list: list of track ids 
    """
    palette = (2 ** 11 - 1, 2 ** 15 - 1, 2 ** 20 - 1)

    color_list = []
    id_list2 = list(range(len(id_list)))

    # adapted from https://github.com/ZQPei/deep_sort_pytorch
    for i in id_list2:
        color = [int((p * ((i + 1) ** 5 - i + 1)) % 255) for p in palette]
        color_list.append(tuple(color))

    color_map = dict(zip(id_list, color_list))

    return color_map
