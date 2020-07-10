# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

from collections import OrderedDict
from typing import Dict, List, Tuple

import cv2
import decord
import io
import IPython.display
import numpy as np
from PIL import Image
from time import sleep

from .bbox import TrackingBbox
from .model import _get_frame


def plot_single_frame(
    input_video: str,
    frame_id: int,
    results: Dict[int, List[TrackingBbox]] = None
) -> None:
    """
    Plot the bounding box and id on a wanted frame. Display as image to front end.

    Args:
        input_video: path to the input video
        frame_id: frame_id for frame to show tracking result
        results: dictionary mapping frame id to a list of predicted TrackingBboxes
    """

    # Extract frame
    im = _get_frame(input_video, frame_id)

    # Overlay results
    if results:
        results = OrderedDict(sorted(results.items()))

        # Assign bbox color per id
        unique_ids = list(
            set([bb.track_id for frame in results.values() for bb in frame])
        )
        color_map = assign_colors(unique_ids)

        # Extract tracking results for wanted frame, and draw bboxes+tracking id, display frame
        cur_tracks = results[frame_id]

        if len(cur_tracks) > 0:
            im = draw_boxes(im, cur_tracks, color_map)

    # Display image
    im = Image.fromarray(im)
    IPython.display.display(im)


def play_video(
    results: Dict[int, List[TrackingBbox]], input_video: str
) -> None:
    """
     Plot the predicted tracks on the input video. Displays to front-end as sequence of images stringed together in a video.

    Args:
        results: dictionary mapping frame id to a list of predicted TrackingBboxes
        input_video: path to the input video
    """

    results = OrderedDict(sorted(results.items()))

    # assign bbox color per id
    unique_ids = list(
        set([bb.track_id for frame in results.values() for bb in frame])
    )
    color_map = assign_colors(unique_ids)

    # read video and initialize new tracking video
    video_reader = decord.VideoReader(input_video)

    # set up ipython jupyter display
    d_video = IPython.display.display("", display_id=1)

    # Read each frame, add bbox+track id, display frame
    for frame_idx in range(len(results) - 1):
        cur_tracks = results[frame_idx]
        im = video_reader.next().asnumpy()

        if len(cur_tracks) > 0:
            cur_image = draw_boxes(im, cur_tracks, color_map)

        f = io.BytesIO()
        im = Image.fromarray(im)
        im.save(f, "jpeg")
        d_video.update(IPython.display.Image(data=f.getvalue()))
        sleep(0.000001)


def write_video(
    results: Dict[int, List[TrackingBbox]], input_video: str, output_video: str
) -> None:
    """
    Plot the predicted tracks on the input video. Write the output to {output_path}.

    Args:
        results: dictionary mapping frame id to a list of predicted TrackingBboxes
        input_video: path to the input video
        output_video: path to write out the output video
    """
    results = OrderedDict(sorted(results.items()))
    # read video and initialize new tracking video
    video = cv2.VideoCapture()
    video.open(input_video)

    im_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    im_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*"MP4V")
    frame_rate = int(video.get(cv2.CAP_PROP_FPS))
    writer = cv2.VideoWriter(
        output_video, fourcc, frame_rate, (im_width, im_height)
    )

    # assign bbox color per id
    unique_ids = list(
        set([bb.track_id for frame in results.values() for bb in frame])
    )
    color_map = assign_colors(unique_ids)

    # create images and add to video writer, adapted from https://github.com/ZQPei/deep_sort_pytorch
    frame_idx = 0
    while video.grab():
        _, im = video.retrieve()
        cur_tracks = results[frame_idx]
        if len(cur_tracks) > 0:
            im = draw_boxes(im, cur_tracks, color_map)
        writer.write(im)
        frame_idx += 1

    print(f"Output saved to {output_video}.")


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
        color = [int((p * ((i + 1) ** 4 - i + 1)) % 255) for p in palette]
        color_list.append(tuple(color))

    color_map = dict(zip(id_list, color_list))

    return color_map
