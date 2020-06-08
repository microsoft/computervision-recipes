from collections import OrderedDict
import os.path as osp
from typing import Dict, List, Tuple
import cv2
import numpy as np
import pandas as pd

from .bbox import TrackingBbox


def plot_results(
    results: Dict[int, List[TrackingBbox]],
    video_path: str,
    frame_rate: int = 30,
) -> str:
    """ 
    Plot the predicted tracks on the input video. Write the output to track_{input_name}.mp4.
    
    Args:
        results: dictionary mapping frame id to a list of predicted TrackingBboxes
        video_path: path to the input video
        frame_rate: frame rate

        Returns (str): 
            path at which the txt file containing the bboxes and ids in MOT format has been saved.    
    """
    # convert results to dataframe in MOT challenge format
    preds = OrderedDict(sorted(results.items()))
    bboxes = [
        [
            bb.frame_id,
            bb.track_id,
            bb.left,
            bb.top,
            bb.right - bb.left,
            bb.bottom - bb.top,
        ]
        for _, v in preds.items()
        for bb in v
    ]

    df = pd.DataFrame(
        bboxes, columns=["frame", "id", "left", "top", "width", "height",],
    )

    data_dir, video_name = osp.split(video_path)
    output_path = osp.join(data_dir, f"track_{video_name}")
    _write_video(df, video_path, output_path, frame_rate)
    print(f"Output saved to {output_path}.")

    return output_path


def _write_video(
    df: pd.DataFrame, input_video: str, output_video: str, frame_rate: int
) -> None:
    """     
    Args:
        df: DataFrame with columns [frame, id, left, top, width, height]
        input_video: path to the input video
        output_video: path to write out the output video
        frame_rate: frame rate
    """

    # read video and initialize new tracking video
    video = cv2.VideoCapture()
    video.open(input_video)

    image_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    image_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*"MP4V")
    writer = cv2.VideoWriter(
        output_video, fourcc, frame_rate, (image_width, image_height)
    )

    # assign bbox color per id:
    id_color_dict = _compute_color_for_all_labels(df.id.unique().tolist())

    # create images and add to video writer, adapted from https://github.com/ZQPei/deep_sort_pytorch
    frame_idx = 1
    while video.grab():
        _, cur_image = video.retrieve()
        cur_frame = df[df.frame == frame_idx]
        if not cur_frame.empty:
            cur_image = _draw_boxes(cur_image, cur_frame, id_color_dict)
        writer.write(cur_image)
        frame_idx += 1


def _xywh_to_xyxy(xywh: List[int]) -> List[int]:
    """ Convert bbox of form (left,top,width,height) to (left,top,right,bottom) """
    left, top, width, height = xywh
    return [left, top, left + width, top + height]


def _draw_boxes(
    im: np.ndarray,
    cur_df: pd.DataFrame,
    id_color_dict: Dict[int, Tuple[int, int, int]],
) -> np.ndarray:
    """ 
    Overlay bbox and id labels onto the frame
    Args:
        im: raw frame
        cur_df: dataframe of bboxes in the current frame
        id_color_dict: dictionary mapping ids to bbox colors
    """

    cur_ids = cur_df.id.tolist()

    for i, id in enumerate(cur_ids):
        xywh = [round(x) for x in cur_df.loc[cur_df.id == id].values[0][2:]]
        left, top, right, bottom = _xywh_to_xyxy(xywh)

        # box text and bar
        color = id_color_dict.get(id)
        label = str(cur_df.loc[cur_df.id == id].id.values[0])

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


def _compute_color_for_all_labels(
    id_list: List[int],
) -> Dict[int, Tuple[int, int, int]]:
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

    id_color_dict = dict(zip(id_list, color_list))

    return id_color_dict
