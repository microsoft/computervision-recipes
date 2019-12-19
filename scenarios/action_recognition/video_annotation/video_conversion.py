# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""
ffmpeg video conversion from 'asf' to 'mp4' and 'avi' without video quality loss:
referenced stackoverflow answer:
https://stackoverflow.com/questions/15049829/remux-to-mkv-but-add-all-streams-using-ffmpeg/15052662#15052662
"""

import argparse
import os

sys.path.append("../../../utils_cv/action_recognition/")
from video_annotation_utils import video_format_conversion


def main(video_dir, output_dir):
    for output_format in ["mp4", "avi"]:
        output_sub_dir = os.path.join(output_dir, output_format)
        if not os.path.exists(output_sub_dir):
            os.makedirs(output_sub_dir)

        # get all the files in the directory
        for video_file in os.listdir(video_dir):
            if video_file[-3:] == "asf":
                video_path = os.path.join(video_dir, video_file)
                output_file_name = video_file[:-4] + ".{}".format(
                    output_format
                )
                output_path = os.path.join(output_sub_dir, output_file_name)
                video_format_conversion(
                    video_path, output_path, h264_format=True
                )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input_dir", help="Input video dir")
    parser.add_argument(
        "-o",
        "--output_dir",
        help="Output dir where the converted videos will be stored",
        default="./outputs",
    )
    args = parser.parse_args()

    main(args.input_dir, args.output_dir)
