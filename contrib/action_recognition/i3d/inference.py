# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

from pathlib import Path
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import datasets, transforms

from videotransforms import (
    GroupScale, GroupCenterCrop, GroupNormalize, Stack
)
from models.pytorch_i3d import InceptionI3d
from dataset import I3DDataSet
from test import load_model


def load_image(frame_file):
    try:
        img = Image.open(frame_file).convert('RGB')
        return img
    except:
        print("Couldn't load image:{}".format(frame_file))
        return None


def load_frames(frame_paths):
    frame_list = []
    for frame in frame_paths:
        frame_list.append(load_image(frame))
    return frame_list


def construct_input(frame_list):

    transform = torchvision.transforms.Compose([
                    GroupScale(config.TRAIN.RESIZE_MIN),
                    GroupCenterCrop(config.TRAIN.INPUT_SIZE),
                    GroupNormalize(modality="RGB"),
                    Stack(),
                ])

    process_data = transform(frame_list)
    return process_data.unsqueeze(0)


def predict_input(model, input):
    input = input.cuda(non_blocking=True)
    output = model(input)
    output = torch.mean(output, dim=2)
    return output


def predict_over_video(video_frame_list, window_width=9, stride=1):

    if window_width < 9:
        raise ValueError("window_width must be 9 or greater")

    print("Loading model...")

    model = load_model(
        modality="RGB",
        state_dict_file="pretrained_chkpt/rgb_hmdb_split1.pt"
    )

    model.eval()

    print("Predicting actions over {0} frames".format(len(video_frame_list)))

    with torch.no_grad():

        window_count = 0

        for i in range(stride+window_width-1, len(video_frame_list), stride):
            window_frame_list = [video_frame_list[j] for j in range(i-window_width, i)]
            frames = load_frames(window_frame_list)
            batch = construct_input(frames)
            window_predictions = predict_input(model, batch)
            window_proba = F.softmax(window_predictions, dim=1)
            window_top_pred = window_proba.max(1)
            print(("Window:{0} Class pred:{1} Class proba:{2}".format(
                window_count,
                window_top_pred.indices.cpu().numpy()[0],
                window_top_pred.values.cpu().numpy()[0])
            ))
            window_count += 1



if __name__ == "__main__":

    # Provide list of filepaths to video frames
    frame_paths = []

    predict_over_video(frame_list, window_width=64, stride=32)