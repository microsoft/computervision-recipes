# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

COCO_keypoint_meta = {
    # `labels` gives the names of keypoints
    "labels": [
        "nose",  # 0
        "left_eye",  # 1
        "right_eye",  # 2
        "left_ear",  # 3
        "right_ear",  # 4
        "left_shoulder",  # 5
        "right_shoulder",  # 6
        "left_elbow",  # 7
        "right_elbow",  # 8
        "left_wrist",  # 9
        "right_wrist",  # 10
        "left_hip",  # 11
        "right_hip",  # 12
        "left_knee",  # 13
        "right_knee",  # 14
        "left_ankle",  # 15
        "right_ankle",  # 16
    ],
    # `skeleton` is used to specify how keypoints are connected with each
    # other when drawing.  For example, `[15, 13]` means left_ankle (15) will
    # connect to left_knee when plotting on the image, and `[13, 11]` means
    # left_knee (13) will connect to left_hip (11).
    "skeleton": [
        [15, 13],  # left_ankle -- left_knee
        [13, 11],  # left_knee -- left_hip
        [16, 14],  # right_ankle -- right_knee
        [14, 12],  # right_knee -- right_hip
        [11, 12],  # left_hip -- right_hip
        [5, 11],  # left_shoulder -- left_hip
        [6, 12],  # right_shoulder -- right_hip
        [5, 6],  # left_shoulder -- right_shoulder
        [5, 7],  # left_shoulder -- left_elbow
        [6, 8],  # right_shoulder -- right_elbow
        [7, 9],  # left_elbow -- left_wrist
        [8, 10],  # right_elbow -- right_wrist
        [1, 2],  # left_eye -- right_eye
        [0, 1],  # nose -- left_eye
        [0, 2],  # nose -- right_eye
        [1, 3],  # left_eye -- left_ear
        [2, 4],  # right_eye -- right_ear
        [3, 5],  # left_ear -- left_shoulder
        [4, 6],  # right_ear -- right_shoulder
    ],
    # When an image is flipped horizontally, some keypoints related to the
    # concept of left and right would change to its opposite meaning.  For
    # example, left eye would become right eye.
    # `hflip_inds` is used in the horizontal flip transformation for data
    # augmentation during training to specify what the keypoint will become.
    # In other words, `COCO_keypoint_meta["hflip_inds"][0]` specify what nose
    # (`COCO_keypoint_meta["labels"][0]`) will become when an image is
    # flipped horizontally.  Because nose would still be nose even after
    # flipping, its value is still 0.  Left eye
    # (`COCO_keypoint_meta["labels"][1]`) will be right eye
    # (`COCO_keypoint_meta["labels"][2]`), so the value of
    # `COCO_keypoint_meta["hflip_inds"][1]` should be 2.
    "hflip_inds": [
        0,  # nose
        2,  # left_eye -> right_eye
        1,  # right_eye -> left_eye
        4,  # left_ear -> right_ear
        3,  # right_ear -> left_ear
        6,  # left_shoulder -> right_shoulder
        5,  # right_shoulder -> left_shoulder
        8,  # left_elbow -> right_elbow
        7,  # right_elbow -> left_elbow
        10,  # left_wrist -> right_wrist
        9,  # right_wrist -> left_wrist
        12,  # left_hip -> right_hip
        11,  # right_hip -> left_hip
        14,  # left_knee -> right_knee
        13,  # right_knee -> left_knee
        16,  # left_ankle -> right_ankle
        15,  # right_ankle -> left_ankle
    ],
}
