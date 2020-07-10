# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

from collections import defaultdict
from copy import deepcopy
import glob
import os
import os.path as osp
from typing import Dict, List, Optional, Tuple

import cv2
import matplotlib.pyplot as plt
import motmetrics as mm
import numpy as np

import torch
import torch.cuda as cuda
from torch.utils.data import DataLoader

from .bbox import TrackingBbox
from ..common.gpu import torch_device
from .dataset import TrackingDataset, boxes_to_mot
from .opts import opts

from .references.fairmot.datasets.dataset.jde import LoadImages, LoadVideo
from .references.fairmot.models.model import (
    create_model,
    load_model,
    save_model,
)
from .references.fairmot.tracker.multitracker import JDETracker
from .references.fairmot.tracking_utils.evaluation import Evaluator
from .references.fairmot.trains.train_factory import train_factory


def _get_gpu_str():
    if cuda.is_available():
        devices = [str(x) for x in range(cuda.device_count())]
        return ",".join(devices)
    else:
        return "-1"  # cpu


def _get_frame(input_video: str, frame_id: int):
    video = cv2.VideoCapture()
    video.open(input_video)
    video.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
    _, im = video.read()
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    return im


def savetxt_results(
    results: Dict[int, List[TrackingBbox]],
    exp_name: str,
    root_path: str,
    result_filename: str,
) -> str:
    """Save tracking results to txt in provided path.

    Args:
        results: prediction results from predict() function, i.e. Dict[int, List[TrackingBbox]]
        exp_name: subfolder for each experiment
        root_path: root path for results saved
        result_filename: saved prediction results txt file; end with '.txt'
    Returns:
        result_path: saved prediction results txt file path
    """

    # Convert prediction results to mot format
    bboxes_mot = boxes_to_mot(results)

    # Save results
    result_path = osp.join(root_path, exp_name, result_filename)
    np.savetxt(result_path, bboxes_mot, delimiter=",", fmt="%s")

    return result_path


def evaluate_mot(gt_root_path: str, exp_name: str, result_path: str) -> object:
    """ eval code that calls on 'motmetrics' package in referenced FairMOT script, to produce MOT metrics on inference, given ground-truth.
    Args:
        gt_root_path: path of dataset containing GT annotations in MOTchallenge format (xywh)
        exp_name: subfolder for each experiment
        result_path: saved prediction results txt file path
    Returns:
        mot_accumulator: MOTAccumulator object from pymotmetrics package
    """
    # Implementation inspired from code found here: https://github.com/ifzhang/FairMOT/blob/master/src/track.py
    evaluator = Evaluator(gt_root_path, exp_name, "mot")

    # Run evaluation using pymotmetrics package
    mot_accumulator = evaluator.eval_file(result_path)

    return mot_accumulator


def mot_summary(accumulators: list, exp_names: list) -> str:
    """Given a list of MOTAccumulators, get total summary by method in 'motmetrics', containing metrics scores

    Args:
        accumulators: list of MOTAccumulators
        exp_names: list of experiment names (str) corresponds to MOTAccumulators
    Returns:
        strsummary: str output by method in 'motmetrics', containing metrics scores
    """
    metrics = mm.metrics.motchallenge_metrics
    mh = mm.metrics.create()

    summary = Evaluator.get_summary(accumulators, exp_names, metrics)
    strsummary = mm.io.render_summary(
        summary,
        formatters=mh.formatters,
        namemap=mm.io.motchallenge_metric_names,
    )

    return strsummary


class TrackingLearner(object):
    """Tracking Learner for Multi-Object Tracking"""

    def __init__(
        self,
        dataset: Optional[TrackingDataset] = None,
        model_path: Optional[str] = None,
        arch: str = "dla_34",
        head_conv: int = -1,
    ) -> None:
        """
        Initialize learner object.

        Defaults to the FairMOT model.

        Args:
            dataset: optional dataset (required for training)
            model_path: optional path to pretrained model (defaults to all_dla34.pth)
            arch: the model architecture
                Supported architectures: resdcn_34, resdcn_50, resfpndcn_34, dla_34, hrnet_32
            head_conv: conv layer channels for output head. None maps to the default setting.
                Set 0 for no conv layer, 256 for resnets, and 256 for dla
        """
        self.opt = opts()
        self.opt.arch = arch
        self.opt.set_head_conv(head_conv)
        self.opt.set_gpus(_get_gpu_str())
        self.opt.device = torch_device()
        self.dataset = dataset
        self.model = None
        self._init_model(model_path)

    def _init_model(self, model_path) -> None:
        """
        Initialize the model.

        Args:
            model_path: optional path to pretrained model (defaults to all_dla34.pth)
        """
        if not model_path:
            model_path = osp.join(self.opt.root_dir, "models", "all_dla34.pth")
        assert osp.isfile(
            model_path
        ), f"Model weights not found at {model_path}"

        self.opt.load_model = model_path

    def fit(
        self, lr: float = 1e-4, lr_step: str = "20,27", num_epochs: int = 30
    ) -> None:
        """
        The main training loop.

        Args:
            lr: learning rate for batch size 32
            lr_step: when to drop learning rate by 10
            num_epochs: total training epochs

        Raise:
            Exception if dataset is undefined

        Implementation inspired from code found here: https://github.com/ifzhang/FairMOT/blob/master/src/train.py
        """
        if not self.dataset:
            raise Exception("No dataset provided")
        if type(lr_step) is not list:
            lr_step = [lr_step]
        lr_step = [int(x) for x in lr_step]

        # update parameters
        self.opt.lr = lr
        self.opt.lr_step = lr_step
        self.opt.num_epochs = num_epochs
        opt = deepcopy(self.opt)  #to avoid fairMOT over-writing opt

        # update dataset options
        opt.update_dataset_info_and_set_heads(self.dataset.train_data)

        # initialize dataloader
        train_loader = self.dataset.train_dl
        self.model = create_model(
            opt.arch, opt.heads, opt.head_conv
        )
        self.model = load_model(self.model, opt.load_model)
        self.optimizer = torch.optim.Adam(self.model.parameters(), opt.lr)
        start_epoch = 0

        Trainer = train_factory[opt.task]
        trainer = Trainer(opt, self.model, self.optimizer)
        trainer.set_device(opt.gpus, opt.chunk_sizes, opt.device)

        # initialize loss vars
        self.losses_dict = defaultdict(list)

        # training loop
        for epoch in range(
            start_epoch + 1, start_epoch + opt.num_epochs + 1
        ):
            print(
                "=" * 5,
                f" Epoch: {epoch}/{start_epoch + opt.num_epochs} ",
                "=" * 5,
            )
            self.epoch = epoch
            log_dict_train, _ = trainer.train(epoch, train_loader)
            for k, v in log_dict_train.items():
                if k == "time":
                    print(f"{k}:{v} min")
                else:
                    print(f"{k}: {v}")
            if epoch in opt.lr_step:
                lr = opt.lr * (0.1 ** (opt.lr_step.index(epoch) + 1))
                for param_group in self.optimizer.param_groups:
                    param_group["lr"] = lr

            # store losses in each epoch
            for k, v in log_dict_train.items():
                if k in ["loss", "hm_loss", "wh_loss", "off_loss", "id_loss"]:
                    self.losses_dict[k].append(v)

    def plot_training_losses(self, figsize: Tuple[int, int] = (10, 5)) -> None:
        """
        Plot training loss.

        Args:
            figsize (optional): width and height wanted for figure of training-loss plot

        """
        fig = plt.figure(figsize=figsize)
        ax1 = fig.add_subplot(1, 1, 1)

        ax1.set_xlim([0, len(self.losses_dict["loss"]) - 1])
        ax1.set_xticks(range(0, len(self.losses_dict["loss"])))
        ax1.set_xlabel("epochs")
        ax1.set_ylabel("losses")

        ax1.plot(self.losses_dict["loss"], c="r", label="loss")
        ax1.plot(self.losses_dict["hm_loss"], c="y", label="hm_loss")
        ax1.plot(self.losses_dict["wh_loss"], c="g", label="wh_loss")
        ax1.plot(self.losses_dict["off_loss"], c="b", label="off_loss")
        ax1.plot(self.losses_dict["id_loss"], c="m", label="id_loss")

        plt.legend(loc="upper right")
        fig.suptitle("Training losses over epochs")

    def save(self, path) -> None:
        """
        Save the model to a specified path.
        """
        model_dir, _ = osp.split(path)
        os.makedirs(model_dir, exist_ok=True)

        save_model(path, self.epoch, self.model, self.optimizer)
        print(f"Model saved to {path}")

    def evaluate(
        self, results: Dict[int, List[TrackingBbox]], gt_root_path: str
    ) -> str:

        """
        Evaluate performance wrt MOTA, MOTP, track quality measures, global ID measures, and more,
        as computed by py-motmetrics on a single experiment. By default, use 'single_vid' as exp_name.

        Args:
            results: prediction results from predict() function, i.e. Dict[int, List[TrackingBbox]]
            gt_root_path: path of dataset containing GT annotations in MOTchallenge format (xywh)
        Returns:
            strsummary: str output by method in 'motmetrics' package, containing metrics scores
        """

        # Implementation inspired from code found here: https://github.com/ifzhang/FairMOT/blob/master/src/track.py
        result_path = savetxt_results(
            results, "single_vid", gt_root_path, "results.txt"
        )
        # Save tracking results in tmp
        mot_accumulator = evaluate_mot(gt_root_path, "single_vid", result_path)
        strsummary = mot_summary([mot_accumulator], ("single_vid",))
        return strsummary

    def eval_mot(
        self,
        conf_thres: float,
        track_buffer: int,
        data_root: str,
        seqs: list,
        result_root: str,
        exp_name: str,
        run_eval: bool = True,
    ) -> str:
        """
        Call the prediction function, saves the tracking results to txt file and provides the evaluation results with motmetrics format.
        Args:
            conf_thres: confidence thresh for tracking
            track_buffer: tracking buffer
            data_root: data root path
            seqs: list of video sequences subfolder names under MOT challenge data
            result_root: tracking result path
            exp_name: experiment name
            run_eval: if we evaluate on provided data
        Returns:
            strsummary: str output by method in 'motmetrics' package, containing metrics scores
        """
        accumulators = []
        eval_path = osp.join(result_root, exp_name)
        if not osp.exists(eval_path):
            os.makedirs(eval_path)

        # Loop over all video sequences
        for seq in seqs:
            result_filename = "{}.txt".format(seq)
            im_path = osp.join(data_root, seq, "img1")
            result_path = osp.join(result_root, exp_name, result_filename)
            with open(osp.join(data_root, seq, "seqinfo.ini")) as seqinfo_file:
                meta_info = seqinfo_file.read()

            # frame_rate is set from seqinfo.ini by frameRate
            frame_rate = int(
                meta_info[
                    meta_info.find("frameRate")
                    + 10 : meta_info.find("\nseqLength")
                ]
            )

            # Run model inference
            if not osp.exists(result_path):
                eval_results = self.predict(
                    im_or_video_path=im_path,
                    conf_thres=conf_thres,
                    track_buffer=track_buffer,
                    frame_rate=frame_rate,
                )
                result_path = savetxt_results(
                    eval_results, exp_name, result_root, result_filename
                )
                print(f"Saved tracking results to {result_path}")
            else:
                print(f"Loaded tracking results from {result_path}")

            # Run evaluation
            if run_eval:
                print(f"Evaluate seq: {seq}")
                mot_accumulator = evaluate_mot(data_root, seq, result_path)
                accumulators.append(mot_accumulator)

        if run_eval:
            strsummary = mot_summary(accumulators, seqs)
            return strsummary
        else:
            return None

    def predict(
        self,
        im_or_video_path: str,
        conf_thres: float = 0.6,
        track_buffer: int = 30,
        min_box_area: float = 200,
        frame_rate: int = 30,
    ) -> Dict[int, List[TrackingBbox]]:
        """
        Run inference on an image or video path.

        Args:
            im_or_video_path: path to image(s) or video. Supports jpg, jpeg, png, tif formats for images.
                Supports mp4, avi formats for video.
            conf_thres: confidence thresh for tracking
            track_buffer: tracking buffer
            min_box_area: filter out tiny boxes
            frame_rate: frame rate

        Returns a list of TrackingBboxes

        Implementation inspired from code found here: https://github.com/ifzhang/FairMOT/blob/master/src/track.py
        """
        self.opt.conf_thres = conf_thres
        self.opt.track_buffer = track_buffer
        self.opt.min_box_area = min_box_area
        opt = deepcopy(self.opt)  #to avoid fairMOT over-writing opt

        # initialize tracker
        tracker = JDETracker(opt, frame_rate=frame_rate, model=self.model)

        # initialize dataloader
        dataloader = self._get_dataloader(im_or_video_path)

        frame_id = 0
        out = {}
        results = []
        for path, img, img0 in dataloader:
            blob = torch.from_numpy(img).cuda().unsqueeze(0)
            online_targets = tracker.update(blob, img0)
            online_bboxes = []
            for t in online_targets:
                tlwh = t.tlwh
                tlbr = t.tlbr
                tid = t.track_id
                vertical = tlwh[2] / tlwh[3] > 1.6
                if tlwh[2] * tlwh[3] > opt.min_box_area and not vertical:
                    bb = TrackingBbox(
                        tlbr[0], tlbr[1], tlbr[2], tlbr[3], frame_id, tid
                    )
                    online_bboxes.append(bb)
            out[frame_id] = online_bboxes
            frame_id += 1

        return out

    def _get_dataloader(self, im_or_video_path: str) -> DataLoader:
        """
        Create a dataloader from images or video in the given path.

        Args:
            im_or_video_path: path to a root directory of images, or single video or image file.
                Supports jpg, jpeg, png, tif formats for images. Supports mp4, avi formats for video

        Return:
            Dataloader

        Raise:
            Exception if file format is not supported

        Implementation inspired from code found here: https://github.com/ifzhang/FairMOT/blob/master/src/lib/datasets/dataset/jde.py
        """
        im_format = [".jpg", ".jpeg", ".png", ".tif"]
        video_format = [".mp4", ".avi"]

        # if path is to a root directory of images

        if (
            osp.isdir(im_or_video_path)
            and len(
                list(
                    filter(
                        lambda x: osp.splitext(x)[1].lower() in im_format,
                        sorted(glob.glob("%s/*.*" % im_or_video_path)),
                    )
                )
            )
            > 0
        ):
            return LoadImages(im_or_video_path)
        # if path is to a single video file
        elif (
            osp.isfile(im_or_video_path)
            and osp.splitext(im_or_video_path)[1] in video_format
        ):
            return LoadVideo(im_or_video_path)
        # if path is to a single image file
        elif (
            osp.isfile(im_or_video_path)
            and osp.splitext(im_or_video_path)[1] in im_format
        ):
            return LoadImages(im_or_video_path)
        else:
            raise Exception("Image or video format not supported")
