# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import argparse
import os
import os.path as osp


class opts(object):
    """
    Defines options for experiment settings, system settings, logging, model params,
    input config, training config, testing config, and tracking params.
    """

    def __init__(
        self,
        load_model: str = "",
        gpus=[0, 1],
        save_all: bool = False,
        arch: str = "dla_34",
        head_conv: int = -1,
        input_h: int = -1,
        input_w: int = -1,
        lr: float = 1e-4,
        lr_step=[20, 27],
        num_epochs: int = 30,
        num_iters: int = -1,
        val_intervals: int = 5,
        conf_thres: float = 0.6,
        det_thres: float = 0.3,
        nms_thres: float = 0.4,
        track_buffer: int = 30,
        min_box_area: float = 200,
        reid_dim: int = 512,
        root_dir: str = os.getcwd(),
    ) -> None:
        # Set defaults for parameters which are less important
        self.task = "mot"
        self.dataset = "jde"
        self.resume = False
        self.exp_id = "default"
        self.test = False
        self.num_workers = 8
        self.not_cuda_benchmark = False
        self.seed = 317
        self.print_iter = 0
        self.hide_data_time = False
        self.metric = "loss"
        self.vis_thresh = 0.5
        self.pad = 31
        self.num_stacks = 1
        self.down_ratio = 4
        self.input_res = -1
        self.num_iters = -1
        self.trainval = False
        self.K = 128
        self.not_prefetch_test = True
        self.keep_res = False
        self.fix_res = not self.keep_res
        self.test_mot16 = False
        self.val_mot15 = False
        self.test_mot15 = False
        self.val_mot16 = False
        self.test_mot16 = False
        self.val_mot17 = False
        self.val_mot20 = False
        self.test_mot20 = False
        self.input_video = ""
        self.output_format = "video"
        self.output_root = ""
        self.data_cfg = ""
        self.data_dir = ""
        self.mse_loss = False
        self.hm_gauss = 8
        self.reg_loss = "l1"
        self.hm_weight = 1
        self.off_weight = 1
        self.wh_weight = 0.1
        self.id_loss = "ce"
        self.id_weight = 1
        self.norm_wh = False
        self.dense_wh = False
        self.cat_spec_wh = False
        self.not_reg_offset = False
        self.reg_offset = not self.not_reg_offset

        # Set/overwrite defaults for parameters which are more important
        self.load_model = load_model
        self.gpus = gpus
        self.save_all = save_all
        self.arch = arch
        self.set_head_conv(head_conv)
        self.input_h = input_h
        self.input_w = input_w
        self.lr = lr
        self.lr_step = lr_step
        self.num_epochs = num_epochs
        self.val_intervals = val_intervals
        self.conf_thres = conf_thres
        self.det_thres = det_thres
        self.nms_thres = nms_thres
        self.track_buffer = track_buffer
        self.min_box_area = min_box_area
        self.reid_dim = reid_dim

        # init
        self._init_root_dir(root_dir)
        self._init_batch_sizes(batch_size=12, master_batch_size=-1)
        self._init_dataset_info()

    def _init_root_dir(self, value):
        self.root_dir = value
        self.exp_dir = osp.join(self.root_dir, "exp", self.task)
        self.save_dir = osp.join(self.exp_dir, self.exp_id)
        self.debug_dir = osp.join(self.save_dir, "debug")

    def _init_batch_sizes(self, batch_size, master_batch_size) -> None:
        self.batch_size = batch_size

        self.master_batch_size = (
            master_batch_size
            if master_batch_size != -1
            else self.batch_size // len(self.gpus)
        )
        rest_batch_size = self.batch_size - self.master_batch_size
        self.chunk_sizes = [self.master_batch_size]
        for i in range(len(self.gpus) - 1):
            chunk = rest_batch_size // (len(self.gpus) - 1)
            if i < rest_batch_size % (len(self.gpus) - 1):
                chunk += 1
            self.chunk_sizes.append(chunk)

    def _init_dataset_info(self) -> None:
        default_dataset_info = {
            "mot": {
                "default_resolution": [608, 1088],
                "num_classes": 1,
                "mean": [0.408, 0.447, 0.470],
                "std": [0.289, 0.274, 0.278],
                "dataset": "jde",
                "nID": 14455,
            }
        }

        class Struct:
            def __init__(self, entries):
                for k, v in entries.items():
                    self.__setattr__(k, v)

        dataset = Struct(default_dataset_info[self.task])
        self.dataset = dataset.dataset
        self.update_dataset_info_and_set_heads(dataset)

    def update_dataset_res(self, input_h, input_w) -> None:
        self.input_h = input_h
        self.input_w = input_w
        self.output_h = self.input_h // self.down_ratio
        self.output_w = self.input_w // self.down_ratio
        self.input_res = max(self.input_h, self.input_w)
        self.output_res = max(self.output_h, self.output_w)

    def update_dataset_info_and_set_heads(self, dataset) -> None:
        input_h, input_w = dataset.default_resolution
        self.mean, self.std = dataset.mean, dataset.std
        self.num_classes = dataset.num_classes

        # input_h(w): input_h overrides input_res overrides dataset default
        input_h = self.input_res if self.input_res > 0 else input_h
        input_w = self.input_res if self.input_res > 0 else input_w
        self.input_h = self.input_h if self.input_h > 0 else input_h
        self.input_w = self.input_w if self.input_w > 0 else input_w
        self.output_h = self.input_h // self.down_ratio
        self.output_w = self.input_w // self.down_ratio
        self.input_res = max(self.input_h, self.input_w)
        self.output_res = max(self.output_h, self.output_w)

        if self.task == "mot":
            self.heads = {
                "hm": self.num_classes,
                "wh": 2 if not self.cat_spec_wh else 2 * self.num_classes,
                "id": self.reid_dim,
            }
            if self.reg_offset:
                self.heads.update({"reg": 2})
            self.nID = dataset.nID
            self.img_size = (self.input_w, self.input_h)
        else:
            assert 0, "task not defined"

    def set_gpus(self, value):
        gpus_list = [int(gpu) for gpu in value.split(",")]
        self.gpus = (
            [i for i in range(len(gpus_list))] if gpus_list[0] >= 0 else [-1]
        )
        self.gpus_str = value

    def set_head_conv(self, value):
        h = value if value != -1 else 256
        self.head_conv = h
