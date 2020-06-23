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
        gpus: str = "0, 1",
        save_all: bool = False,
        arch: str = "dla_34",
        head_conv: int = -1,
        input_h: int = -1,
        input_w: int = -1,
        lr: float = 1e-4,
        lr_step: str = "20,27",
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
        self._init_opt()

        self.load_model = load_model
        self.gpus = gpus
        self.save_all = save_all
        self.arch = arch
        self.head_conv = head_conv
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
        self.root_dir = root_dir

        self._init_batch_sizes(batch_size=12, master_batch_size=-1)
        self._init_dataset_info()

    def _init_opt(self) -> None:
        """ Default values for params that aren't exposed by TrackingLearner """
        self._opt = argparse.Namespace()

        self._opt.task = "mot"
        self._opt.dataset = "jde"
        self._opt.resume = False
        self._opt.exp_id = "default"
        self._opt.test = False
        self._opt.num_workers = 8
        self._opt.not_cuda_benchmark = False
        self._opt.seed = 317
        self._opt.print_iter = 0
        self._opt.hide_data_time = False
        self._opt.metric = "loss"
        self._opt.vis_thresh = 0.5
        self._opt.pad = 31
        self._opt.num_stacks = 1
        self._opt.down_ratio = 4
        self._opt.input_res = -1
        self._opt.num_iters = -1
        self._opt.trainval = False
        self._opt.K = 128
        self._opt.not_prefetch_test = True
        self._opt.keep_res = False
        self._opt.fix_res = not self._opt.keep_res
        self._opt.test_mot16 = False
        self._opt.val_mot15 = False
        self._opt.test_mot15 = False
        self._opt.val_mot16 = False
        self._opt.test_mot16 = False
        self._opt.val_mot17 = False
        self._opt.val_mot20 = False
        self._opt.test_mot20 = False
        self._opt.input_video = ""
        self._opt.output_format = "video"
        self._opt.output_root = ""
        self._opt.data_cfg = ""
        self._opt.data_dir = ""
        self._opt.mse_loss = False
        self._opt.hm_gauss = 8
        self._opt.reg_loss = "l1"
        self._opt.hm_weight = 1
        self._opt.off_weight = 1
        self._opt.wh_weight = 0.1
        self._opt.id_loss = "ce"
        self._opt.id_weight = 1
        self._opt.norm_wh = False
        self._opt.dense_wh = False
        self._opt.cat_spec_wh = False
        self._opt.not_reg_offset = False
        self._opt.reg_offset = not self._opt.not_reg_offset

    def _init_batch_sizes(self, batch_size, master_batch_size) -> None:
        self._opt.batch_size = batch_size

        self._opt.master_batch_size = (
            master_batch_size
            if master_batch_size != -1
            else self._opt.batch_size // len(self._opt.gpus)
        )
        rest_batch_size = self._opt.batch_size - self._opt.master_batch_size
        self._opt.chunk_sizes = [self._opt.master_batch_size]
        for i in range(len(self.gpus) - 1):
            chunk = rest_batch_size // (len(self._opt.gpus) - 1)
            if i < rest_batch_size % (len(self._opt.gpus) - 1):
                chunk += 1
            self._opt.chunk_sizes.append(chunk)

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

        dataset = Struct(default_dataset_info[self._opt.task])
        self._opt.dataset = dataset.dataset
        self.update_dataset_info_and_set_heads(dataset)

    def update_dataset_res(self, input_h, input_w) -> None:
        self._opt.input_h = input_h
        self._opt.input_w = input_w
        self._opt.output_h = self._opt.input_h // self._opt.down_ratio
        self._opt.output_w = self._opt.input_w // self._opt.down_ratio
        self._opt.input_res = max(self._opt.input_h, self._opt.input_w)
        self._opt.output_res = max(self._opt.output_h, self._opt.output_w)

    def update_dataset_info_and_set_heads(self, dataset) -> None:
        input_h, input_w = dataset.default_resolution
        self._opt.mean, self._opt.std = dataset.mean, dataset.std
        self._opt.num_classes = dataset.num_classes

        # input_h(w): opt.input_h overrides opt.input_res overrides dataset default
        input_h = self._opt.input_res if self._opt.input_res > 0 else input_h
        input_w = self._opt.input_res if self._opt.input_res > 0 else input_w
        self.input_h = self._opt.input_h if self._opt.input_h > 0 else input_h
        self.input_w = self._opt.input_w if self._opt.input_w > 0 else input_w
        self._opt.output_h = self._opt.input_h // self._opt.down_ratio
        self._opt.output_w = self._opt.input_w // self._opt.down_ratio
        self._opt.input_res = max(self._opt.input_h, self._opt.input_w)
        self._opt.output_res = max(self._opt.output_h, self._opt.output_w)

        if self._opt.task == "mot":
            self._opt.heads = {
                "hm": self._opt.num_classes,
                "wh": 2
                if not self._opt.cat_spec_wh
                else 2 * self._opt.num_classes,
                "id": self._opt.reid_dim,
            }
            if self._opt.reg_offset:
                self._opt.heads.update({"reg": 2})
            self._opt.nID = dataset.nID
            self._opt.img_size = (self._opt.input_w, self._opt.input_h)
        else:
            assert 0, "task not defined"

    ### getters and setters ###
    @property
    def load_model(self):
        return self._load_model

    @load_model.setter
    def load_model(self, value):
        self._load_model = value
        self._opt.load_model = self._load_model

    @property
    def gpus(self):
        return self._gpus

    @gpus.setter
    def gpus(self, value):
        self._gpus_str = value
        gpus_list = [int(gpu) for gpu in value.split(",")]
        self._gpus = (
            [i for i in range(len(gpus_list))] if gpus_list[0] >= 0 else [-1]
        )
        self._opt.gpus_str = self._gpus_str
        self._opt.gpus = self._gpus

    @property
    def save_all(self):
        return self._save_all

    @save_all.setter
    def save_all(self, value):
        self._save_all = value
        self._opt.save_all = self._save_all

    @property
    def arch(self):
        return self._arch

    @arch.setter
    def arch(self, value):
        self._arch = value
        self._opt.arch = self._arch

    @property
    def head_conv(self):
        return self._head_conv

    @head_conv.setter
    def head_conv(self, value):
        self._head_conv = value if value != -1 else 256
        self._opt.head_conv = self._head_conv

    @property
    def input_h(self):
        return self._input_h

    @input_h.setter
    def input_h(self, value):
        self._input_h = value
        self._opt.input_h = self._input_h

    @property
    def input_w(self):
        return self._input_w

    @input_w.setter
    def input_w(self, value):
        self._input_w = value
        self._opt.input_w = self._input_w

    @property
    def lr(self):
        return self._lr

    @lr.setter
    def lr(self, value):
        self._lr = value
        self._opt.lr = self._lr

    @property
    def lr_step(self):
        return self._lr_step

    @lr_step.setter
    def lr_step(self, value):
        self._lr_step = [int(i) for i in value.split(",")]
        self._opt.lr_step = self._lr_step

    @property
    def num_epochs(self):
        return self._num_epochs

    @num_epochs.setter
    def num_epochs(self, value):
        self._num_epochs = value
        self._opt.num_epochs = self._num_epochs

    @property
    def val_intervals(self):
        return self._val_intervals

    @val_intervals.setter
    def val_intervals(self, value):
        self._val_intervals = value
        self._opt.val_intervals = self._val_intervals

    @property
    def conf_thres(self):
        return self._conf_thres

    @conf_thres.setter
    def conf_thres(self, value):
        self._conf_thres = value
        self._opt.conf_thres = self._conf_thres

    @property
    def det_thres(self):
        return self._det_thres

    @det_thres.setter
    def det_thres(self, value):
        self._det_thres = value
        self._opt.det_thres = self._det_thres

    @property
    def nms_thres(self):
        return self._nms_thres

    @nms_thres.setter
    def nms_thres(self, value):
        self._nms_thres = value
        self._opt.nms_thres = self._nms_thres

    @property
    def track_buffer(self):
        return self._track_buffer

    @track_buffer.setter
    def track_buffer(self, value):
        self._track_buffer = value
        self._opt.track_buffer = self._track_buffer

    @property
    def min_box_area(self):
        return self._min_box_area

    @min_box_area.setter
    def min_box_area(self, value):
        self._min_box_area = value
        self._opt.min_box_area = self._min_box_area

    @property
    def reid_dim(self):
        return self._reid_dim

    @reid_dim.setter
    def reid_dim(self, value):
        self._reid_dim = value
        self._opt.reid_dim = self._reid_dim

    @property
    def root_dir(self):
        return self._root_dir

    @root_dir.setter
    def root_dir(self, value):
        self._root_dir = value
        self._opt.root_dir = self._root_dir

        self._opt.exp_dir = osp.join(self._root_dir, "exp", self._opt.task)
        self._opt.save_dir = osp.join(self._opt.exp_dir, self._opt.exp_id)
        self._opt.debug_dir = osp.join(self._opt.save_dir, "debug")

    @property
    def device(self):
        return self._device

    @device.setter
    def device(self, value):
        self._device = value
        self._opt.device = self._device

    ### getters only ####
    @property
    def opt(self):
        return self._opt

    @property
    def resume(self):
        return self._resume

    @property
    def task(self):
        return self._opt.task

    @property
    def save_dir(self):
        return self._opt.save_dir

    @property
    def chunk_sizes(self):
        return self._opt.chunk_sizes

    @property
    def heads(self):
        return self._opt.heads
