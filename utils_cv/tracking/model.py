import os
import os.path as osp
import torch
import argparse

from typing import List, Tuple, Union, Optional, Dict
from pathlib import Path
from torch.utils.data import DataLoader
from torchvision.transforms import transforms as T

from .bbox import TrackingBbox
from .references.fairmot.opts import opts
from .references.fairmot.datasets.dataset.jde import LoadImages, LoadVideo
from .references.fairmot.tracker.multitracker import JDETracker
from .references.fairmot.models.model import create_model, load_model, save_model
from ..common.gpu import torch_device

class TrackingLearner(object):
    def __init__(self) -> None:
        """
        Initialize learner object.
        Defaults to FairMOT
        """
        # TODO setup logging, savedir, etc

    def fit(
        self,
        data_root,
        data_path,
        im_size,
        lr: float = 1e-4,
        lr_step: str = "20,27",
        num_epochs: int = 30,
        batch_size: int = 12,
        num_iters: int = -1,
        val_intervals: int = 5,
    ) -> None:
        # prepare options
        train_args = f"--lr {lr} \
                        --lr_step {lr_step} \
                        --num_epochs {num_epochs} \
                        --batch_size {batch_size} \
                        --num_iters {num_iters} \
                        --val_intervals {val_intervals}"
        opt = opts().init(train_args)
        opt.device = torch_device()

        # initialize dataset
        dataset = self._init_dataset(opt, data_root, train_path, im_size)
        opt = opts().update_dataset_info_and_set_heads(opt, dataset)
        
        train_loader = DataLoader(
            dataset,
            batch_size=opt.batch_size,
            shuffle=True,
            num_workers=opt.num_workers,
            pin_memory=True,
            drop_last=True)

        # initialize model
        model, optimizer, start_epoch = self._init_model(opt)

        Trainer = train_factory[opt.task]
        trainer = Trainer(opt, model, optimizer)
        trainer.set_device(opt.gpus, opt.chunk_sizes, opt.device)
        best = 1e10

        # training loop
        for epoch in range(start_epoch+1, opt.num_epochs+1):
            mark = epoch if opt.save_all else 'last'
            log_dict_train, _ = trainer.train(epoch, train_loader)
            ## TODO logging
            if opt.val_intervals > 0 and epoch % opt.val_intervals == 0:
                save_model(osp.join(opt.save_dir, f"model_{mark}.pth"), epoch, model, optimizer)
            else:
                save_model(osp.join(opt.save_dir, f"model_last.pth", epoch, model, optimizer))

        for epoch in opt.lr_step:
            save_model(os.path.join(opt.save_dir, f"model_{epoch}.pth"), epoch, model, optimizer)
            lr = opt.lr * (0.1**opt.lr_step.index(epoch)+1)
            ## TODO logging
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

        if epoch % 5 == 0: ## TODO make this a param
            save_model(osp.join(opt.save_dir, f"model_{epoch}.pth", epoch, model, optimizer))

    def _init_dataset(self, opt, data_root, train_path, im_size):
        Dataset = get_dataset(opt.dataset, opt.task)
        transforms = T.Compose([T.ToTensor()])
        return Dataset(opt, data_root, train_path, im_size, augment=True, transforms=transforms)

    def _init_model(self, opt):
        model = create_model(opt.arch, opt.heads, opt.head_conv)
        optimizer = torch.optim.Adam(model.parameters(), opt.lr)
        start_epoch = 0
        if opt.load_model != '':
            model, optimizer, start_epoch = load_model(model, opt.load_model, optimizer, opt.resume, opt.lr, opt.lr_step)
        return model, optimizer, start_epoch
                                    
    def predict(self,
        im_path: Path = None,
        video_path: Path = None,
        load_model: Path = None,
        conf_thres: float = 0.6,
        det_thres: float = 0.3,
        nms_thres: float = 0.4,
        track_buffer: int = 30,
        min_box_area: float = 200,
        frame_rate: int = 30) -> Dict[int, List[TrackingBbox]]:
        # if im_path is None, video_path must not be, and vice versa
        assert(bool(im_path) ^ bool(video_path), "Either im_path or video_path must be defined.")

        device = torch_device()
        track_args = f"--conf_thres {conf_thres} \
                        --det_thres {det_thres} \
                        --nms_thres {nms_thres} \
                        --track_buffer {track_buffer} \
                        --min_box_area {min_box_area} \
                        --gpus {device}"
        opt = opts().init(track_args)
        self.tracker = JDETracker(opt, frame_rate=frame_rate)

        if video_path is not None:
            dataloader = LoadVideo(video_path, im_size)
        else:
            dataloader = LoadImages(im_path, im_size)

        frame_id = 0
        out = {}
        for path, img, img0 in dataloader:
            online_targets = self.tracker.update(blob, img0)
            online_bboxes = []
            for t in online_targets:
                tlwh = t.tlwh
                tlbr = t.tlbr
                tid = t.track_id
                vertical = tlwh[2] / tlwh[3] > 1.6
                if tlwh[2] * tlwh[3] > opt.min_box_area and not vertical:
                    bb = TrackingBbox(tlbr[1], tlbr[0], tlbr[3], tlbr[2], frame_id, tid)
                    online_bboxes.append(bb)
            out[frame_id] = online_bboxes
            frame_id += 1
        return out
            
