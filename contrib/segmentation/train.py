"""
"""
import argparse
import copy
import logging
import multiprocessing
import time
import uuid
from os.path import join
from pathlib import Path
from typing import Dict, Tuple

import albumentations as A
import torch
import torch.nn as nn
from torch.utils.data.dataloader import DataLoader

from src.datasets.semantic_segmentation import (
    SemanticSegmentationDataset,
    SemanticSegmentationStochasticPatchingDataset,
    ToySemanticSegmentationDataset,
)
from src.losses.loss import semantic_segmentation_class_balancer
from src.metrics.metrics import get_semantic_segmentation_metrics, log_metrics
from src.models.deeplabv3 import get_deeplabv3
from src.models.fcn_resnet50 import get_fcn_resnet50

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def get_augmentation_for_erosion(patch_dim: Tuple[int, int] = (512, 512)):
    transform = A.Compose(
        [
            # This allows meaningful yet stochastic cropped views
            A.CropNonEmptyMaskIfExists(patch_dim[0], patch_dim[1], p=1),
            A.RandomRotate90(p=0.5),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.Blur(p=0.25),
            A.ColorJitter(p=0.25),
            A.GaussNoise(p=0.25),
            A.Cutout(p=0.5, num_holes=64, max_h_size=8, max_w_size=8),
            A.RandomBrightnessContrast(p=0.25),
        ],
    )
    return transform


def get_validation_preprocessing(patch_dim: Tuple[int, int] = (512, 512)):
    """Due to the image size being too large, to get some idea of the validation
    performance without needlessly running through the full high resolution images
    we take a cropped view of the masks
    """
    transform = A.Compose(
        [
            # This allows meaningful yet stochastic cropped views
            A.CropNonEmptyMaskIfExists(patch_dim[0], patch_dim[1], p=1),
        ],
    )
    return transform


def get_preprocessing(
    p_hflip=0.5,
    p_random_rotate=0.5,
    p_vflip=0.5,
    p_blur=0.5,
    p_jitter=0.5,
    p_gauss_noise=0.5,
    p_cutout=0.5,
    p_random_brightness=0.5,
    p_random_sun_flare=0.5,
    p_transform=1,
):
    transform = A.Compose(
        [
            A.OneOf(
                [
                    A.HorizontalFlip(p=p_hflip),
                    A.RandomRotate90(p=p_random_rotate),
                    A.VerticalFlip(p=p_vflip),
                    A.Blur(p=p_blur),
                    A.ColorJitter(p=p_jitter),
                    A.GaussNoise(p=p_gauss_noise),
                    A.Cutout(p=p_cutout),
                    A.RandomBrightnessContrast(p=p_random_brightness),
                    A.RandomSunFlare(p=p_random_sun_flare),
                ],
                p=p_transform,
            )
        ]
    )
    return transform


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


class DeepLabModelWrapper(nn.Module):
    def __init__(
        self, n_classes: int, pretrained: bool, is_feature_extracting: bool
    ):
        super().__init__()
        self.model = get_deeplabv3(
            n_classes,
            pretrained=pretrained,
            is_feature_extracting=is_feature_extracting,
        )

    def forward(self, x):
        with torch.cuda.amp.autocast():
            return self.model.forward(x)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--train-dir", type=str, required=True)
    parser.add_argument("--val-dir", type=str, required=True)
    parser.add_argument("--cache-dir", type=str, required=False, default=None)
    parser.add_argument(
        "--model-name", type=str, required=False, default="deeplab"
    )
    parser.add_argument("--epochs", type=int, required=False, default=10)
    parser.add_argument("--batch-size", type=int, required=False, default=2)
    parser.add_argument(
        "--learning-rate", type=float, required=False, default=0.001
    )
    parser.add_argument(
        "--aux-loss-weight", type=float, required=False, default=0.4
    )
    parser.add_argument(
        "--patch-strategy",
        type=str,
        required=False,
        default="deterministic_center_crop",
    )
    parser.add_argument(
        "--val-patch-strategy",
        type=str,
        required=False,
        default="",
    )
    parser.add_argument("--toy", type=bool, required=False, default=False)
    parser.add_argument("--classes", type=str, default="1, 2, 3, 4")
    parser.add_argument(
        "--log-file", type=str, required=False, default="train.log"
    )
    parser.add_argument("--p-hflip", type=float, required=False, default=0.5)
    parser.add_argument(
        "--p-random_rotate", type=float, required=False, default=0.5
    )
    parser.add_argument("--p-vflip", type=float, required=False, default=0.5)
    parser.add_argument("--p-blur", type=float, required=False, default=0.5)
    parser.add_argument("--p-jitter", type=float, required=False, default=0.5)
    parser.add_argument(
        "--p-gauss_noise", type=float, required=False, default=0.5
    )
    parser.add_argument(
        "--p-random_brightness", type=float, required=False, default=0.5
    )
    parser.add_argument(
        "--p-random_sun_flare", type=float, required=False, default=0.5
    )
    parser.add_argument("--p-cutout", type=float, required=False, default=0.5)
    parser.add_argument(
        "--p-random-sun-flare", type=float, required=False, default=0.5
    )
    parser.add_argument(
        "--p-random-brightness", type=float, required=False, default=0.5
    )
    parser.add_argument("--p-transform", type=float, required=False, default=1)

    parser.add_argument(
        "--batch-validation-perc", type=float, required=False, default=1.0
    )
    parser.add_argument("--patch-dim", type=str, default="512, 512")
    parser.add_argument("--resize-dim", type=str, default="3632, 5456")
    parser.add_argument(
        "--pretrained", required=False, type=str2bool, default=True
    )
    parser.add_argument(
        "--iou-thresholds", type=str, required=False, default="0.5, 0.3"
    )
    parser.add_argument(
        "--class-balance", type=str2bool, required=False, default=False
    )
    parser.add_argument(
        "--cache-strategy", type=str, required=False, default="memory"
    )
    args = parser.parse_args()

    fh = logging.FileHandler(str(args.log_file))
    log.addHandler(fh)

    train_dir = str(args.train_dir)
    val_dir = str(args.val_dir)
    experiment_dir = str(uuid.uuid4())
    model_dir = join(train_dir, experiment_dir)
    Path(model_dir).mkdir(parents=True, exist_ok=True)

    if args.cache_dir is not None:
        cache_dir = str(args.cache_dir)
    else:
        cache_dir = join("/tmp", str(uuid.uuid4()))
    cache_strategy = str(args.cache_strategy)

    model_name = str(args.model_name)
    n_epochs = int(args.epochs)
    batch_size = int(args.batch_size)
    learning_rate = float(args.learning_rate)
    aux_loss_weight = float(args.aux_loss_weight)
    patch_strategy = str(args.patch_strategy).lower()
    val_patch_strategy = str(args.val_patch_strategy).lower()
    is_toy = bool(args.toy)

    classes = [int(c) for c in args.classes.split(",")]
    class_balance = bool(args.class_balance)

    batch_validation_perc = float(args.batch_validation_perc)
    pretrained: bool = bool(args.pretrained)

    patch_dim: Tuple[int, int] = tuple(
        [int(x) for x in args.patch_dim.split(",")]
    )
    resize_dim: Tuple[int, int] = tuple(
        [int(x) for x in args.resize_dim.split(",")]
    )
    iou_thresholds = [float(x) for x in args.iou_thresholds.split(",")]

    p_hflip = float(args.p_hflip)
    p_random_rotate = float(args.p_random_rotate)
    p_vflip = float(args.p_vflip)
    p_blur = float(args.p_blur)
    p_jitter = float(args.p_jitter)
    p_gauss_noise = float(args.p_gauss_noise)
    p_cutout = float(args.p_cutout)
    p_random_brightness = float(args.p_random_brightness)
    p_random_sun_flare = float(args.p_random_sun_flare)
    p_transform = float(args.p_transform)

    # train on the GPU or on the CPU, if a GPU is not available
    device = (
        torch.device("cuda")
        if torch.cuda.is_available()
        else torch.device("cpu")
    )

    log.info(f"Model Name: {model_name}")
    log.info(f"Epochs: {n_epochs}")
    log.info(f"Learning Rate: {learning_rate}")
    log.info(f"Auxiliary Loss Weight: {aux_loss_weight}")
    log.info(f"Batch Size: {batch_size}")
    log.info(f"Patch Strategy: {patch_strategy}")
    log.info(f"Classes:  {classes}")
    log.info(f"Toy: {is_toy}")
    log.info(f"GPU: {torch.cuda.is_available()}")
    log.info(f"Patch Dimension: {patch_dim}")
    log.info(f"Resize Dimension: {resize_dim}")
    log.info(f"Pretrained: {pretrained}")

    train_labels_filepath = join(train_dir, "train.json")
    val_labels_filepath = join(val_dir, "val.json")

    preprocessing = get_preprocessing(
        p_hflip,
        p_random_rotate,
        p_vflip,
        p_blur,
        p_jitter,
        p_gauss_noise,
        p_cutout,
        p_random_brightness,
        p_random_sun_flare,
        p_transform,
    )
    # Toy Dataset for Integration Testing Purposes
    Dataset = (
        SemanticSegmentationDataset
        if not is_toy
        else ToySemanticSegmentationDataset
    )

    # Validation patch strategy may differ from train patch strategy
    if val_patch_strategy == "":
        if patch_strategy == "resize":
            val_patch_strategy = "resize"
        else:
            val_patch_strategy = "crop_all"

    if patch_strategy == "stochastic":
        dataset = SemanticSegmentationStochasticPatchingDataset(
            f"{train_dir}/patch/*.png",
            f"{train_dir}/mask",
            augmentation=preprocessing,
        )
        dataset_val = SemanticSegmentationStochasticPatchingDataset(
            f"{val_dir}/patch/*.png",
            f"{val_dir}/mask",
            augmentation=preprocessing,
        )
    else:
        if patch_strategy == "resize":
            augmentation = get_augmentation_for_erosion()
        else:
            # TODO: Potential different augmentation strategy for non-resize
            augmentation = get_augmentation_for_erosion()

        dataset = Dataset(
            labels_filepath=train_labels_filepath,
            classes=classes,
            annotation_format="coco",
            root_dir=train_dir,
            cache_dir=join(cache_dir, "train"),
            cache_strategy=cache_strategy,
            # preprocessing=get_preprocessing(),
            augmentation=augmentation,
            patch_strategy=patch_strategy,
            patch_dim=patch_dim,
            resize_dim=resize_dim,
        )
        dataset_val = Dataset(
            labels_filepath=val_labels_filepath,
            classes=classes,
            annotation_format="coco",
            root_dir=val_dir,
            cache_dir=join(cache_dir, "val"),
            cache_strategy=cache_strategy,
            # Specified as augmentation because it's not guaranteed to target
            # the correct instances
            augmentation=get_validation_preprocessing(),
            patch_strategy=val_patch_strategy,
            patch_dim=patch_dim,
            resize_dim=resize_dim,
        )

    dataset_len = len(dataset)
    dataset_val_len = len(dataset_val)
    tot_training_batches = dataset_len // batch_size
    tot_validation_batches = dataset_val_len // batch_size

    print(
        f"Train dataset number of images: {dataset_len} | Batch size: {batch_size} | Expected number of batches: {tot_training_batches}"
    )
    print(
        f"Validation dataset number of images: {dataset_val_len} | Batch size: {batch_size} | Expected number of batches: {tot_validation_batches}"
    )

    num_classes: int = classes[-1] + 1  # Plus 1 for background
    classes = [class_id_to_class_name[i] for i in range(num_classes)]

    # define training and validation data loaders
    # drop_last True to avoid single instances which throw an error on batch norm layers

    # Maxing the num_workers at 8 due to shared memory limitations
    num_workers = min(int(round(multiprocessing.cpu_count() * 2 / 3)), 8)

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=True,
        pin_memory=True,
    )
    dataloader_val = DataLoader(
        dataset_val,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        drop_last=True,
        pin_memory=True,
    )

    # get the model using our helper function
    if model_name == "fcn":
        model = get_fcn_resnet50(num_classes, pretrained=pretrained)
    elif model_name == "deeplab":
        model = DeepLabModelWrapper(
            num_classes, pretrained=pretrained
        )  # get_deeplabv3(num_classes, is_feature_extracting=pretrained)
    else:
        raise ValueError(
            f'Provided model name "{model_name}" is not supported.'
        )

    model = torch.nn.DataParallel(model)
    # move model to the right device
    model.to(device)

    # Create balanced cross entropy loss
    if class_balance:
        weights = semantic_segmentation_class_balancer(dataset)
        weights = weights.to(device)
        criterion = nn.CrossEntropyLoss(weight=weights)
    else:
        criterion = nn.CrossEntropyLoss()
    criterion = criterion.to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(
        params, lr=learning_rate, momentum=0.9, weight_decay=0.0005
    )
    lr_scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=3, gamma=0.1
    )

    metrics = get_semantic_segmentation_metrics(
        num_classes, thresholds=iou_thresholds
    )
    metrics = metrics.to(device)
    best_mean_iou = 0

    best_model_wts: Dict = copy.deepcopy(model.state_dict())

    scaler = torch.cuda.amp.GradScaler()

    for epoch in range(n_epochs):
        start = time.time()
        train_loss = 0
        val_loss = 0

        # Switch to train mode for training
        model.train()

        for batch_num, (images, targets) in enumerate(dataloader, 0):
            batch_time = time.time()
            images: torch.Tensor = images.to(device).float()
            targets: torch.Tensor = targets.to(device).long()

            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                out = model(images)

                outputs = out["out"]
                loss = criterion(outputs, targets)

                # FCN Model w/o pre-training does not have an auxiliary loss component
                # so we avoid this calculation
                if not (not pretrained and model_name == "fcn"):
                    aux_outputs = out["aux"]
                    aux_loss = criterion(aux_outputs, targets)
                    loss = loss + aux_loss_weight * aux_loss

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            preds = torch.argmax(outputs, dim=1)

            train_loss += loss.item() * images.size(0)

            metrics(preds, targets)

            print(
                f"Train Epoch: {epoch} | Batch: {batch_num} | Batch Loss: {loss.item()} | Batch Time: {time.time() - batch_time}"
            )

        train_loss /= len(dataloader.dataset)
        print(f"Epoch: {epoch} | Train Loss: {train_loss}")

        # Compute and log training metrics
        results = metrics.compute()
        results["loss"] = train_loss
        log_metrics(results, classes, split="train")
        metrics.reset()

        # Switch to eval mode for validation
        model.eval()

        if epoch < n_epochs - 1 and batch_validation_perc < 1.0:
            max_batch_num = int(tot_validation_batches * batch_validation_perc)
        else:
            max_batch_num = -1

        with torch.no_grad():
            for batch_num, (images, targets) in enumerate(dataloader_val, 0):
                if max_batch_num == -1 or batch_num < max_batch_num:
                    images: torch.Tensor = images.to(device).float()
                    targets: torch.Tensor = targets.to(device).long()

                    with torch.cuda.amp.autocast():
                        outputs = model(images)["out"]
                        loss = criterion(outputs, targets)
                    preds = torch.argmax(outputs, dim=1)

                    val_loss += loss.item() * images.size(0)
                    metrics(preds, targets)

                    print(
                        f"Validation Epoch: {epoch} | Batch {batch_num} | Batch Loss: {loss.item()}"
                    )

        val_loss /= len(dataloader_val.dataset)
        print(f"Epoch: {epoch} | Val Loss: {val_loss}")

        # Compute and log validation metrics
        results = metrics.compute()
        results["loss"] = val_loss
        log_metrics(results, classes, split="val")
        metrics.reset()

        mean_iou = float(results["mean_iou_0_5"])
        if mean_iou > best_mean_iou:
            best_mean_iou = mean_iou
            best_model_wts = copy.deepcopy(model.state_dict())
            torch.save(
                model.state_dict(),
                join(model_dir, f"{model_name}_checkpoint_{epoch}.pth"),
            )

    model.load_state_dict(best_model_wts)
    torch.save(model.state_dict(), join(model_dir, f"{model_name}_final.pth"))
