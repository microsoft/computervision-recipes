# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import pytest
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

from utils_cv.detection.plot import (
    PlotSettings,
    plot_boxes,
    plot_boxes_stats,
    plot_grid,
    plot_detections,
    _setup_pr_axes,
    _get_precision_recall_settings,
    _plot_pr_curve_iou_range,
    _plot_pr_curve_iou_mean,
    plot_pr_curves,
    plot_counts_curves,
    plot_masks,
    plot_keypoints,
)


@pytest.fixture(scope="session")
def basic_plot_settings() -> PlotSettings:
    return PlotSettings()


@pytest.fixture(scope="session")
def basic_ax() -> plt.Axes:
    fig = plt.figure()
    return fig.add_subplot(111)


def test_plot_setting_init(basic_plot_settings):
    assert basic_plot_settings.rect_th is not None
    assert basic_plot_settings.rect_color is not None
    assert basic_plot_settings.text_size is not None
    assert basic_plot_settings.text_color is not None
    assert basic_plot_settings.mask_color is not None
    assert basic_plot_settings.mask_alpha is not None
    assert basic_plot_settings.keypoint_th is not None
    assert basic_plot_settings.keypoint_color is not None


def test_plot_boxes_stats(tiny_od_detection_keypoint_dataset):
    # simply test that this is error free
    plot_boxes_stats(tiny_od_detection_keypoint_dataset)


def test_plot_boxes(od_cup_path, od_cup_anno_bboxes, basic_plot_settings):
    """ Test that `plot_boxes` doesn't throw error. """
    im = Image.open(od_cup_path).convert("RGB")

    # basic case
    plot_boxes(im=im, bboxes=od_cup_anno_bboxes)

    # with update plot_settings
    plot_boxes(
        im=im, bboxes=od_cup_anno_bboxes, plot_settings=basic_plot_settings
    )


def test_plot_masks(od_mask_rects):
    """ Test that `plot_mask` works. """
    plot_setting = PlotSettings(mask_color=(10, 20, 128))
    _, mask, rects, im = od_mask_rects

    # plot mask
    im = plot_masks(im, mask, plot_settings=plot_setting).convert("RGB")
    im = np.transpose(np.array(im), (2, 0, 1))

    # validate each channel matches the mask
    for ch in im:
        ch_uniques = np.unique(ch)
        foreground_uniques = np.unique(ch[np.where(mask != 0)])
        assert len(foreground_uniques) == 1
        assert foreground_uniques[0] == ch_uniques[1]
        background_uniques = np.unique(ch[np.where(mask == 0)])
        assert len(background_uniques) == 1
        assert background_uniques[0] == ch_uniques[0]


def test_plot_keypoints(basic_plot_settings):
    # a completely black image
    im = Image.fromarray(np.zeros((500, 600, 3), dtype=np.uint8))

    # dummy keypoints
    keypoints = np.array([[[100, 200, 2], [200, 200, 2]]])
    keypoint_meta = {"skeleton": [[0, 1]]}

    # basic case
    plot_keypoints(im, keypoints, keypoint_meta)

    # with update plot_settings
    plot_keypoints(
        im, keypoints, keypoint_meta, plot_settings=basic_plot_settings
    )


def test_plot_detections(
    od_sample_detection,
    od_detection_mask_dataset,
    od_sample_keypoint_detection,
    tiny_od_detection_keypoint_dataset,
):
    plot_detections(od_sample_detection)
    plot_detections(od_sample_detection, od_detection_mask_dataset)
    plot_detections(od_sample_detection, od_detection_mask_dataset, 0)

    # plot keypoints
    plot_detections(
        od_sample_keypoint_detection,
        keypoint_meta=tiny_od_detection_keypoint_dataset.keypoint_meta,
    )
    plot_detections(
        od_sample_keypoint_detection,
        tiny_od_detection_keypoint_dataset,
        keypoint_meta=tiny_od_detection_keypoint_dataset.keypoint_meta,
    )
    plot_detections(
        od_sample_keypoint_detection,
        tiny_od_detection_keypoint_dataset,
        0,
        keypoint_meta=tiny_od_detection_keypoint_dataset.keypoint_meta,
    )


def test_plot_grid(
    od_sample_detection,
    od_detection_mask_dataset,
    od_sample_keypoint_detection,
    tiny_od_detection_keypoint_dataset,
):
    """ Test that `plot_grid` works. """

    # test callable args
    def callable_args():
        return od_sample_detection, None, None, None

    plot_grid(plot_detections, callable_args, rows=1)

    def callable_args():
        return od_sample_detection, od_detection_mask_dataset, None, None

    plot_grid(plot_detections, callable_args, rows=1)

    def callable_args():
        return (
            od_sample_keypoint_detection,
            tiny_od_detection_keypoint_dataset,
            None,
            tiny_od_detection_keypoint_dataset.keypoint_meta,
        )

    plot_grid(plot_detections, callable_args, rows=1)

    # test iterable args
    def iterator_args():
        for detection in [od_sample_detection, od_sample_detection]:
            yield detection, None, None, None

    plot_grid(plot_detections, iterator_args(), rows=1, cols=2)

    def iterator_args():
        for detection in [od_sample_detection, od_sample_detection]:
            yield detection, od_detection_mask_dataset, None, None

    plot_grid(plot_detections, iterator_args(), rows=1, cols=2)

    def iterator_args():
        for detection in [
            od_sample_keypoint_detection,
            od_sample_keypoint_detection,
        ]:
            yield (
                detection,
                tiny_od_detection_keypoint_dataset,
                None,
                tiny_od_detection_keypoint_dataset.keypoint_meta,
            )

    plot_grid(plot_detections, iterator_args(), rows=1, cols=2)


def test__setup_pr_axes(basic_ax):
    """ Test that `_setup_pr_axes` works. """
    _setup_pr_axes(basic_ax, "dummy_title")


def test__get_precision_recall_settings():
    """ Test that `_get_precision_recall_settings` works. """
    ret = _get_precision_recall_settings(1)
    assert len(ret) == 5
    ret = _get_precision_recall_settings(slice(0, 2))
    assert len(ret) == 5


@pytest.mark.gpu
def test__plot_pr_curve_iou_range(od_detection_eval, basic_ax):
    """ Test that `_plot_pr_curve_iou_range` works. """
    _plot_pr_curve_iou_range(basic_ax, od_detection_eval.coco_eval["bbox"])


@pytest.mark.gpu
def test__plot_pr_curve_iou_mean(od_detection_eval, basic_ax):
    """ Test that `_plot_pr_curve_iou_mean` works. """
    _plot_pr_curve_iou_mean(basic_ax, od_detection_eval.coco_eval["bbox"])


@pytest.mark.gpu
def test_plot_pr_curves(od_detection_eval, od_detection_mask_eval):
    """ Test that `plot_pr_curves` works. """
    plot_pr_curves(od_detection_eval)
    plot_pr_curves(od_detection_mask_eval)


@pytest.mark.gpu
def test_plot_counts_curves(od_detection_dataset, od_detections):
    """ Test that `plot_counts_curves` works. """
    plot_counts_curves(
        od_detections, od_detection_dataset.test_ds, od_detections
    )
