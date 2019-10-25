# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import pytest
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

from utils_cv.detection.plot import (
    PlotSettings,
    plot_boxes,
    display_bboxes,
    plot_grid,
    plot_detection_vs_ground_truth,
    _setup_pr_axes,
    _get_precision_recall_settings,
    _plot_pr_curve_iou_range,
    _plot_pr_curve_iou_mean,
    plot_pr_curves,
    plot_mask,
    display_image,
    display_bbox_mask,
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


def test_plot_boxes(od_cup_path, od_cup_anno_bboxes, basic_plot_settings):
    """ Test that `plot_boxes` doesn't throw error. """
    im = Image.open(od_cup_path).convert("RGB")

    # basic case
    plot_boxes(im=im, bboxes=od_cup_anno_bboxes)

    # with update plot_settings
    plot_boxes(
        im=im, bboxes=od_cup_anno_bboxes, plot_settings=basic_plot_settings
    )


def test_plot_mask(od_mask_rects):
    """ Test that `plot_mask` works. """
    plot_setting = PlotSettings()
    _, mask, rects, im = od_mask_rects
    # plot mask
    im = plot_mask(im, mask, plot_settings=plot_setting).convert('RGB')
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


def test_display_image(od_cup_path, basic_ax):
    """ Test that `display_image` works. """
    # test image path
    display_image(od_cup_path, basic_ax)
    # test PIL.Image
    display_image(Image.open(od_cup_path), basic_ax)
    # test numpy array
    display_image(np.array(Image.open(od_cup_path)), basic_ax)


def test_display_bboxes(od_cup_anno_bboxes, od_cup_path):
    """ Test that `display_bboxes` works. """
    display_bboxes(bboxes=od_cup_anno_bboxes, im_path=od_cup_path)


def test_display_bbox_mask(
    od_cup_anno_bboxes,
    od_cup_path,
    od_cup_mask_path,
    basic_ax
):
    """ Test that `display_bbox_mask` works. """
    display_bbox_mask(
        bboxes=od_cup_anno_bboxes,
        im_path=od_cup_path,
        mask_path=od_cup_mask_path,
        ax=basic_ax
    )


def test_plot_grid(od_cup_anno_bboxes, od_cup_path):
    """ Test that `plot_grid` works. """

    # test callable args
    def callable_args():
        return od_cup_anno_bboxes, od_cup_path

    plot_grid(display_bboxes, callable_args, rows=1)

    # test iterable args
    od_cup_paths = [od_cup_path, od_cup_path, od_cup_path]
    od_cup_annos = [od_cup_anno_bboxes, od_cup_anno_bboxes, od_cup_anno_bboxes]

    def iterator_args():
        for path, bboxes in zip(od_cup_paths, od_cup_annos):
            yield bboxes, path

    plot_grid(display_bboxes, iterator_args(), rows=1)


def test_plot_detection_vs_ground_truth(
    od_cup_path, od_cup_det_bboxes, od_cup_anno_bboxes, basic_ax
):
    """ Test that `plot_detection_vs_ground_truth` works. """
    plot_detection_vs_ground_truth(
        od_cup_path, od_cup_det_bboxes, od_cup_anno_bboxes, ax=basic_ax
    )


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
