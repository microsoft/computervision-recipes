# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

from typing import List, Union
from urllib.parse import urljoin
from PIL import Image
from pathlib import Path

import json
import numpy as np
import shutil
import urllib.request
import xml.etree.ElementTree as ET

from .references.anno_coco2voc import coco2voc_main


class Urls:
    # for now hardcoding base url into Urls class
    base = "https://cvbp-secondary.z19.web.core.windows.net/datasets/object_detection/"

    # traditional datasets
    fridge_objects_path = urljoin(base, "odFridgeObjects.zip")
    fridge_objects_watermark_path = urljoin(
        base, "odFridgeObjectsWatermark.zip"
    )
    fridge_objects_tiny_path = urljoin(base, "odFridgeObjectsTiny.zip")
    fridge_objects_watermark_tiny_path = urljoin(
        base, "odFridgeObjectsWatermarkTiny.zip"
    )

    # mask datasets
    fridge_objects_mask_path = urljoin(base, "odFridgeObjectsMask.zip")
    fridge_objects_mask_tiny_path = urljoin(
        base, "odFridgeObjectsMaskTiny.zip"
    )

    # keypoint datasets
    fridge_objects_keypoint_milk_bottle_path = urljoin(
        base, "odFridgeObjectsMilkbottleKeypoint.zip"
    )
    fridge_objects_keypoint_milk_bottle_tiny_path = urljoin(
        base, "odFridgeObjectsMilkbottleKeypointTiny.zip"
    )
    fridge_objects_keypoint_top_bottom_path = urljoin(
        base, "odFridgeObjectsKeypointTopBottom.zip"
    )
    fridge_objects_keypoint_top_bottom_tiny_path = urljoin(
        base, "odFridgeObjectsKeypointTopBottomTiny.zip"
    )

    @classmethod
    def all(cls) -> List[str]:
        return [v for k, v in cls.__dict__.items() if k.endswith("_path")]


def coco_labels() -> List[str]:
    """ List of Coco labels with the original indexing.

    Reference: https://github.com/pytorch/vision/blob/master/docs/source/models.rst

    Returns:
        Coco labels
    """

    return [
        "__background__",
        "person",
        "bicycle",
        "car",
        "motorcycle",
        "airplane",
        "bus",
        "train",
        "truck",
        "boat",
        "traffic light",
        "fire hydrant",
        "N/A",
        "stop sign",
        "parking meter",
        "bench",
        "bird",
        "cat",
        "dog",
        "horse",
        "sheep",
        "cow",
        "elephant",
        "bear",
        "zebra",
        "giraffe",
        "N/A",
        "backpack",
        "umbrella",
        "N/A",
        "N/A",
        "handbag",
        "tie",
        "suitcase",
        "frisbee",
        "skis",
        "snowboard",
        "sports ball",
        "kite",
        "baseball bat",
        "baseball glove",
        "skateboard",
        "surfboard",
        "tennis racket",
        "bottle",
        "N/A",
        "wine glass",
        "cup",
        "fork",
        "knife",
        "spoon",
        "bowl",
        "banana",
        "apple",
        "sandwich",
        "orange",
        "broccoli",
        "carrot",
        "hot dog",
        "pizza",
        "donut",
        "cake",
        "chair",
        "couch",
        "potted plant",
        "bed",
        "N/A",
        "dining table",
        "N/A",
        "N/A",
        "toilet",
        "N/A",
        "tv",
        "laptop",
        "mouse",
        "remote",
        "keyboard",
        "cell phone",
        "microwave",
        "oven",
        "toaster",
        "sink",
        "refrigerator",
        "N/A",
        "book",
        "clock",
        "vase",
        "scissors",
        "teddy bear",
        "hair drier",
        "toothbrush",
    ]


def coco2voc(
    anno_path: str,
    output_dir: str,
    anno_type: str = "instance",
    download_images: bool = False,
) -> None:
    """ Convert COCO annotation (single .json file) to Pascal VOC annotations
        (multiple .xml files).

    Args:
        anno_path: path to coco-formated .json annotation file
        output_dir: root output directory
        anno_type: "instance" for rectangle annotation, or "keypoint" for keypoint annotation.
        download_images: if true then download images from their urls.
    """
    coco2voc_main(anno_path, output_dir, anno_type, download_images)


def extract_masks_from_labelbox_json(
    labelbox_json_path: Union[str, Path],
    data_dir: Union[str, Path],
    mask_data_dir: Union[str, Path] = None,
) -> None:
    """ Extract masks from Labelbox annotation JSON file.

    It reads in an annotation JSON file created by the Labelbox annotation UI
    (https://labelbox.com/), downloads the binary segmentation masks for all
    objects, merges them in the order of the bounding boxes described in the
    corresponding PASCAL VOC annotation file, and then writes the resultant
    mask into a directory called "segmentation-masks".

    The annotation files in
    [odFridgeObjects](https://cvbp-secondary.z19.web.core.windows.net/datasets/object_detection/odFridgeObjects.zip)
    are in the format of PASCAL VOC shown in our
    [01 notebook](../../scenarios/detection/01_training_introduction.ipynb).

    The data structure of the export JSON file from Labelbox looks like:

    ```
    {"Dataset Name": "odFridgeObjects",
     "External ID": "117.jpg",
     "Label": {"objects": [{"color": "#00D4FF",
                            "featureId": "ck1iu6m3suwmo0944zoufayto",
                            "instanceURI": "https://api.labelbox.com/masks/ck1iphg4xsqhe0944bbbiwrak",
                            "schemaId": "ck1ipz4v5s5rd0701j2mfc4ii",
                            "title": "water_bottle",
                            "value": "water_bottle"},
                           {"color": "#00FFFF",
                            "featureId": "ck1iuonmvryt608388vlq6t9z",
                            "instanceURI": "https://api.labelbox.com/masks/ck1iphg4xsqhe0944bbbiwrak",
                            "schemaId": "ck1ipz4v5s5re0701sojrveb3",
                            "title": "milk_bottle",
                            "value": "milk_bottle"}]},
     "Labeled Data": "https://storage.labelbox.com/58d748d4418a-117.jpg",
     "View Label": "https://editor.labelbox.com?project=ck1iphg4xsqhe&label=ck1iq31v1qqht086"}
    ```

    It is a list of `Dict` where each `Dict` is the meta data for an
    image.  Key fields include:
    * **`annos[n]["External ID"]`**: Original image file name
    * `annos[n]["Labeled Data"]`: URL of the original image
    * `annos[n]["View Label"]`: URL of the image with labels or masks
    * `annos[n]["Label"]`: Dict.  Meta data of all annotations of the
      image
    * `annos[n]["Label"]["objects"]`: List.  Meta data of all objects of
      the image.
    * `annos[n]["Label"]["objects"][0]["value"]`: Object name (category)
    * **`annos[n]["Label"]["objects"][0]["instanceURI"]`**: URL of the
      binary mask of the object, with 0 as background, 255 as the object.

    Take the
    [`odFridgeObjects`](https://cvbp-secondary.z19.web.core.windows.net/datasets/object_detection/odFridgeObjects.zip)
    dataset as an example.  Here the XML annotations are in the
    `odFridgeObjects/annotations` folder and the original images are in
    the `odFridgeObjects/images` folder.  For an arbitrary image
    `odFridgeObjects/images/xyz.jpg`, its corresponding XML annotation
    file is `odFridgeObjects/annotations/xyz.xml`.

    Because the missing parts are the masks annotated in LabelBox, the
    only thing we need to do is to combine all binary masks
    (`[obj["instanceURI"] for obj in annos[0]["Label"]["objects"]]`) of
    individual objects from an image (`annos[0]["External ID"]`) into a
    single mask image (`annos[0]["External ID"][:-4] + ".png"`) in a
    directory called `segmentation-masks`.

    Args:
        labelbox_json_path: mask annotation JSON file from Labelbox
        data_dir: path to dataset.  The path should contain the "images" and
            "annotations" subdirectories which store the original images and
            PASCAL VOC annotation XML files.
        mask_data_dir: path to the result.  It will contain a
            "segmentation-masks" subdirectory as well as "images" and
            "annotations".  Only images with masks described in labelbox_json_path
             will be stored in mask_data_dir.  Mask images extracted into
             "segmentation-masks" will be PNG files.
    """

    src_im_dir = Path(data_dir) / "images"  # image folder
    src_anno_dir = Path(data_dir) / "annotations"  # annotation folder

    dst_im_dir = Path(mask_data_dir) / "images"
    dst_anno_dir = Path(mask_data_dir) / "annotations"
    dst_mask_dir = Path(mask_data_dir) / "segmentation-masks"  # mask folder

    # create directories for annotated dataset
    dst_im_dir.mkdir(parents=True, exist_ok=True)
    dst_anno_dir.mkdir(parents=True, exist_ok=True)
    dst_mask_dir.mkdir(parents=True, exist_ok=True)

    # read exported LabelBox annotation JSON file
    with open(labelbox_json_path) as f:
        annos = json.load(f)

    # process one image per iteration
    for anno in annos:
        # get related file paths
        im_name = anno["External ID"]  # image file name
        anno_name = im_name[:-4] + ".xml"  # annotation file name
        mask_name = im_name[:-4] + ".png"  # mask file name

        print("Processing image: {}".format(im_name))

        src_im_path = src_im_dir / im_name
        src_anno_path = src_anno_dir / anno_name

        dst_im_path = dst_im_dir / im_name
        dst_anno_path = dst_anno_dir / anno_name
        dst_mask_path = dst_mask_dir / mask_name

        # copy original image and annotation file
        shutil.copy(src_im_path, dst_im_path)
        shutil.copy(src_anno_path, dst_anno_path)

        # read mask images
        mask_urls = [obj["instanceURI"] for obj in anno["Label"]["objects"]]
        labels = [obj["value"] for obj in anno["Label"]["objects"]]
        binary_masks = np.array(
            [
                np.array(Image.open(urllib.request.urlopen(url)))[..., 0]
                == 255
                for url in mask_urls
            ]
        )

        # rearrange masks with regard to annotation
        tree = ET.parse(dst_anno_path)
        root = tree.getroot()
        rects = []
        for obj in root.findall("object"):
            label = obj.find("name").text
            bnd_box = obj.find("bndbox")
            left = int(bnd_box.find("xmin").text)
            top = int(bnd_box.find("ymin").text)
            right = int(bnd_box.find("xmax").text)
            bottom = int(bnd_box.find("ymax").text)
            rects.append((label, left, top, right, bottom))

        assert len(rects) == len(binary_masks)
        matches = []
        # find matched binary mask and annotation
        for label, left, top, right, bottom in rects:
            match = 0
            min_overlap = binary_masks.shape[1] * binary_masks.shape[2]
            for i, bmask in enumerate(binary_masks):
                bmask_out = bmask.copy()
                bmask_out[top : (bottom + 1), left : (right + 1)] = False
                non_overlap = np.sum(bmask_out)
                if non_overlap < min_overlap:
                    match = i
                    min_overlap = non_overlap
            assert label == labels[match], "{}: {}".format(
                label, labels[match]
            )
            matches.append(match)

        assert len(set(matches)) == len(matches), "{}: {}".format(
            len(set(matches)), len(matches)
        )

        binary_masks = binary_masks[matches]

        # merge binary masks
        obj_values = np.arange(len(binary_masks)) + 1
        labeled_masks = binary_masks * obj_values[:, None, None]
        mask = np.max(labeled_masks, axis=0).astype(np.uint8)

        # save mask image
        Image.fromarray(mask, mode="L").save(dst_mask_path)


def extract_keypoints_from_labelbox_json(
    labelbox_json_path: Union[str, Path],
    data_dir: Union[str, Path],
    keypoint_data_dir: Union[str, Path] = None,
) -> None:
    """ Extract keypoints from Labelbox annotation JSON file.

    It reads in an annotation JSON file created by the Labelbox annotation UI
    (https://labelbox.com/), extracts the annotated keypoints for all objects,
    and then writes them into the corresponding PASCAL VOC annotation file.

    The data structure of the export JSON file from Labelbox looks like:

    ```
    {"Dataset Name": "odFridgeObjects",
     "External ID": "21.jpg",
     "Label": {"carton_left_back_bottom": [{"geometry": {"x": 217, "y": 277}}],
               "carton_left_back_shoulder": [{"geometry": {"x": 410, "y": 340}}],
               "carton_left_collar": [{"geometry": {"x": 416, "y": 367}}],
               "carton_left_front_bottom": [{"geometry": {"x": 161, "y": 299}}],
               "carton_left_front_shoulder": [{"geometry": {"x": 359, "y": 375}}],
               "carton_left_top": [{"geometry": {"x": 438, "y": 379}}],
               "carton_lid": [{"geometry": {"x": 392, "y": 427}}],
               "carton_right_collar": [{"geometry": {"x": 398, "y": 450}}],
               "carton_right_front_bottom": [{"geometry": {"x": 166, "y": 371}}],
               "carton_right_front_shoulder": [{"geometry": {"x": 350, "y": 462}}],
               "carton_right_top": [{"geometry": {"x": 424, "y": 455}}],
               "water_bottle_lid_left_bottom": [{"geometry": {"x": 243, "y": 444}}],
               "water_bottle_lid_left_top": [{"geometry": {"x": 266, "y": 456}}],
               "water_bottle_lid_right_bottom": [{"geometry": {"x": 220,
                                                               "y": 499}}],
               "water_bottle_lid_right_top": [{"geometry": {"x": 243, "y": 511}}],
               "water_bottle_wrapper_left_bottom": [{"geometry": {"x": 77,
                                                                  "y": 344}}],
               "water_bottle_wrapper_left_top": [{"geometry": {"x": 161,
                                                               "y": 379}}],
               "water_bottle_wrapper_right_bottom": [{"geometry": {"x": 30,
                                                                   "y": 424}}],
               "water_bottle_wrapper_right_top": [{"geometry": {"x": 120,
                                                                "y": 477}}]},
     "Labeled Data": "https://storage.labelbox.com/ck1ipbufauu4f072105748106f5ce6-21.jpg",
     "View Label": "https://image-segmentation-v4.labelbox.com?project=ck36v24&label=ck36xdrzryw"}
    ```

    It is a list of `Dict` where each `Dict` is the meta data for an
    image.  Key fields include:
    * **`annos[n]["External ID"]`**: Original image file name
    * `annos[n]["Labeled Data"]`: URL of the original image
    * `annos[n]["View Label"]`: URL of the image with labels or masks
    * `annos[n]["Label"]`: Dict.  Meta data of all annotations of the
      image.  Its keys are the labels of keypoints, and its values are the
      coordinates.
    * **`annos[n]["Label"]["xxx"][0]["geometry"]["x"]`**: The x coordinate
      of the label `xxx`.
    * **`annos[n]["Label"]["xxx"][0]["geometry"]["y"]`**: The y coordinate
      of the label `xxx`.

    **NOTE** that things become tricky when there are multiple instances
    of the same category exist in an image.  But for now in the
    odFridgeObjects dataset, no more than one instance of a category
    exists in an image.  In addition, there is no natural way of
    specifying a point belongs to a label in Labelbox.  Therefore, for
    example, we use the prefix `carton_` to indicate the point labeled
    `carton_left_back_bottom` is a point that belongs to a carton.

    Args:
        labelbox_json_path: keypoint annotation JSON file from Labelbox
        data_dir: path to dataset.  The path should contain the "images" and
            "annotations" subdirectories which store the original images and
            PASCAL VOC annotation XML files.
        keypoint_data_dir: path to the result.  It will contain the "images"
            and "annotations" subdirectories.  Only images with keypoints
            described in labelbox_json_path will be stored in keypoint_data_dir.
            The XML files in the "annotations" directory will also include the
            keypoint annotations extracted from Labelbox"s JSON file.
    """

    # original image folder
    src_im_dir = Path(data_dir) / "images"
    # original annotation folder
    src_anno_dir = Path(data_dir) / "annotations"

    # keypoint image folder
    dst_im_dir = Path(keypoint_data_dir) / "images"
    # keypoint annotation folder
    dst_anno_dir = Path(keypoint_data_dir) / "annotations"

    # create directories for annotated dataset
    dst_im_dir.mkdir(parents=True, exist_ok=True)
    dst_anno_dir.mkdir(parents=True, exist_ok=True)

    # read exported LabelBox annotation JSON file
    with open(labelbox_json_path) as f:
        annos = json.load(f)

    # process one image keypoints annotation per iteration
    for anno in annos:
        # get related file paths
        im_name = anno["External ID"]  # image file name
        anno_name = im_name[:-4] + ".xml"  # annotation file name

        print("Processing image: {}".format(im_name))

        src_im_path = src_im_dir / im_name
        src_anno_path = src_anno_dir / anno_name

        dst_im_path = dst_im_dir / im_name
        dst_anno_path = dst_anno_dir / anno_name

        # copy original image
        shutil.copy(src_im_path, dst_im_path)

        # add keypoints annotation into PASCAL VOC XML file
        keypoints_annos = anno["Label"]
        tree = ET.parse(src_anno_path)
        root = tree.getroot()
        for obj in root.findall("object"):
            prefix = obj.find("name").text + "_"
            # add "keypoints" node for current object
            keypoints = ET.SubElement(obj, "keypoints")
            for k in keypoints_annos.keys():
                if k.startswith(prefix):
                    # add keypoint into "keypoints" node
                    pt = ET.SubElement(keypoints, k[len(prefix) :])
                    x = ET.SubElement(pt, "x")  # add x coordinate
                    y = ET.SubElement(pt, "y")  # add y coordinate
                    geo = keypoints_annos[k][0]["geometry"]
                    x.text = str(geo["x"])
                    y.text = str(geo["y"])

        # write modified annotation file
        tree.write(dst_anno_path)
