# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

# NOTE: This file is used by pytest to inject fixtures automatically. As it is explained in the documentation
# https://docs.pytest.org/en/latest/fixture.html:
# "If during implementing your tests you realize that you want to use a fixture function from multiple test files
# you can move it to a conftest.py file. You don't need to import the fixture you want to use in a test, it
# automatically gets discovered by pytest."

coco_sample = """
{
    "info": {
        "year": "2020",
        "version": "1",
        "description": "",
        "contributor": "",
        "url": "https://url",
        "date_created": "2020-01-01T00:00:00+00:00"
    },
    "licenses": [
        {
            "id": 1,
            "url": "",
            "name": "name"
        }
    ],
    "categories": [
        {
            "id": 0,
            "name": "bottle",
            "supercategory": "none"
        },
        {
            "id": 1,
            "name": "can",
            "supercategory": "cells"
        },
        {
            "id": 2,
            "name": "carton",
            "supercategory": "cells"
        }
    ],
    "images": [
        {
            "id": 0,
            "license": 1,
            "file_name": "image0.jpg",
            "height": 480,
            "width": 640,
            "date_captured": "2020-01-01T00:00:00+00:00",
			"coco_url": "http://image0.jpg"
        },
        {
            "id": 1,
            "license": 1,
            "file_name": "image1.jpg",
            "height": 480,
            "width": 640,
            "date_captured": "2020-01-01T00:00:00+00:00",
			"url": "http://image1.jpg"
        },
        {
            "id": 2,
            "license": 1,
            "file_name": "image2.jpg",
            "height": 480,
            "width": 640,
            "date_captured": "2020-01-01T00:00:00+00:00"
        }
    ],
    "annotations": [
        {
            "id": 0,
            "image_id": 0,
            "category_id": 2,
            "bbox": [
                100,
                200,
                300,
                400
            ],
            "area": 10000,
            "segmentation": [],
            "iscrowd": 0
        },
        {
            "id": 1,
            "image_id": 0,
            "category_id": 1,
            "bbox": [
                100,
                200,
                300,
                400
            ],
            "area": 10000,
            "segmentation": [],
            "iscrowd": 0
        },
        {
            "id": 2,
            "image_id": 1,
            "category_id": 1,
            "bbox": [
                100,
                200,
                300,
                400
            ],
            "area": 10000,
            "segmentation": [],
            "iscrowd": 0
        },
        {
            "id": 3,
            "image_id": 2,
            "category_id": 1,
            "bbox": [
                100,
                200,
                300,
                400
            ],
            "area": 10000,
            "segmentation": [],
            "iscrowd": 0
        }
    ]
}
"""
