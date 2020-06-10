# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
import requests
from utils_cv.classification.data import Urls


def test_urls():
    # Test if all urls are valid
    all_urls = Urls.all()
    for url in all_urls:
        with requests.get(url):
            pass
