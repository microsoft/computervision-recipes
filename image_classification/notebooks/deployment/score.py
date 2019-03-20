# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license.

from base64 import b64encode, b64decode
from io import BytesIO
import json
import numpy as np
from azureml.core.model import Model
from fastai.vision import *
from fastai.vision import Image as FImage

def init():
    global model
    model_path = Model.get_model_path(model_name='im_classif_fridge_obj')
    # ! We cannot use MODEL_NAME here otherwise the execution on Azure will fail !
    
    actual_path, actual_file = os.path.split(model_path)
    model = load_learner(path=actual_path, fname=actual_file)


def run(raw_data):

    # Expects raw_data to be a list within a json file
    result = []
    all_data = [b64decode(im) for im in json.loads(raw_data)['data']]
    
    for im_bytes in all_data:
        try:
            im = open_image(BytesIO(im_bytes))
            pred_class, pred_idx, outputs = model.predict(im)
            result_dict = {"label": str(pred_class), "probability": str(float(outputs[pred_idx]))}
            result.append(result_dict)
        except Exception as e:
            result_dict = {"label": str(e), "probability": ''}
            result.append(result_dict)
    return result
