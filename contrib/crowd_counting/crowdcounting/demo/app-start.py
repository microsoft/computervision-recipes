import os
import sys
from crowdcounting import CrowdCountModelPose, CrowdCountModelMCNN, Router
from flask import Flask, Response, json, request
import numpy as np
import logging
import time
import argparse

# logging
logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(10)

parser = argparse.ArgumentParser(description="A demo app.")
parser.add_argument("-p", "--path", help="Path to MCNN model file", required=True)
args = parser.parse_args()

# flask
app = Flask(__name__)

gpu_id = 0
mcnn_model_path = args.path  # "./data/models/mcnn_shtechA_660.h5"

model = Router(gpu_id, mcnn_model_path=mcnn_model_path, cutoff_pose=20, cutoff_mcnn=50)


@app.route("/score", methods=["POST"])
def score():

    result = model.score(request.data, return_image=False, img_dim=1750)

    js = json.dumps({"count": int(np.round(result["pred"]))})
    resp = Response(js, status=200, mimetype="application/json")
    return resp


if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=5000, threaded=True)
