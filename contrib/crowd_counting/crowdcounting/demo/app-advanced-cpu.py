import os
import sys
from crowdcounting import CrowdCountModelPose, CrowdCountModelMCNN, Router

from flask import (
    Flask,
    Response,
    json,
    jsonify,
    render_template,
    request,
    send_file,
    send_from_directory,
)
import numpy as np
import logging
import time
import argparse
import cv2
import urllib
import base64
from io import BytesIO

# logging
# logging.basicConfig()
# logger = logging.getLogger()
# logger.setLevel(10)

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
handler = logging.StreamHandler()
handler.setLevel(logging.DEBUG)
formatter = logging.Formatter("[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)

parser = argparse.ArgumentParser(description="A demo app.")
parser.add_argument("-p", "--path", help="Path to MCNN model file", required=True)
args = parser.parse_args()

# flask
app = Flask(__name__)

APP_ROOT = os.path.dirname(os.path.abspath(__file__))

# default images
target = os.path.join(APP_ROOT, "images/")
image_names = os.listdir(target)
image_names.sort()

actual_dict = {"1.jpg": 3, "2.jpg": 60, "3.jpg": 7}
actual_counts = [
    "Actual count: " + str(actual_dict[image_name]) for image_name in image_names
]

gpu_id = -1
mcnn_model_path = args.path  # "./data/models/mcnn_shtechA_660.h5"

model = CrowdCountModelPose(gpu_id)
# model = CrowdCountModelMCNN(gpu_id, model_path=mcnn_model_path)
# model = Router(gpu_id, mcnn_model_path=mcnn_model_path, cutoff_pose=20, cutoff_mcnn=50)

@app.route("/", methods=["GET"])
def load():
    return render_template(
        "index.html", image_names=image_names, actual_counts=actual_counts
    )


@app.route("/upload/<filename>")
def send_image(filename):
    return send_from_directory("images", filename)


@app.route("/uploadfile", methods=["POST"])
def use_upload_file():
    uploaded_file = request.files["file"]
    request_data = uploaded_file.read()

    # score
    result = model.score(request_data, return_image=True, img_dim=1750)

    pred = result["pred"]
    scored_image = result["image"]

    txt = "Predicted count: {0}".format(pred)
    logger.info("use uploaded file")
    return render_template("result.html", scored_image=scored_image, txt=txt)


@app.route("/sitefile", methods=["POST"])
def use_site_file():
    target = os.path.join(APP_ROOT, "images")
    result = request.form["fileindex"]

    local_image = "/".join([target, result])

    with open(local_image, "rb") as f:
        file_bytes = f.read()

    # score
    result = model.score(file_bytes, return_image=True, img_dim=1750)

    pred = result["pred"]
    scored_image = result["image"]

    txt = "Predicted count: {0}".format(pred)

    return render_template("result.html", scored_image=scored_image, txt=txt)


@app.route("/score", methods=["POST"])
def score():

    result = model.score(request.data, return_image=False, img_dim=1750)

    js = json.dumps({"count": int(np.round(result["pred"]))})
    resp = Response(js, status=200, mimetype="application/json")
    return resp


@app.route("/score_alt", methods=["POST"])
def score_alt():

    result = model.score(request.data, return_image=True, img_dim=1750)

    t = urllib.parse.unquote(result["image"])
    image = base64.b64decode(t)

    return send_file(
        BytesIO(image),
        as_attachment=True,
        attachment_filename="pred.png",
        mimetype="image/png",
    )


if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=5000, threaded=True)
