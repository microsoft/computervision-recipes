import json
import logging
import os
import requests

from base64 import b64encode
from flask import Flask, request, send_from_directory, render_template
from pathlib import Path
from typing import Union
from werkzeug.utils import secure_filename

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

UPLOAD_FOLDER = os.path.join(os.getcwd(), "uploads")
ALLOWED_EXTENSIONS = set(["txt", "pdf", "png", "jpg", "jpeg", "gif"])

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER


def im2base64(im_path: Union[Path, str]) -> bytes:
    """
    Converts an image into a base 64 encoded object
    Args:
        im_path (string): Path to the image

    Returns: im_bytes

    """

    with open(im_path, "rb") as image:
        # Extract image bytes
        im_content = image.read()
        # Convert bytes into a string
        im_bytes = b64encode(im_content)

    return im_bytes


def ims2strlist(im_path_list: list) -> list:
    """
    Converts base 64 encoded objects into strings and appends them to a list

    Args:
        im_path_list (list of strings): List of image paths

    Returns: im_string_list: List containing based64-encoded images
    decoded into strings

    """

    im_string_list = []
    for im_path in im_path_list:
        im_string_list.append(im2base64(im_path).decode("utf-8"))

    return im_string_list


def allowed_file(filename):
    """
    Determines whether the image file considered is legitimate or not

    Args:
        filename: (string) Name of image file processed

    Returns: (boolean) True if file name extension is in ALLOWED_EXTENSIONS

    """

    return (
        "." in filename
        and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS
    )


def predict_from_service(folder, filenames_list):
    """
    Calls the webservice to get the predicted classes
    and probabilities of each passed image

    Args:
        folder: (string) Name of the folder in which the images were uploaded
        filenames_list: (list of strings) List of image file names

    Returns: (list of dictionaries and integer) List of {label, probability}
    and size of that list
    """

    logging.info("Calling the image classification endpoint ...")
    logging.info(f"Images stored in {UPLOAD_FOLDER}")
    im_paths = [os.path.join(folder, im_name) for im_name in filenames_list]
    im_string_list = ims2strlist(im_paths)
    data_for_service = json.dumps({"data": im_string_list})

    # Setting of the authorization header
    # (Authentication is enabled by default when deploying to AKS)
    # key = auth_key
    webservice_url = "<service_uri>"
    key = "<primary_key>"
    headers = {"Content-Type": "application/json"}
    headers["Authorization"] = f"Bearer {key}"

    res = requests.post(webservice_url, data=data_for_service, headers=headers)

    if res.ok:
        # If service succeeds in computing predictions,
        # return them and the number of such predictions
        # (needed for rendering of the results in an HTML table)
        service_results = res.json()
        results_length = len(service_results)
        return service_results, results_length
    else:
        # If service fails to return predictions, raise an error
        error_message = res.reason
        if error_message == "Request Entity Too Large":
            error_message = "{} -- Please select smaller or fewer images".format(
                error_message
            )
        raise ValueError(error_message)


@app.route("/")
def index():
    """
    Displays the "upload image page"

    Returns: HTML rendering
    """
    return render_template("index.html")


@app.route("/", methods=["POST"])
def upload_file():
    """
    Checks that each uploaded image is legitimate and stores
    its file name in a list
    Renders an HTML table with the results if the image file names
    are legitimate, or the upload page if not

    Returns: HTML rendering with images and associated predictions
    """
    uploaded_files = request.files.getlist("file_list")
    filenames = []

    if not os.path.exists(UPLOAD_FOLDER):
        os.mkdir(UPLOAD_FOLDER)

    for file in uploaded_files:
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config["UPLOAD_FOLDER"], filename))
            filenames.append(filename)
    logging.info("Files found: {}".format(filenames))
    if filenames:
        return render_template(
            "template.html",
            all_filenames=filenames,
            return_predictions=return_predictions,
        )
    else:
        return render_template("index.html")


@app.route("/uploads/<file_name>")
def send_file(file_name):
    """
    Fetches the image which file name is passed as argument

    Args:
        file_name: (string) Uploaded image file name

    Returns: Call to the send_from_directory() function
    """
    return send_from_directory(UPLOAD_FOLDER, file_name)


@app.route("/predictions/<all_filenames>")
def return_predictions(all_filenames):
    """
    Calls the webservice to get the predicted classes of the uploaded images

    Args:
        all_filenames: (list of strings) List of uploaded image file names

    Returns: Call to the predict_from_service() function
    """
    return predict_from_service(UPLOAD_FOLDER, all_filenames)


if __name__ == "__main__":
    app.run()
