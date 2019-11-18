import os
import time
import base64
import urllib
from io import BytesIO
import cv2
import numpy as np
from tf_pose.estimator import TfPoseEstimator
from tf_pose.networks import get_graph_path, model_wh
from PIL import Image
import tensorflow as tf
import logging

import sys

# dir_path = os.path.dirname(os.path.realpath(__file__))
# sys.path.append(dir_path + "/../third_party/mcnn/src")
# import network
# from crowd_count import CrowdCounter

from crowdcountmcnn.src import network
from crowdcountmcnn.src.crowd_count import CrowdCounter

from abc import ABC, abstractmethod


class CrowdCounting(ABC):
    @abstractmethod
    def score(self): # pragma: no cover
        raise NotImplementedError


class Router(CrowdCounting):
    """Router model definition.
    
    Args:
        gpu_id: GPU ID, integer starting from 0. 
    """

    def __init__(
        self,
        gpu_id=0,
        mcnn_model_path="mcnn_shtechA_660.h5",
        cutoff_pose=20,
        cutoff_mcnn=50,
    ):
        self._model_openpose = CrowdCountModelPose(gpu_id)
        self._model_mcnn = CrowdCountModelMCNN(gpu_id, model_path=mcnn_model_path)
        self._cutoff_pose = cutoff_pose
        self._cutoff_mcnn = cutoff_mcnn
        self._logger = logging.getLogger(__name__)

    def score(self, filebytes, return_image=False, img_dim=1750):
        dict_openpose = self._model_openpose.score(
            filebytes, return_image, img_dim=img_dim
        )
        result_openpose = dict_openpose["pred"]

        dict_mcnn = self._model_mcnn.score(filebytes, return_image, img_dim=img_dim)
        result_mcnn = dict_mcnn["pred"]

        self._logger.info("OpenPose results: {}".format(result_openpose))
        self._logger.info("MCNN results: {}".format(result_mcnn))

        if result_openpose > self._cutoff_pose and result_mcnn > self._cutoff_mcnn:
            return dict_mcnn
        else:
            return dict_openpose


class CrowdCountModelMCNN(CrowdCounting):
    """MCNN model definition.
    
    Args:
        gpu_id: GPU ID, integer starting from 0. 
    """

    def __init__(self, gpu_id=0, model_path="mcnn_shtechA_660.h5"):
        # load MCNN
        self._net = CrowdCounter()
        network.load_net(model_path, self._net)
        if gpu_id == -1:
            self._net.cpu()
        else:
            self._net.cuda(gpu_id)
        self._net.eval()
        self._logger = logging.getLogger(__name__)

    def score(self, filebytes, return_image=False, img_dim=1750):
        """Score an image. 
        
        Args:
            filebytes: Image in stream.
            return_image (optional): Whether a scored image needs to be returned, defaults to False. 
            img_dim (optional): Max dimension of image, defaults to 1750.
        
        Returns:
            A dictionary with number of people in image, timing for steps, and optionally, returned image.
        """
        self._logger.info("---started scoring image using MCNN---")
        t = time.time()
        image = load_jpg(filebytes, img_dim)
        t_image_prepare = round(time.time() - t, 3)

        self._logger.info("time on preparing image: {} seconds".format(t_image_prepare))
        t = time.time()
        pred_mcnn, model_output = score_mcnn(self._net, image)
        t_score = round(time.time() - t, 3)
        self._logger.info("time on scoring image: {} seconds".format(t_score))

        result = {}
        result["pred"] = int(round(pred_mcnn, 0))

        if not return_image:
            dict_time = dict(
                zip(["t_image_prepare", "t_score"], [t_image_prepare, t_score])
            )
        else:
            t = time.time()
            scored_image = draw_image_mcnn(model_output)
            t_image_draw = round(time.time() - t, 3)
            self._logger.info("time on drawing image: {}".format(t_image_draw))
            t = time.time()
            scored_image = web_encode_image(scored_image)
            t_image_encode = round(time.time() - t, 3)
            self._logger.info("time on encoding image: {}".format(t_image_encode))

            dict_time = dict(
                zip(
                    ["t_image_prepare", "t_score", "t_image_draw", "t_image_encode"],
                    [t_image_prepare, t_score, t_image_draw, t_image_encode],
                )
            )
            result["image"] = scored_image
        # sum up total time
        t_total = 0
        for k in dict_time:
            t_total += dict_time[k]
        dict_time["t_total"] = round(t_total, 3)
        self._logger.info("total time: {}".format(round(t_total, 3)))
        result["time"] = dict_time
        self._logger.info("---finished scoring image---")
        return result


class CrowdCountModelPose(CrowdCounting):
    """OpenPose model definition.
    
    Args:
        gpu_id: GPU ID, integer starting from 0. Set it to -1 to use CPU.
    """

    def __init__(self, gpu_id=0):
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True
        config.log_device_placement = True
        model = "cmu"
        resize = "656x368"
        self._w, self._h = model_wh(resize)
        self._model = init_model(gpu_id, model, self._w, self._h, config)
        self._logger = logging.getLogger(__name__)

    def score(self, filebytes, return_image=False, img_dim=1750):
        """Score an image. 
        
        Args:
            filebytes: Image in stream.
            return_image (optional): Whether a scored image needs to be returned, defaults to False. 
            img_dim (optional): Max dimension of image, defaults to 1750.
        
        Returns:
            A dictionary with number of people in image, timing for steps, and optionally, returned image.
        """
        self._logger.info("---started scoring image using OpenPose---")
        t = time.time()
        img = create_openpose_image(filebytes, img_dim)
        t_image_prepare = round(time.time() - t, 3)
        self._logger.info("time on preparing image: {} seconds".format(t_image_prepare))
        t = time.time()
        humans = score_openpose(self._model, img, self._w, self._h)
        t_score = round(time.time() - t, 3)
        self._logger.info("time on scoring image: {} seconds".format(t_score))
        result = {}
        result["pred"] = len(humans)

        if not return_image:
            dict_time = dict(
                zip(["t_image_prepare", "t_score"], [t_image_prepare, t_score])
            )
        else:
            t = time.time()
            scored_image = draw_image(img, humans)
            t_image_draw = round(time.time() - t, 3)
            self._logger.info("time on drawing image: {}".format(t_image_draw))
            t = time.time()
            scored_image = web_encode_image(scored_image)
            t_image_encode = round(time.time() - t, 3)
            self._logger.info("time on encoding image: {}".format(t_image_encode))

            dict_time = dict(
                zip(
                    ["t_image_prepare", "t_score", "t_image_draw", "t_image_encode"],
                    [t_image_prepare, t_score, t_image_draw, t_image_encode],
                )
            )
            result["image"] = scored_image
        # sum up total time
        t_total = 0
        for k in dict_time:
            t_total += dict_time[k]
        dict_time["t_total"] = round(t_total, 3)
        self._logger.info("total time: {}".format(round(t_total, 3)))
        result["time"] = dict_time
        self._logger.info("---finished scoring image---")
        return result


def init_model(gpu_id, model, w, h, config):
    """Initialize model.
    
    Args:
        gpu_id: GPU ID. 
    
    Returns:
        A TensorFlow model object.
    """

    # if w == 0 or h == 0:
    #     w, h = 432, 368

    if gpu_id == -1: # pragma: no cover
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        e = TfPoseEstimator(get_graph_path(model), target_size=(w, h), tf_config=config)
    else:
        with tf.device("/device:GPU:{}".format(gpu_id)):
            e = TfPoseEstimator(
                get_graph_path(model), target_size=(w, h), tf_config=config
            )
    return e


def create_openpose_image(filebytes, img_dim):
    """Create image from file bytes.
    
    Args:
        filebytes: Image in stream.
        img_dim: Max dimension of image.
    
    Returns:
        Image in CV2 format. 
    """
    # file_bytes = np.asarray(bytearray(BytesIO(filebytes).read()), dtype=np.uint8)
    file_bytes = np.fromstring(filebytes, np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    img, _ = imresizeMaxDim(img, img_dim)
    return img


def load_jpg(file_bytes, img_dim):
    image = np.fromstring(file_bytes, np.uint8)
    image = cv2.imdecode(image, 0).astype(np.float32)

    image, _ = imresizeMaxDim(image, img_dim)

    ht = image.shape[0]
    wd = image.shape[1]
    ht_1 = int(ht / 4) * 4
    wd_1 = int(wd / 4) * 4
    image = cv2.resize(image, (wd_1, ht_1))
    image = image.reshape((1, 1, image.shape[0], image.shape[1]))
    return image


def score_openpose(e, image, w, h):
    """Score an image using OpenPose model.
    
    Args:
        e: OpenPose model.
        image: Image in CV2 format.
    
    Returns:
        Nubmer of people in image.
    """
    resize_out_ratio = 4.0
    humans = e.inference(
        image, resize_to_default=(w > 0 and h > 0), upsample_size=resize_out_ratio
    )
    return humans


def draw_image(image, humans):
    image = TfPoseEstimator.draw_humans(image, humans, imgcopy=False)
    img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    imgDebug = Image.fromarray(img)
    return imgDebug


def web_encode_image(scored_image):
    ret_imgio = BytesIO()
    scored_image.save(ret_imgio, "PNG")
    processed_file = base64.b64encode(ret_imgio.getvalue())
    scored_image = urllib.parse.quote(processed_file)
    return scored_image


def imresizeMaxDim(img, maxDim, boUpscale=False, interpolation=cv2.INTER_CUBIC):
    """Resize image.
    
    Args:
        img: Image in CV2 format. 
        maxDim: Maximum dimension. 
        boUpscale (optional): Defaults to False. 
        interpolation (optional): Defaults to cv2.INTER_CUBIC. 
    
    Returns:
        Resized image and scale.
    """
    scale = 1.0 * maxDim / max(img.shape[:2])
    if scale < 1 or boUpscale:
        img = cv2.resize(img, (0, 0), fx=scale, fy=scale, interpolation=interpolation)
    else:
        scale = 1.0
    return img, scale


def score_mcnn(net, image):
    model_output = net(image)
    model_output_np = model_output.data.cpu().numpy()
    estimated_count = np.sum(model_output_np)
    return estimated_count, model_output


def draw_image_mcnn(model_output):
    estimated_density = model_output.data.cpu().numpy()[0, 0, :, :]
    estimated_density = np.uint8(estimated_density * 255 / estimated_density.max())
    im = Image.fromarray(estimated_density, "L")
    return im
