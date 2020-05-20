import glob
import os
import os.path as osp
import random

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torchvision.ops import nms

# import maskrcnn_benchmark.layers.nms as nms
# Set printoptions
torch.set_printoptions(linewidth=1320, precision=5, profile='long')
np.set_printoptions(linewidth=320, formatter={'float_kind': '{:11.5g}'.format})  # format short g, %precision=5

def mkdir_if_missing(d):
    if not osp.exists(d):
        os.makedirs(d)


def float3(x):  # format floats to 3 decimals
    return float(format(x, '.3f'))


def init_seeds(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_classes(path):
    """
    Loads class labels at 'path'
    """
    fp = open(path, 'r')
    names = fp.read().split('\n')
    return list(filter(None, names))  # filter removes empty strings (such as last line)


def model_info(model):  # Plots a line-by-line description of a PyTorch model
    n_p = sum(x.numel() for x in model.parameters())  # number parameters
    n_g = sum(x.numel() for x in model.parameters() if x.requires_grad)  # number gradients
    print('\n%5s %50s %9s %12s %20s %12s %12s' % ('layer', 'name', 'gradient', 'parameters', 'shape', 'mu', 'sigma'))
    for i, (name, p) in enumerate(model.named_parameters()):
        name = name.replace('module_list.', '')
        print('%5g %50s %9s %12g %20s %12.3g %12.3g' % (
            i, name, p.requires_grad, p.numel(), list(p.shape), p.mean(), p.std()))
    print('Model Summary: %g layers, %g parameters, %g gradients\n' % (i + 1, n_p, n_g))



def plot_one_box(x, img, color=None, label=None, line_thickness=None):  # Plots one bounding box on image img
    tl = line_thickness or round(0.0004 * max(img.shape[0:2])) + 1  # line thickness
    color = color or [random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(img, c1, c2, color, thickness=tl)
    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(img, c1, c2, color, -1)  # filled
        cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.03)
    elif classname.find('BatchNorm2d') != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.03)
        torch.nn.init.constant_(m.bias.data, 0.0)


def xyxy2xywh(x):
    # Convert bounding box format from [x1, y1, x2, y2] to [x, y, w, h]
    y = torch.zeros(x.shape) if x.dtype is torch.float32 else np.zeros(x.shape)
    y[:, 0] = (x[:, 0] + x[:, 2]) / 2
    y[:, 1] = (x[:, 1] + x[:, 3]) / 2
    y[:, 2] = x[:, 2] - x[:, 0]
    y[:, 3] = x[:, 3] - x[:, 1]
    return y


def xywh2xyxy(x):
    # Convert bounding box format from [x, y, w, h] to [x1, y1, x2, y2]
    y = torch.zeros(x.shape) if x.dtype is torch.float32 else np.zeros(x.shape)
    y[:, 0] = (x[:, 0] - x[:, 2] / 2)
    y[:, 1] = (x[:, 1] - x[:, 3] / 2)
    y[:, 2] = (x[:, 0] + x[:, 2] / 2)
    y[:, 3] = (x[:, 1] + x[:, 3] / 2)
    return y


def scale_coords(img_size, coords, img0_shape):
    # Rescale x1, y1, x2, y2 from 416 to image size
    gain_w = float(img_size[0]) / img0_shape[1]  # gain  = old / new
    gain_h = float(img_size[1]) / img0_shape[0]
    gain = min(gain_w, gain_h)
    pad_x = (img_size[0] - img0_shape[1] * gain) / 2  # width padding
    pad_y = (img_size[1] - img0_shape[0] * gain) / 2  # height padding
    coords[:, [0, 2]] -= pad_x
    coords[:, [1, 3]] -= pad_y
    coords[:, 0:4] /= gain
    coords[:, :4] = torch.clamp(coords[:, :4], min=0)
    return coords


def ap_per_class(tp, conf, pred_cls, target_cls):
    """ Compute the average precision, given the recall and precision curves.
    Method originally from https://github.com/rafaelpadilla/Object-Detection-Metrics.
    # Arguments
        tp:    True positives (list).
        conf:  Objectness value from 0-1 (list).
        pred_cls: Predicted object classes (list).
        target_cls: True object classes (list).
    # Returns
        The average precision as computed in py-faster-rcnn.
    """

    # lists/pytorch to numpy
    tp, conf, pred_cls, target_cls = np.array(tp), np.array(conf), np.array(pred_cls), np.array(target_cls)

    # Sort by objectness
    i = np.argsort(-conf)
    tp, conf, pred_cls = tp[i], conf[i], pred_cls[i]

    # Find unique classes
    unique_classes = np.unique(np.concatenate((pred_cls, target_cls), 0))

    # Create Precision-Recall curve and compute AP for each class
    ap, p, r = [], [], []
    for c in unique_classes:
        i = pred_cls == c
        n_gt = sum(target_cls == c)  # Number of ground truth objects
        n_p = sum(i)  # Number of predicted objects

        if (n_p == 0) and (n_gt == 0):
            continue
        elif (n_p == 0) or (n_gt == 0):
            ap.append(0)
            r.append(0)
            p.append(0)
        else:
            # Accumulate FPs and TPs
            fpc = np.cumsum(1 - tp[i])
            tpc = np.cumsum(tp[i])

            # Recall
            recall_curve = tpc / (n_gt + 1e-16)
            r.append(tpc[-1] / (n_gt + 1e-16))

            # Precision
            precision_curve = tpc / (tpc + fpc)
            p.append(tpc[-1] / (tpc[-1] + fpc[-1]))

            # AP from recall-precision curve
            ap.append(compute_ap(recall_curve, precision_curve))

    return np.array(ap), unique_classes.astype('int32'), np.array(r), np.array(p)


def compute_ap(recall, precision):
    """ Compute the average precision, given the recall and precision curves.
    Code originally from https://github.com/rbgirshick/py-faster-rcnn.
    # Arguments
        recall:    The recall curve (list).
        precision: The precision curve (list).
    # Returns
        The average precision as computed in py-faster-rcnn.
    """
    # correct AP calculation
    # first append sentinel values at the end

    mrec = np.concatenate(([0.], recall, [1.]))
    mpre = np.concatenate(([0.], precision, [0.]))

    # compute the precision envelope
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    # to calculate area under PR curve, look for points
    # where X axis (recall) changes value
    i = np.where(mrec[1:] != mrec[:-1])[0]

    # and sum (\Delta recall) * prec
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap


def bbox_iou(box1, box2, x1y1x2y2=False):
    """
    Returns the IoU of two bounding boxes
    """
    N, M = len(box1), len(box2)
    if x1y1x2y2:
        # Get the coordinates of bounding boxes
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[:, 0], box1[:, 1], box1[:, 2], box1[:, 3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[:, 0], box2[:, 1], box2[:, 2], box2[:, 3]
    else:
        # Transform from center and width to exact coordinates
        b1_x1, b1_x2 = box1[:, 0] - box1[:, 2] / 2, box1[:, 0] + box1[:, 2] / 2
        b1_y1, b1_y2 = box1[:, 1] - box1[:, 3] / 2, box1[:, 1] + box1[:, 3] / 2
        b2_x1, b2_x2 = box2[:, 0] - box2[:, 2] / 2, box2[:, 0] + box2[:, 2] / 2
        b2_y1, b2_y2 = box2[:, 1] - box2[:, 3] / 2, box2[:, 1] + box2[:, 3] / 2

    # get the coordinates of the intersection rectangle
    inter_rect_x1 = torch.max(b1_x1.unsqueeze(1), b2_x1)
    inter_rect_y1 = torch.max(b1_y1.unsqueeze(1), b2_y1)
    inter_rect_x2 = torch.min(b1_x2.unsqueeze(1), b2_x2)
    inter_rect_y2 = torch.min(b1_y2.unsqueeze(1), b2_y2)
    # Intersection area
    inter_area = torch.clamp(inter_rect_x2 - inter_rect_x1, 0) * torch.clamp(inter_rect_y2 - inter_rect_y1, 0)
    # Union Area
    b1_area = ((b1_x2 - b1_x1) * (b1_y2 - b1_y1))
    b1_area = ((b1_x2 - b1_x1) * (b1_y2 - b1_y1)).view(-1,1).expand(N,M)
    b2_area = ((b2_x2 - b2_x1) * (b2_y2 - b2_y1)).view(1,-1).expand(N,M)

    return inter_area / (b1_area + b2_area - inter_area + 1e-16)


def build_targets_max(target, anchor_wh, nA, nC, nGh, nGw):
    """
    returns nT, nCorrect, tx, ty, tw, th, tconf, tcls
    """
    nB = len(target)  # number of images in batch

    txy = torch.zeros(nB, nA, nGh, nGw, 2).cuda()  # batch size, anchors, grid size
    twh = torch.zeros(nB, nA, nGh, nGw, 2).cuda()
    tconf = torch.LongTensor(nB, nA, nGh, nGw).fill_(0).cuda()
    tcls = torch.ByteTensor(nB, nA, nGh, nGw, nC).fill_(0).cuda()  # nC = number of classes
    tid = torch.LongTensor(nB, nA, nGh, nGw, 1).fill_(-1).cuda() 
    for b in range(nB):
        t = target[b]
        t_id = t[:, 1].clone().long().cuda()
        t = t[:,[0,2,3,4,5]]
        nTb = len(t)  # number of targets
        if nTb == 0:
            continue

        #gxy, gwh = t[:, 1:3] * nG, t[:, 3:5] * nG
        gxy, gwh = t[: , 1:3].clone() , t[:, 3:5].clone()
        gxy[:, 0] = gxy[:, 0] * nGw
        gxy[:, 1] = gxy[:, 1] * nGh
        gwh[:, 0] = gwh[:, 0] * nGw
        gwh[:, 1] = gwh[:, 1] * nGh
        gi = torch.clamp(gxy[:, 0], min=0, max=nGw -1).long()
        gj = torch.clamp(gxy[:, 1], min=0, max=nGh -1).long()

        # Get grid box indices and prevent overflows (i.e. 13.01 on 13 anchors)
        #gi, gj = torch.clamp(gxy.long(), min=0, max=nG - 1).t()
        #gi, gj = gxy.long().t()

        # iou of targets-anchors (using wh only)
        box1 = gwh
        box2 = anchor_wh.unsqueeze(1)
        inter_area = torch.min(box1, box2).prod(2)
        iou = inter_area / (box1.prod(1) + box2.prod(2) - inter_area + 1e-16)

        # Select best iou_pred and anchor
        iou_best, a = iou.max(0)  # best anchor [0-2] for each target

        # Select best unique target-anchor combinations
        if nTb > 1:
            _, iou_order = torch.sort(-iou_best)  # best to worst

            # Unique anchor selection
            u = torch.stack((gi, gj, a), 0)[:, iou_order]
            # _, first_unique = np.unique(u, axis=1, return_index=True)  # first unique indices
            first_unique = return_torch_unique_index(u, torch.unique(u, dim=1))  # torch alternative
            i = iou_order[first_unique]
            # best anchor must share significant commonality (iou) with target
            i = i[iou_best[i] > 0.60]  # TODO: examine arbitrary threshold
            if len(i) == 0:
                continue

            a, gj, gi, t = a[i], gj[i], gi[i], t[i]
            t_id = t_id[i]
            if len(t.shape) == 1:
                t = t.view(1, 5)
        else:
            if iou_best < 0.60:
                continue
        
        tc, gxy, gwh = t[:, 0].long(), t[:, 1:3].clone(), t[:, 3:5].clone()
        gxy[:, 0] = gxy[:, 0] * nGw
        gxy[:, 1] = gxy[:, 1] * nGh
        gwh[:, 0] = gwh[:, 0] * nGw
        gwh[:, 1] = gwh[:, 1] * nGh

        # XY coordinates
        txy[b, a, gj, gi] = gxy - gxy.floor()

        # Width and height
        twh[b, a, gj, gi] = torch.log(gwh / anchor_wh[a])  # yolo method
        # twh[b, a, gj, gi] = torch.sqrt(gwh / anchor_wh[a]) / 2 # power method

        # One-hot encoding of label
        tcls[b, a, gj, gi, tc] = 1
        tconf[b, a, gj, gi] = 1
        tid[b, a, gj, gi] = t_id.unsqueeze(1)
    tbox = torch.cat([txy, twh], -1)
    return tconf, tbox, tid




def generate_anchor(nGh, nGw, anchor_wh):
    nA = len(anchor_wh)
    yy, xx =torch.meshgrid(torch.arange(nGh), torch.arange(nGw))
    xx, yy = xx.cuda(), yy.cuda()

    mesh = torch.stack([xx, yy], dim=0)                                              # Shape 2, nGh, nGw
    mesh = mesh.unsqueeze(0).repeat(nA,1,1,1).float()                                # Shape nA x 2 x nGh x nGw
    anchor_offset_mesh = anchor_wh.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, nGh,nGw) # Shape nA x 2 x nGh x nGw
    anchor_mesh = torch.cat([mesh, anchor_offset_mesh], dim=1)                       # Shape nA x 4 x nGh x nGw
    return anchor_mesh

def encode_delta(gt_box_list, fg_anchor_list):
    px, py, pw, ph = fg_anchor_list[:, 0], fg_anchor_list[:,1], \
                     fg_anchor_list[:, 2], fg_anchor_list[:,3]
    gx, gy, gw, gh = gt_box_list[:, 0], gt_box_list[:, 1], \
                     gt_box_list[:, 2], gt_box_list[:, 3]
    dx = (gx - px) / pw
    dy = (gy - py) / ph
    dw = torch.log(gw/pw)
    dh = torch.log(gh/ph)
    return torch.stack([dx, dy, dw, dh], dim=1)

def decode_delta(delta, fg_anchor_list):
    px, py, pw, ph = fg_anchor_list[:, 0], fg_anchor_list[:,1], \
                     fg_anchor_list[:, 2], fg_anchor_list[:,3]
    dx, dy, dw, dh = delta[:, 0], delta[:, 1], delta[:, 2], delta[:, 3]
    gx = pw * dx + px
    gy = ph * dy + py
    gw = pw * torch.exp(dw)
    gh = ph * torch.exp(dh)
    return torch.stack([gx, gy, gw, gh], dim=1)

def decode_delta_map(delta_map, anchors):
    '''
    :param: delta_map, shape (nB, nA, nGh, nGw, 4)
    :param: anchors, shape (nA,4)
    '''
    nB, nA, nGh, nGw, _ = delta_map.shape
    anchor_mesh = generate_anchor(nGh, nGw, anchors) 
    anchor_mesh = anchor_mesh.permute(0,2,3,1).contiguous()              # Shpae (nA x nGh x nGw) x 4
    anchor_mesh = anchor_mesh.unsqueeze(0).repeat(nB,1,1,1,1)
    pred_list = decode_delta(delta_map.view(-1,4), anchor_mesh.view(-1,4))
    pred_map = pred_list.view(nB, nA, nGh, nGw, 4)
    return pred_map


def pooling_nms(heatmap, kernel=1):
    pad = (kernel -1 ) // 2
    hmax = F.max_pool2d(heatmap, (kernel, kernel), stride=1, padding=pad)
    keep = (hmax == heatmap).float()
    return keep * heatmap


def non_max_suppression(prediction, conf_thres=0.5, nms_thres=0.2):
    """
    Removes detections with lower object confidence score than 'conf_thres'
    Non-Maximum Suppression to further filter detections.
    Returns detections with shape:
        (x1, y1, x2, y2, object_conf, class_score, class_pred)
    """

    output = [None for _ in range(len(prediction))]
    for image_i, pred in enumerate(prediction):
        # Filter out confidence scores below threshold
        # Get score and class with highest confidence

        v = pred[:, 4] > conf_thres
        v = v.nonzero().squeeze()
        if len(v.shape) == 0:
            v = v.unsqueeze(0)

        pred = pred[v]

        # If none are remaining => process next image
        nP = pred.shape[0]
        if not nP:
            continue
        # From (center x, center y, width, height) to (x1, y1, x2, y2)
        pred[:, :4] = xywh2xyxy(pred[:, :4])
        nms_indices = nms(pred[:, :4], pred[:, 4], nms_thres)
        det_max = pred[nms_indices]        

        if len(det_max) > 0:
            # Add max detections to outputs
            output[image_i] = det_max if output[image_i] is None else torch.cat((output[image_i], det_max))

    return output


def return_torch_unique_index(u, uv):
    n = uv.shape[1]  # number of columns
    first_unique = torch.zeros(n, device=u.device).long()
    for j in range(n):
        first_unique[j] = (uv[:, j:j + 1] == u).all(0).nonzero()[0]

    return first_unique


def strip_optimizer_from_checkpoint(filename='weights/best.pt'):
    # Strip optimizer from *.pt files for lighter files (reduced by 2/3 size)

    a = torch.load(filename, map_location='cpu')
    a['optimizer'] = []
    torch.save(a, filename.replace('.pt', '_lite.pt'))


def plot_results():
    # Plot YOLO training results file 'results.txt'
    # import os; os.system('wget https://storage.googleapis.com/ultralytics/yolov3/results_v1.txt')

    plt.figure(figsize=(14, 7))
    s = ['X + Y', 'Width + Height', 'Confidence', 'Classification', 'Total Loss', 'mAP', 'Recall', 'Precision']
    files = sorted(glob.glob('results*.txt'))
    for f in files:
        results = np.loadtxt(f, usecols=[2, 3, 4, 5, 6, 9, 10, 11]).T  # column 11 is mAP
        x = range(1, results.shape[1])
        for i in range(8):
            plt.subplot(2, 4, i + 1)
            plt.plot(x, results[i, x], marker='.', label=f)
            plt.title(s[i])
            if i == 0:
                plt.legend()
