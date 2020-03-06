# Most of the code in this file is copied and slightly modified from:
# https://github.com/layumi/Person_reID_baseline_pytorch/blob/master/evaluate.py

import numpy as np
import time
import torch

from .re_ranking import re_ranking


def evaluate(ds, dnn_features, use_rerank = False):  #labels, features
    labels = np.array([ds.y[i].obj for i in range(len(ds.y))])
    features = np.array([dnn_features[str(s)] for s in ds.items])  #np.array(list(dnn_features.values()))

    # Assign each image into its own group. This serves as id and is used to determine 
    # which reference and query image are the same image.
    groups = np.array(list(range(len(labels))))
    assert len(labels) == len(groups) == features.shape[0]

    # Use all images in the reference set also as query images 
    return evaluate_separate_query_set(labels, groups, features, labels, groups, features, use_rerank)



def evaluate_separate_query_set(gallery_labels, gallery_groups, gallery_features, query_labels, 
                                query_groups, query_features, use_rerank = False, is_market1501 = False):

    # Init
    ap = 0.0
    CMC = torch.IntTensor(len(gallery_labels)).zero_()
    
    # Compute pairwise distance
    q_g_dist = np.dot(query_features, np.transpose(gallery_features))
    q_g_dist.shape

    # Compute re-ranking
    if use_rerank:
        print('Calculate re-ranked distances..')
        q_q_dist = np.dot(query_features, np.transpose(query_features))
        g_g_dist = np.dot(gallery_features, np.transpose(gallery_features))
        since = time.time()
        distances = re_ranking(q_g_dist, q_q_dist, g_g_dist, k1=20, k2=6, lambda_value=0.3)
        time_elapsed = time.time() - since
        print('Reranking complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    else:
        distances = -q_g_dist

    # Compute accuracies
    norm = 0
    skip = 1 #to to >1 to only consider a subset of the query images
    for i in range(len(query_labels))[::skip]:
        ap_tmp, CMC_tmp = evaluate_helper(distances[i,:], query_labels[i], query_groups[i], gallery_labels, gallery_groups, is_market1501)
        if CMC_tmp[0]==-1:
            continue
        norm +=1
        ap += ap_tmp
        CMC = CMC + CMC_tmp

    # Print accuracy. Note that Market1501 normalizes by dividing over number of query images.
    if is_market1501:
        norm = len(query_labels)/float(skip)
    ap = ap/norm
    CMC = CMC.float()
    CMC = CMC/norm
    print('Rank@1:{:.1f}, rank@5:{:.1f}, rank@10:{:.1f}, mAP:{:.2f}'.format(100*CMC[0], 100*CMC[4], 100*CMC[9], ap))

    return (CMC, ap)


# Evaluation implementation
# Explanation:
# - query_index: all images in the reference set with the same label as the query image ("true match")
# - camera_index: all images which share the same group (here called "camera"). 
# - junk_index2: all reference images with the same group (here called "camera") as the query are considered "false matches".
# - junk_index1: for the market1501 dataset, images with label -1 should be ignored.
def evaluate_helper(score,ql,qc,gl,gc,is_market1501 = False):
    assert type(gl) == np.ndarray, "Input gl has to be a numpy ndarray"
    assert type(gc) == np.ndarray, "Input gc has to be a numpy ndarray"
    
    # Sort scores 
    index = np.argsort(score)  #from small to large

    # Determine which refernces images, when compared to the query image, are considered 
    # "true matches" or "false matches"  
    query_index = np.argwhere(gl==ql)
    camera_index = np.argwhere(gc==qc)
    good_index = np.setdiff1d(query_index, camera_index, assume_unique=True)
    junk_index2 = np.intersect1d(query_index, camera_index)
    
    # For market 1501 dataset, ignore images with label -1
    if is_market1501:
        junk_index1a = np.argwhere(gl==-1)
        junk_index1b = np.argwhere(gl=="-1")
        junk_index = np.append(np.append(junk_index2, junk_index1a), junk_index1b)
    else:
        junk_index = junk_index2
    
    CMC_tmp = compute_mAP(index, good_index, junk_index)
    return CMC_tmp


def compute_mAP(index, good_index, junk_index):
    ap = 0
    cmc = torch.IntTensor(len(index)).zero_()
    if good_index.size==0:   # if empty
        cmc[0] = -1
        return ap,cmc

    # remove junk_index
    mask = np.in1d(index, junk_index, invert=True)
    index = index[mask]

    # find good_index index
    ngood = len(good_index)
    mask = np.in1d(index, good_index)
    rows_good = np.argwhere(mask==True)
    rows_good = rows_good.flatten()
    
    cmc[rows_good[0]:] = 1
    for i in range(ngood):
        d_recall = 1.0/ngood
        precision = (i+1)*1.0/(rows_good[i]+1)
        if rows_good[i]!=0:
            old_precision = i*1.0/rows_good[i]
        else:
            old_precision=1.0
        ap = ap + d_recall*(old_precision + precision)/2

    return ap, cmc