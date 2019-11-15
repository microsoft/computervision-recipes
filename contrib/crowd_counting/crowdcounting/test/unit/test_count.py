import os
import sys
import pytest

# from crowdcounting import CrowdCountModelPose, CrowdCountModelMCNN, Router
from crowdcounting.api.model_crowdcount import CrowdCountModelPose, CrowdCountModelMCNN, Router

@pytest.fixture
def local_root():
    dir_path = os.path.dirname(os.path.realpath(__file__))
    root_dir = os.path.abspath(dir_path+"/../..")
    return root_dir

@pytest.fixture
def local_image_sparse(local_root):
    local_image_sparse = os.path.join(local_root, "data/images/1.jpg")    
    return local_image_sparse

@pytest.fixture
def local_image_dense(local_root):
    local_image_dense = os.path.join(local_root, "data/images/2.jpg")    
    return local_image_dense

@pytest.fixture
def mcnn_model(local_root):
    mcnn_model_path = os.path.join(local_root, "data/models/mcnn_shtechA_660.h5")      
    return mcnn_model_path

def test_pose_init_gpu():
    gpu_id = 0
    model = CrowdCountModelPose(gpu_id)

def test_pose_score_large_scale(local_image_sparse):
    gpu_id = 0
    model = CrowdCountModelPose(gpu_id)
    with open(local_image_sparse, 'rb') as f:
        file_bytes = f.read()
    result = model.score(file_bytes, return_image=True, img_dim=1750)    
    assert result['pred'] == 3

def test_pose_score_small_scale(local_image_sparse):
    gpu_id = 0
    model = CrowdCountModelPose(gpu_id)
    with open(local_image_sparse, 'rb') as f:
        file_bytes = f.read()
    result = model.score(file_bytes, return_image=True, img_dim=500)    

def test_mcnn_init_cpu(mcnn_model):
    gpu_id = -1
    model = CrowdCountModelMCNN(gpu_id, model_path=mcnn_model)
            
def test_mcnn_init_gpu(mcnn_model):
    gpu_id = 0
    model = CrowdCountModelMCNN(gpu_id, model_path=mcnn_model)
                        
def test_mcnn_score(mcnn_model, local_image_sparse):
    gpu_id = 0
    model = CrowdCountModelMCNN(gpu_id, model_path=mcnn_model)
    with open(local_image_sparse, 'rb') as f:
        file_bytes = f.read()
    result = model.score(file_bytes, return_image=True, img_dim=1750)    
    assert result['pred'] == 12    

def test_router_init(mcnn_model):
    gpu_id = 0
    model = Router(gpu_id, mcnn_model_path=mcnn_model, cutoff_pose=20, cutoff_mcnn=50)

def test_router_score_sparse(mcnn_model, local_image_sparse):
    gpu_id = 0
    model = Router(gpu_id, mcnn_model_path=mcnn_model, cutoff_pose=20, cutoff_mcnn=50)
    with open(local_image_sparse, 'rb') as f:
        file_bytes = f.read()
    result = model.score(file_bytes, return_image=True, img_dim=1750)    

def test_router_score_dense(mcnn_model, local_image_dense):
    gpu_id = 0
    model = Router(gpu_id, mcnn_model_path=mcnn_model, cutoff_pose=20, cutoff_mcnn=50)
    with open(local_image_dense, 'rb') as f:
        file_bytes = f.read()
    result = model.score(file_bytes, return_image=False, img_dim=1750)    
