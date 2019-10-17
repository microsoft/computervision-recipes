Setup environment
```
conda env create -f environment.yaml
conda activate i3d
```

Download pretrained models
```
bash download_models.sh
```

Train RGB model
```
python train.py --cfg config/train_rgb.yaml
```

Train flow model
```
python train.py --cfg config/train_flow.yaml
```

Evaluate combined model
```
python test.py
```