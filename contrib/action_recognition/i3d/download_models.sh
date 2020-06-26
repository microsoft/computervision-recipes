#!/usr/bin/env bash
wget https://har.blob.core.windows.net/i3dmodels/flow_hmdb_split1.pt
wget https://har.blob.core.windows.net/i3dmodels/rgb_hmdb_split1.pt
wget https://har.blob.core.windows.net/i3dmodels/flow_imagenet_kinetics.pt
wget https://har.blob.core.windows.net/i3dmodels/rgb_imagenet_kinetics.pt

mv flow_hmdb_split1.pt pretrained_models/flow_hmdb_split1.pt
mv rgb_hmdb_split1.pt pretrained_models/rgb_hmdb_split1.pt
mv flow_imagenet_kinetics.pt pretrained_models/flow_imagenet_kinetics.pt
mv rgb_imagenet_kinetics.pt pretrained_models/rgb_imagenet_kinetics.pt