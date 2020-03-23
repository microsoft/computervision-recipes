#!/bin/bash
cd /data

# clone repo and install the conda env 
git clone https://www.github.com/microsoft/computervision 
# change permission as we copy this into each user's folder
chmod -R ugo+rwx /data/computervision

# enable conda if not done yet
. /data/anaconda/etc/profile.d/conda.sh

# create conda env and kernel
conda env create -f /data/computervision/environment.yml --name cv
conda activate cv 
python -m ipykernel install --name cv --display-name "MLADS CV LAB" 

# add users to jupyterhub
echo 'c.Authenticator.whitelist = {"user1", "user2"}' | tee -a /etc/jupyterhub/jupyterhub_config.py

# create the users on the vm 
for i in $(seq 2)
do
    USERNAME=user$i
    PASSWORD=password$i
    adduser --quiet --disabled-password --gecos "" $USERNAME
    echo "$USERNAME:$PASSWORD" | chpasswd
    rm -rf /data/home/$USERNAME/notebooks/*
    # copy repo
    cp -ar /data/computervision /data/home/$USERNAME/notebooks
done

# restart jupyterhub service
systemctl stop jupyterhub 
systemctl start jupyterhub 

exit
