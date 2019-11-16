#!/bin/bash

echo "[download] model"
DIR="$(cd "$(dirname "$0")" && pwd)"
echo "curr dir" $DIR
download_url="https://www.dropbox.com/s/8bxwvr4cj4bh5d8/mcnn_shtechA_660.h5?dl=0"
wget -c --tries=2 $download_url -O $DIR/mcnn_shtechA_660.h5
echo "[download] end"


