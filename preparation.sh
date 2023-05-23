#! /bin/bash

git clone https://github.com/SHI-Labs/Compact-Transformers.git
mv Compact-Transformers/ ct
git clone https://github.com/akamaster/pytorch_resnet_cifar10.git
wget https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-320.tgz
tar zxvf imagenette2-320.tgz
mkdir data
mv imagenette2-320 data/
pip install -r requirements.txt