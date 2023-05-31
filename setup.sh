#!/bin/bash
# General
## root dir, you can change it to your specified dir, but remember to change the path in the following script, root_dir in the config.py and src/utils/write_xxx.py
sudo mkdir -p ~/MT
sudo chmod -R 777 ~/MT
cd ~/MT

## plugins
sudo apt-get install software-properties-common

# ManagerTower
## update Python
sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt update
sudo apt install python3.8 python3.8-venv

## create virtual environment
cd ~/MT
python3.8 -m venv venv_ManagerTower
# echo "alias venv_ManagerTower='source ~/MT/venv_ManagerTower/bin/activate'" >> ~/.bashrc
# echo "source ~/MT/venv_ManagerTower/bin/activate" >> ~/.bashrc
# source ~/.bashrc

## git clone
cd ~/MT
git clone https://github.com/LooperXX/ManagerTower.git
cd BridgeTower

## dependency
source ~/MT/venv_ManagerTower/bin/activate
pip install --upgrade pip
pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html
pip install -r requirements.txt
pip install evalai
pip install --upgrade requests click python-dateutil

## mkdir
sudo mkdir -p ~/MT
sudo mkdir -p ~/MT/dataset/
sudo mkdir -p ~/MT/best_checkpoints/
sudo mkdir -p ~/MT/checkpoints/
sudo mkdir -p ~/MT/logs/
sudo chmod -R 777 ~/MT

## download data and checkpoints, and put them in ~/MT/dataset/ and ~/MT/best_checkpoints/
## the final file structure of ~/MT/dataset/ should be like this:
# root
#  └── dataset
#      ├── pre-train
#      ├── fine-tune
#      ├── sbu
#      ├── cc
#      ├── nlvr
#      │   ├── dev
#      │   ├── images
#      │   ├── nlvr
#      │   ├── nlvr2
#      │   ├── test1
#      │   └── README.md
#      ├── vg
#      │   ├── annotations
#      │   ├── coco_splits
#      │   ├── images
#      │   ├── vgqa
#      │   └── image_data.json
#      └── mscoco_flickr30k_vqav2_snli_ve
#          ├── flickr30k-images
#          ├── karpathy
#          ├── snli_ve
#          ├── test2015
#          ├── train2014
#          ├── val2014
#          └── vqav2

## then run the src/utils/write_xxx.py to convert the dataset to pyarrow binary file.

# python src/utils/write_coco_karpathy.py
# python src/utils/write_conceptual_caption.py
# python src/utils/write_f30k_karpathy.py
# python src/utils/write_nlvr2.py
# python src/utils/write_sbu.py
# python src/utils/write_vg.py
# python src/utils/write_vqa.py
# python src/utils/vgqa_split.py
# python src/utils/write_vgqa.py
# cp ~/MT/dataset/pre-train/coco_caption_karpathy_* ~/MT/dataset/fine-tune