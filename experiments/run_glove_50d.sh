#!/bin/bash

# set -e

DATA_DIR='data/'

python preprocess.py --data_dir ${DATA_DIR} --clean

# python train.py