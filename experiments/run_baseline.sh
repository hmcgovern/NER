#!/bin/bash

# set -e

DATA_DIR="data/"
KEYWORD="baseline"

OUTPUT_DIR="experiments/baseline/" # this will change depending on experiment
CHECKPOINT_DIR="checkpoints/baseline"

# if the output and checkpoint directories don't already exist, make them
if [ ! -d ${OUTPUT_DIR} ]; then
    mkdir -p ${OUTPUT_DIR}
fi

if [ ! -d ${CHECKPOINT_DIR} ]; then
    mkdir -p ${CHECKPOINT_DIR}
fi

# only run this if .npy files don't exist
# arbitrarily choosing train.seqs as the check
if [ -f "${DATA_DIR}/padded/train.pkl" ] && [ -f "${DATA_DIR}/seqs/train_seqs.pkl" ]; then
    echo "Using existing processed data files..."
else
    echo "No existing data files found, preprocessing now..."
    echo "Input data and labels will be saved in ${DATA_DIR}/tmp"
    python preprocess.py --output-dir ${OUTPUT_DIR} --data-dir ${DATA_DIR}
fi
  
python train.py \
                --train-data-path ${DATA_DIR}/padded/train.pkl \
                --dev-data-path ${DATA_DIR}/padded/dev.pkl \
                --test-data-path ${DATA_DIR}/padded/test.pkl \
                --token-vocab ${DATA_DIR}/token_vocab.pkl \
                --train-epochs 2 \
                --batch-size 32 \
                --checkpoint-dir ${CHECKPOINT_DIR}/baseline.ckpt \
                --hparams-path ${OUTPUT_DIR}/hparams.json \
                --output-file ${OUTPUT_DIR}/predictions.txt \
                --overwrite
                #--real-deal
    

# python train.py predict BUT ON TEST DATA, and direct it to the results dir for final eval, not the corresponding exp folder.
python evaluate.py ${OUTPUT_DIR}/predictions.txt > ${OUTPUT_DIR}/eval.txt






