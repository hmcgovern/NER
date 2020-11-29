#!/bin/bash

# set -e

DATA_DIR="data/"
KEYWORD="downweight001"

OUTPUT_DIR="experiments/"$KEYWORD # this will change depending on experiment
CHECKPOINT_DIR="checkpoints/"$KEYWORD

# if the output and checkpoint directories don't already exist, make them
# make the output dir and place in a template of the json
if [ ! -d ${OUTPUT_DIR} ]; then
    echo "no checkpoint directory found, please create one with a correct hparams file"
    # mkdir -p ${OUTPUT_DIR}
    # # create a dummy hparams file
    # cp "experiments/baseline/hparams.json" "${OUTPUT_DIR}/hparams.json"
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
                --train-epochs 100 \
                --batch-size 32 \
                --checkpoint-dir ${CHECKPOINT_DIR}/baseline.ckpt \
                --hparams-path ${OUTPUT_DIR}/hparams.json \
                --output-file ${OUTPUT_DIR}/predictions.txt \
                #--real-deal
    

# # python train.py predict BUT ON TEST DATA, and direct it to the results dir for final eval, not the corresponding exp folder.
python evaluate.py ${OUTPUT_DIR}/predictions.txt > ${OUTPUT_DIR}/eval.txt






