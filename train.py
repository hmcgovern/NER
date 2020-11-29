
from tensorflow import keras
import tensorflow as tf
import pandas as pd
import numpy as np
import argparse as ap
import os
import random
import pickle
import json

from collections import Counter

import utils

# TODO: change this so the embedding matrix doens't have to be passed in, but do that with Mittens
def make_model(metrics, hp, embedding_matrix=None):
    # TODO: don't hardcode this 
    n_labs = 4
    if hp['output_bias'] is not None:
        hp['output_bias'] = tf.keras.initializers.Constant(hp['output_bias'])
    if hp['glove_embedding_dim'] is not None:
        embed = keras.layers.Embedding(
                input_dim=hp['vocab_size'],
                output_dim=hp['glove_embedding_dim'],
                # TODO: factor out this embedding_matrix
                embeddings_initializer=keras.initializers.Constant(embedding_matrix),
                trainable=False)
    else:
        embed = keras.layers.Embedding(
                input_dim=hp['vocab_size'], 
                output_dim=hp['embed_size'], 
                input_length=hp['seq_length'], 
                mask_zero=True, 
                trainable=True)
  
    model = keras.Sequential([
        embed,
        keras.layers.Bidirectional(keras.layers.LSTM(units=hp['LSTM_units'], return_sequences=True, dropout=0.2, recurrent_dropout=0.2)),  # 2 directions, 50 units each, concatenated (can change this)
        keras.layers.Dropout(0.5),
        keras.layers.TimeDistributed(keras.layers.Dense(n_labs, activation='softmax', bias_initializer=hp['output_bias'])),
    ])
    model.compile(optimizer=keras.optimizers.Adam(lr=hp['lr']), loss=keras.losses.CategoricalCrossentropy(), metrics=metrics)
    return model

def main(args):
    # rendering the model deterministic (hopefully)
    SEED = 42
    os.environ['PYTHONHASHSEED']=str(SEED)
    os.environ['TF_CUDNN_DETERMINISTIC'] = '1'  # added in TF 2.1
    random.seed(SEED)
    np.random.seed(SEED)
    tf.random.set_seed(SEED)

    # load the hyperparameters from .json config file
    with open(args.hparams_path, 'r') as f:
        hparams = json.load(f)
    
    # here load the data from pickles saved during preprocessing
    with open(args.train_data_path, 'rb') as f:
        data = pickle.load(f)
        X = data['X']
        y = data['y']

    with open(args.dev_data_path, 'rb') as f:
        data = pickle.load(f)
        dev_X = data['dev_X']
        dev_y = data['dev_y']

    with open(args.test_data_path, 'rb') as f:
        data = pickle.load(f)
        test_X = data['test_X']

    # X = np.load(args.train_seqs_path)   
    # y = np.load(args.train_labs_path) 

    # remember to downweight the labels according to the hp
    weighted_y = utils.downweight(y, hparams['class_weights'])

    # test_X = np.load(args.test_seqs_path)  
    # test_y = np.load(args.test_labs_path) 

    # # # with open(args.token_vocab, 'rb') as f:
    # # #     token_vocab = pickle.load(f)

    # # # seq_len = X.shape[1]
    
    # some evaluation metrics
    METRICS = [
        keras.metrics.TruePositives(name='tp'),
        keras.metrics.FalsePositives(name='fp'),
        keras.metrics.TrueNegatives(name='tn'),
        keras.metrics.FalseNegatives(name='fn'), 
        keras.metrics.BinaryAccuracy(name='accuracy'),
        keras.metrics.Precision(name='precision'),
        keras.metrics.Recall(name='recall'),
        keras.metrics.AUC(name='auc'),
    ]

    # # # HP = {'output_bias': [0.005542151675485009, 0.0033213403880070548, 0.1667583774250441, 0.8243781305114638],
    # # #     'glove_embedding_dim': None,
    # # #     'class_weights': [1,1,0.1,0.1],
    # # #     'vocab_size': len(token_vocab)+2,
    # # #     'embed_size': 128,
    # # #     'seq_length': seq_len, 
    # # #     'LSTM_units': 50,
    # # #     'lr': 1e-3,}


    model = make_model(metrics=METRICS, hp=hparams)

    # early stopping criteria based on area under the curve: will stop if no improvement after 10 epochs
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_auc', verbose=1, patience=10, \
        mode='max', restore_best_weights=True)

    # Create a callback that saves the model's weights
    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=args.checkpoint_dir,
                                                 save_weights_only=True,
                                                 verbose=1)


    EPOCHS = args.train_epochs
    BATCH_SIZE = args.batch_size
    print(EPOCHS, BATCH_SIZE)

    print(model.summary())
    results = model.evaluate(X, y, batch_size=BATCH_SIZE, verbose=0)
    print("Loss: {:0.4f}".format(results[0])) #0.9711

    # fitting the model
    # model.fit(X, weighted_y, batch_size=BATCH_SIZE, epochs=EPOCHS, callbacks = [early_stopping, cp_callback], validation_data=(dev_X, dev_y))
    print("Loading weights")
    model.load_weights(args.checkpoint_dir).expect_partial() # idk why I need to have the expect_partial but error otherwise
    results = model.evaluate(X, y, batch_size=BATCH_SIZE, verbose=0)
    print("Loss: {:0.4f}".format(results[0])) #0.0568

    # if we are evaluating on the test data, aka done with development
    # it's different because we don't have gold labels for the test so we can't evaluate our predictions
    if args.real_deal:
        pass
    else:
        # get predictions on the dev data
        pass
    # get predictions on data, if real-deal flag is passed, it will evaluate on the test data and not run the evaluate script (it will be blind submission)
    # if not real deal, it will get preds on dev data and in bash will get passed to evaluate script

    # this file outputs a file called predictions.txt that will be passed to evaluate.py
    
if __name__ == '__main__':
    p = ap.ArgumentParser()
    # p.add_argument('mode', type=str, choices=['fit', 'predict'])
    p.add_argument('--train-data-path', type=str)
    p.add_argument('--dev-data-path', type=str)
    p.add_argument('--test-data-path', type=str)
    p.add_argument('--token-vocab', type=str)
    # p.add_argument('--model', default=None, type=str)
    p.add_argument('--hparams-path', default=None, type=str)
    p.add_argument('--label-map-path', type=str, default=None)
    p.add_argument('--output-file', type=str, default='predictions.txt')
    p.add_argument('--train-epochs', default=100, type=int)
    p.add_argument('--batch-size', type=int, default=32)
    p.add_argument('--checkpoint-dir', type=str, default=None)
    p.add_argument('--real-deal', action='store_true', help='if predicting on test data')

    args = p.parse_args()
    # TODO: write this to a text file somewhere just to have on hand

    main(args)