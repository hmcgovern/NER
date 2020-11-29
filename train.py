
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


def make_model(metrics, hp):
    n_labs = hp["n_labs"] # ==> 4
    if hp['output_bias'] is not None:
        hp['output_bias'] = tf.keras.initializers.Constant(hp['output_bias'])
    if type(hp['glove_embedding_dim']) == int:
        embed = keras.layers.Embedding(
                input_dim=hp['vocab_size'],
                output_dim=hp['glove_embedding_dim'],
                embeddings_initializer=keras.initializers.Constant(hp['embedding_matrix']),
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

def get_preds(model, seqs_padded, seqs, istest=False):
    preds = np.argmax(model.predict(seqs_padded), axis=-1)
    flat_preds = [p for pred in preds for p in pred]
    print(Counter(flat_preds))

    # start a new column for the model predictions
    seqs['prediction'] = ''

    # for each text: get original sequence length and trim predictions accordingly
    # (_trim_ because we know that our seq length is longer than the longest seq in dev)
    for i in seqs.index:
        this_seq_length = len(seqs['token'][i])
        seqs['prediction'][i] = preds[i][:this_seq_length].astype(int)

    # print(seqs.head())

    # use sequence number as the index and apply pandas explode to all other columns
    long = seqs.set_index('sequence_num').apply(pd.Series.explode).reset_index()
    # print(long.head())
    # re-using the BIO integer-to-character function from last time
    def reverse_bio(ind):
        bio = 'O'  # for any pad=3 predictions
        if ind==0:
            bio = 'B'
        elif ind==1:
            bio = 'I'
        elif ind==2:
            bio = 'O'
        return bio

    if not istest: # test file doesn't have bio_only
        bio_labs = [reverse_bio(b) for b in long['bio_only']]
        long['bio_only'] = bio_labs
    pred_labs = [reverse_bio(b) for b in long['prediction']]
    long['prediction'] = pred_labs

    print(long.prediction.value_counts())
    return long


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

    # remember to downweight the labels according to the hp
    weighted_y = utils.downweight(y, class_wts=hparams['class_weights'])


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

    ############## adding GloVe pre-trained embeddings ################
    if hparams['glove_embedding_dim'] is not None:
        with open(args.token_vocab, 'rb') as f:
            token_vocab = pickle.load(f)
        vocab_size = hparams["vocab_size"]

        path_to_glove_file = os.path.join(args.glove_path, 'glove.6B.'+str(hparams['glove_embedding_dim'])+'d.txt')

        embeddings_index = {}
        with open(path_to_glove_file) as f:
            for line in f:
                word, coefs = line.split(maxsplit=1)
                coefs = np.fromstring(coefs, "f", sep=" ")
                embeddings_index[word] = coefs

        print("Found %s word vectors." % len(embeddings_index))

        word_index = dict(enumerate(token_vocab))
        embedding_dim = hparams['glove_embedding_dim']
        hits = 0
        misses = 0

        # Prepare embedding matrix
        embedding_matrix = np.zeros((vocab_size, embedding_dim))
        for i, word in word_index.items():
            # without lowercasing: Converted 5389 words (9412 misses)
            # with lowercasing: Converted 9565 words (5236 misses)
            try:
                word = str(np.char.lower(word))
            except:
                print(word)
            embedding_vector = embeddings_index.get(word)
            if embedding_vector is not None:
                # Words not found in embedding index will be all-zeros.
                # This includes the representation for "padding" and "OOV"
                embedding_matrix[i] = embedding_vector
                hits += 1
            else:
                misses += 1
        print("Converted %d words (%d misses)" % (hits, misses))

        # now add these to the hparams, overwriting embed dim and adding one for embedding matrix
        hparams["embedding_matrix"] = embedding_matrix


    model = make_model(metrics=METRICS, hp=hparams)

    # early stopping criteria based on area under the curve: will stop if no improvement after 10 epochs
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_auc', verbose=3, patience=10, \
        mode='max', restore_best_weights=True)

    # Create a callback that saves the model's weights
    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=args.checkpoint_dir,
                                                 save_weights_only=True,
                                                 verbose=3)

    EPOCHS = hparams["epochs"]
    BATCH_SIZE = hparams["batch_size"]
    
    print(model.summary())
    # results = model.evaluate(X, y, batch_size=BATCH_SIZE, verbose=0)
    # print("Loss: {:0.4f}".format(results[0])) #0.9711

    if not args.overwrite:
        try:
            model.load_weights(args.checkpoint_dir).expect_partial() # idk why I need to have the expect_partial but error otherwise
            print('Found existing checkpoint directory, loading weights...')
        except:
            # fitting the model
            model.fit(X, weighted_y, batch_size=BATCH_SIZE, epochs=EPOCHS, callbacks = [early_stopping, cp_callback], validation_data=(dev_X, dev_y), verbose=3)
    else:
        model.fit(X, weighted_y, batch_size=BATCH_SIZE, epochs=EPOCHS, callbacks = [early_stopping, cp_callback], validation_data=(dev_X, dev_y))
    # results = model.evaluate(X, y, batch_size=BATCH_SIZE, verbose=0)
    # print("Loss: {:0.4f}".format(results[0])) #0.0568

    # if we are evaluating on the test data, aka done with development
    # it's different because we don't have gold labels for the test so we can't evaluate our predictions
    if args.real_deal:
        # load back in the processed sequences
        test_seqs = pd.read_pickle('data/seqs/test_seqs.pkl')

        # get predictions on the test data
        full_df = get_preds(model, test_X, test_seqs, istest=True)
        full_df.to_csv(args.output_file+'_test', sep=' ', index=False, header=True)
    else:
        # load back in the processed sequences
        dev_seqs = pd.read_pickle('data/seqs/dev_seqs.pkl')

        # get predictions on the dev data
        full_df = get_preds(model, dev_X, dev_seqs)
        full_df.to_csv(args.output_file, sep=' ', index=False, header=True)
    # get predictions on data, if real-deal flag is passed, it will evaluate on the test data and not run the evaluate script (it will be blind submission)
    # if not real deal, it will get preds on dev data and in bash will get passed to evaluate script
    # this file outputs a file called predictions.txt that will be passed to evaluate.py
    
if __name__ == '__main__':
    p = ap.ArgumentParser()
    p.add_argument('--train-data-path', type=str)
    p.add_argument('--dev-data-path', type=str)
    p.add_argument('--test-data-path', type=str)
    p.add_argument('--token-vocab', type=str)
    p.add_argument('--hparams-path', default=None, type=str)
    p.add_argument('--label-map-path', type=str, default=None)
    p.add_argument('--glove-path', type=str, default=None)
    p.add_argument('--output-file', type=str, default='predictions.txt')
    # p.add_argument('--train-epochs', default=100, type=int)
    # p.add_argument('--batch-size', type=int, default=32)
    p.add_argument('--checkpoint-dir', type=str, default=None)
    p.add_argument('--real-deal', action='store_true', help='if predicting on test data')
    p.add_argument('--overwrite', action='store_true', help='to overwrite checkpoint dirs')

    args = p.parse_args()
    # TODO: write this to a text file somewhere just to have on hand

    main(args)