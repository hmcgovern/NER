
# standard 
import pandas as pd
import numpy as np
import argparse as ap
import pickle
import os

# 3rd party
from keras.utils import to_categorical

# local 
import utils


def main(args):
    wnuttrain = args.data_dir+'/wnuttrain.txt'
    train = pd.read_table(wnuttrain, header=None, names=['token', 'label', 'bio_only', 'upos'])

    wnutdev = args.data_dir+'/wnutdev.txt'
    dev = pd.read_table(wnutdev, header=None, names=['token', 'label', 'bio_only', 'upos'])

    wnuttest = args.data_dir+'/wnuttest.txt'
    test = pd.read_table(wnuttest, header=None, names=['token', 'upos'])

    # extracting features
    train_copy = utils.extract_features(train)
    dev_copy = utils.extract_features(dev)
    test_copy = utils.extract_features(test, istest=True)

    # TODO: this will depend on whether or not it needs to be cleaned
    # for now just using pkls for development to save time
    train_seqs = utils.tokens2sequences(train_copy)
    dev_seqs = utils.tokens2sequences(dev_copy)
    test_seqs = utils.tokens2sequences(test_copy, istest=True)
    
    train_seqs.to_pickle(args.data_dir+'/seqs/train_seqs.pkl')
    dev_seqs.to_pickle(args.data_dir+'/seqs/dev_seqs.pkl')
    test_seqs.to_pickle(args.data_dir+'/seqs/test_seqs.pkl')

    # train_seqs = pd.read_pickle(args.data_dir+'/seqs/train_seqs.pkl')
    # dev_seqs = pd.read_pickle(args.data_dir+'/seqs/dev_seqs.pkl')
    # test_seqs = pd.read_pickle(args.data_dir+'/seqs/test_seqs.pkl')

    # take the longest sequence and make it the sequence length
    seq_length = utils.find_seq_len(train_seqs, dev_seqs, test_seqs)

    token_vocab = train.token.unique().tolist()
    oov = len(token_vocab)  # OOV (out of vocabulary) token as vocab length (because that's max.index + 1)
    
    with open(args.data_dir + '/token_vocab.pkl', 'wb') as f:
        pickle.dump(token_vocab, f)

    # a new dummy token index, one more than OOV
    padtok = oov+1
    print('The padding token index is %i' % padtok)
    padlab = 3
    print('The padding label index is %i' % padlab)

    train_seqs_padded, train_labs_padded = utils.pad(train_seqs, seq_length, padtok, padlab)
    dev_seqs_padded, dev_labs_padded = utils.pad(dev_seqs, seq_length, padtok, padlab)
    test_seqs_padded = utils.pad(test_seqs, seq_length, padtok, padlab, istest=True)

    # convert those labels to one-hot encoding
    n_labs = 4  # we have 3 labels: B, I, O (0, 1, 2) + the pad label 3
    train_labs_onehot = [to_categorical(i, num_classes=n_labs) for i in train_labs_padded]
    dev_labs_onehot = [to_categorical(i, num_classes=n_labs) for i in dev_labs_padded]

    X = np.array(train_seqs_padded)
    y = np.array(train_labs_onehot)

    train_pkl = {'X': X, 'y': y}

    dev_X = np.array(dev_seqs_padded)
    dev_y = np.array(dev_labs_onehot) 

    dev_pkl = {'dev_X': dev_X, 'dev_y': dev_y}

    test_X = np.array(test_seqs_padded)
    
    test_pkl = {'test_X': test_X, 'test_y': None}

    ############### writing to file ################
    with open(args.data_dir+'/padded/train.pkl', 'wb') as f:
        pickle.dump(train_pkl, f)
    with open(args.data_dir+'/padded/dev.pkl', 'wb') as f:
        pickle.dump(dev_pkl, f)
    with open(args.data_dir+'/padded/test.pkl', 'wb') as f:
        pickle.dump(test_pkl, f)

if __name__ == "__main__":
    
    p = ap.ArgumentParser() 

    p.add_argument('--output-dir', required=True, \
        help='output directory for label list, word files, and pickles')
    p.add_argument('--data-dir', required=True, \
        help='current model\'s location (generated from a bash script using Job ID)')
    p.add_argument('--clean', action='store_true', \
        help='an extra preprocessing step to scrub handles and urls from data and replace with dummy token')
    # p.add_argument('--dataset', required=True, \
    #     help='keyword for dataset to use')
    args = p.parse_args()
    
    main(args)

