#!usr/bin/python
import pandas as pd
import numpy as np
import copy

from collections import Counter
from keras.preprocessing.sequence import pad_sequences



# figure out the label distribution in our fixed-length texts
def get_initial_bias(padded_labels):
    all_labs = [l for lab in padded_labels for l in lab]
    label_count = Counter(all_labs)
    total_labs = len(all_labs)
    print(label_count)
    print(total_labs)

    # use this to define an initial model bias
    initial_bias=[(label_count[0]/total_labs), (label_count[1]/total_labs),
                (label_count[2]/total_labs), (label_count[3]/total_labs)]
    print('Initial bias:')
    print(initial_bias)
    return initial_bias


def downweight(onehot_weights, class_wts=[1,1,0.1,0.1]):
    # use deep copy to ensure we aren't updating original values

    onehot_weights = copy.deepcopy(onehot_weights)

    # our first-pass class weights: normal for named entities (0 and 1), down-weighted for non named entities (2 and 3)

    # apply our weights to the label lists
    for i,labs in enumerate(onehot_weights):
        for j,lablist in enumerate(labs):
            lablistaslist = lablist.tolist()
            whichismax = lablistaslist.index(max(lablistaslist))
            onehot_weights[i][j][whichismax] = class_wts[whichismax]
    return onehot_weights

# replace handles and urls with dummy token
def clean_text(orig_txt):
    txt = orig_txt.copy()
    # use a regex to replace any twitter handles with '<USER>'
    txt['token'] = txt['token'].str.replace(r'(^|[^@\w])@(\w{1,15})\b', '<USER>', regex=True)
    # use a regex to replace any urls with '<URL>'
    txt['token'] = txt['token'].str.replace(r"(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:'\".,<>?«»“”‘’]))", '<URL>', regex=True)

    return txt

# convert word tokens to integers
def token_index(tok, token_vocab, oov):
    ind = tok
    if not pd.isnull(tok):  # new since last time: deal with the empty lines which we didn't drop yet
        if tok in token_vocab:  # if token in vocabulary
            ind = token_vocab.index(tok)
        else:  # else it's OOV
            ind = oov
    return ind

# training labels: convert BIO to integers
def bio_index(bio):
    ind = bio
    if not pd.isnull(bio):  # deal with empty lines
        if bio=='B':
            ind = 0
        elif bio=='I':
            ind = 1
        elif bio=='O':
            ind = 2
    return ind

# pass a data frame through our feature extractor
def extract_features(txt_orig, istest=False, scrub=False):
    txt = txt_orig.copy()
    token_vocab = txt.token.unique().tolist()
    oov = len(token_vocab)  # OOV (out of vocabulary) token as vocab length (because that's max.index + 1)
    
    if scrub:
        txt = clean_text(txt)
    tokinds = [token_index(u, token_vocab, oov) for u in txt['token']]
    txt['token_indices'] = tokinds
    if not istest:  # can't do this with the test set
        bioints = [bio_index(b) for b in txt['bio_only']]
        txt['bio_only'] = bioints
    return txt

def tokens2sequences(txt_orig,istest=False):
    '''
    Takes panda dataframe as input, copies, and adds a sequence index based on full-stops.
    Outputs a dataframe with sequences of tokens, named entity labels, and token indices as lists.
    '''
    txt = txt_orig.copy()
    txt['sequence_num'] = 0
    seqcount = 0
    for i in txt.index:  # in each row...
        txt.loc[i,'sequence_num'] = seqcount  # set the sequence number
        if pd.isnull(txt.loc[i,'token']):  # increment sequence counter at empty lines
            seqcount += 1
    # now drop the empty lines, group by sequence number and output df of sequence lists
    txt = txt.dropna()
    if istest:  # test set doesn't have labels
        txt_seqs = txt.groupby(['sequence_num'],as_index=False)[['token', 'token_indices']].agg(lambda x: list(x))
    else:
        txt_seqs = txt.groupby(['sequence_num'],as_index=False)[['token', 'bio_only', 'token_indices']].agg(lambda x: list(x))
    return txt_seqs

def find_longest_sequence(txt,longest_seq):
    '''find the longest sequence in the dataframe'''
    for i in txt.index:
        seqlen = len(txt['token'][i])
        if seqlen > longest_seq:  # update high water mark if new longest sequence encountered
            longest_seq = seqlen
    return longest_seq

def find_seq_len(train, dev, test):
    seq_len_list = [find_longest_sequence(i, 0) for i in [train, dev, test]]
    return max(seq_len_list)


def pad(seqs, sequence_len, token_pad, label_pad, istest=False):
    pad_seqs = pad_sequences(seqs['token_indices'].tolist(), 
                                maxlen=sequence_len,
                                dtype='int32', 
                                padding='post', 
                                truncating='post', 
                                value=token_pad)
    if not istest:
        pad_labs = pad_sequences(seqs['bio_only'].tolist(), 
                                    maxlen=sequence_len,
                                    dtype='int32', 
                                    padding='post', 
                                    truncating='post', 
                                    value=label_pad)
        return pad_seqs, pad_labs
    else:
        return pad_seqs



def eval_model(model, seqs_padded, seqs):
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

    print(seqs.head())

    # use sequence number as the index and apply pandas explode to all other columns
    long = seqs.set_index('sequence_num').apply(pd.Series.explode).reset_index()
    print(long.head())

    bio_labs = [reverse_bio(b) for b in long['bio_only']]
    long['bio_only'] = bio_labs
    pred_labs = [reverse_bio(b) for b in long['prediction']]
    long['prediction'] = pred_labs

    long.head()
    print(long.prediction.value_counts())
    return long


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


 
