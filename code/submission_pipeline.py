## EEG Detection Kaggle
## https://www.kaggle.com/c/grasp-and-lift-eeg-detection

## Robert Dadashi-Tazehozi
## Final ranking: 46/379

import csv
import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout, TimeDistributedDense
from keras.layers.recurrent import LSTM, GRU
from keras.datasets.data_utils import get_file
from keras.callbacks import ModelCheckpoint, EarlyStopping
import time
import sklearn.metrics
from glob import glob
from sklearn.preprocessing import StandardScaler
import pickle
from multiprocessing import Pool
import sys
import os
from scipy.signal import lfilter, butter

## Model metadata 
class Model:
    def __init__(self, n_id, window_size, subsample, overlap):
        self.window_size = window_size
        self.subsample = subsample
        self.overlap = overlap
        self.n_id = n_id
        self.roc_auc = np.zeros(shape=[12, 6])
        
n_id = 7
dirname = 'model_'+str(n_id)
os.makedirs(dirname)
subsample = 10
window_size = 120
n_in = 32
overlap = 10
model_infos = Model(n_id=n_id,
                    window_size=window_size,
                    subsample=subsample,
                    overlap=overlap)
pickle.dump(model_infos, open(dirname+'/'+dirname+'.p','wb'))


def prepare_data_train(fname):
    data = pd.read_csv(fname)
    events_fname = fname.replace('_data','_events')
    labels= pd.read_csv(events_fname)
    clean=data.drop(['id' ], axis=1)
    labels=labels.drop(['id' ], axis=1)
    return  clean,labels


def pad_sequence(X, n_pad, dim):
    x_zer = np.array([0 for i in range(dim)])
    x_zeros = np.array([x_zer for i in range(n_pad)])
    return np.concatenate((x_zeros, X))

# The data is scaled, filter, and reorganized as overlapping windows
def build_feature_data(X, window_size, subsample, overlap, dim):
    X_pad = pad_sequence(X,
                         n_pad=window_size*subsample,
                         dim=dim)
    X_r = X_pad[::subsample,:]    
    X_n = np.zeros(shape=X_r.shape)
    fs = 500/subsample
    lowcut = 0.1
    highcut = 3
    for i in range(dim):
        X_n[:,i] = butter_bandpass_filter(X_r[:,i], lowcut, highcut, fs, order=2)
    X_f = []
    length = len(X_r)

    for k in range(window_size, length, overlap):
        X_f.append(X_n[k-window_size:k])
    return np.array(X_f)


def build_label_data(X, window_size, subsample, overlap, dim):
    X_pad = pad_sequence(X, n_pad=window_size*subsample, dim=dim)
    X_r = X_pad[::subsample,:]    
    X_f = []  
    length = len(X_r)
    
    for k in range(window_size, length, overlap):
        X_f.append(X_r[k])    
    return np.array(X_f)

## The architecture of the model is inspired by the Baidu Deep Speech model:
## http://arxiv.org/pdf/1412.5567v2.pdf
def build_classifier():
    
    n_in = 32
    n_hidden = 64
    n_out = 1
    drop = 0.5
    
    model = Sequential()
    model.add(TimeDistributedDense(n_in, n_hidden))
    model.add(Dropout(drop))
    model.add(TimeDistributedDense(n_hidden, n_hidden))
    model.add(Dropout(drop))
    model.add(GRU(n_hidden, n_hidden, activation='tanh', inner_activation='sigmoid'))
    model.add(Dropout(drop))
    model.add(Dense(n_hidden, n_out))
    model.add(Activation('sigmoid'))

    model.compile(loss='binary_crossentropy', optimizer='adam', class_mode='binary')    
    return model    


## The scaler is build based on the training data
def make_scaler(subject):
    
    raw = []
    fnames =  glob('../data/subj%d_series[1-7]_data.csv' % (subject))
    for fname in fnames:
        data, _ = prepare_data_train(fname)
        raw.append(data)
    X = pd.concat(raw)
    X = np.asarray(X.astype(float))
    
    scaler = StandardScaler()
    scaler.fit_transform(X)
    return scaler


## A left filter is built to cut the signals to low frequencies
def butter_bandpass(lowcut, highcut, fs, order=5):
    
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y

subjects = list(range(1,13))
cols = ['HandStart','FirstDigitTouch',
        'BothStartLoadPhase','LiftOff',
        'Replace','BothReleased']


## The hyperparameters of the model are optimized for each task and each subject
def process_subject(subject):    

    sys.stdout = open(dirname+"/subject_"+str(subject)+".txt", "w")
    scaler = make_scaler(subject)

    ## Creating training data
    y_raw= []
    raw = []
    fnames =  glob('../data/subj%d_series[1-7]_data.csv' % (subject))
    for fname in fnames:
        X, labels=prepare_data_train(fname)
        X = np.asarray(X.astype(float))
        X = scaler.transform(X)
        X = build_feature_data(X, window_size=window_size, subsample=subsample, overlap=overlap, dim=32)
        labels = np.asarray(labels.astype(float))
        labels = build_label_data(labels, window_size=window_size, subsample=subsample, overlap=overlap, dim=6)  
        raw.append(X)
        y_raw.append(labels)
    X_train = np.concatenate(np.array(raw))
    y_train = np.concatenate(np.array(y_raw))

    ## Creating validating data
    y_raw= []
    raw = []
    fnames =  glob('../data/subj%d_series8_data.csv' % (subject))
    for fname in fnames:
        X, labels=prepare_data_train(fname)
        X = np.asarray(X.astype(float))
        X = scaler.transform(X)
        X = build_feature_data(X, window_size=window_size, subsample=subsample, overlap=1, dim=32)
        labels = np.asarray(labels.astype(float))
        labels = build_label_data(labels, window_size=window_size, subsample=subsample, overlap=1, dim=6)  
        raw.append(X)
        y_raw.append(labels)
    X_val = np.concatenate(np.array(raw))
    y_val = np.concatenate(np.array(y_raw))

    ## Creating testing data
    fnames =  glob('../data/test/subj%d_series*_data.csv' % (subject))
    X_test = []
    idx=[]
    for fname in fnames:
        X = pd.read_csv(fname)
        idx.append(np.array(X['id']))
        X = X.drop(['id'], axis=1)
        X =np.asarray(X.astype(float))
        X = scaler.transform(X)
        X = build_feature_data(X, window_size=window_size, subsample=subsample, overlap=1, dim=32)
        X_test.append(X)
    ids=np.concatenate(idx)
    length = len(ids)
    X_test = np.concatenate(np.array(X_test))

    pred_test = np.empty((length,6))
    for i in range(6):
        y_train_ = y_train[:,i]
        y_val_ = y_val[:, i]

        print('Train subject %d, class %s' % (subject, cols[i]))        
        model = build_classifier()
        stopper = EarlyStopping(monitor='val_loss', patience=15, verbose=0)
        checkpointer = ModelCheckpoint(filepath=dirname+"/weights_subj_"+str(subject)+"_"+cols[i]+".hdf5",
                                       verbose=0,
                                       save_best_only=True)  
        model.fit(X_train, y_train_, batch_size=256, nb_epoch=40, 
                  validation_data=(X_val, y_val_), shuffle=True,
                  show_accuracy=True, verbose=1, callbacks=[checkpointer, stopper])        
        model.load_weights(dirname+"/weights_subj_"+str(subject)+"_"+cols[i]+".hdf5")
        p = model.predict(X_val)[:,0]
        score = sklearn.metrics.roc_auc_score(y_val_, p)
        print(score)
        model_infos = pickle.load(open(dirname+'/'+dirname+'.p','rb'))
        model_infos.roc_auc[subject-1, i] = score
        pickle.dump(model_infos, open(dirname+'/'+dirname+'.p','wb'))

        p = model.predict(X_test)[:,0]
        P = []
        for i_ in range(X_test.shape[0]):
            for j_ in range(subsample):
                P.append(p[i_])
        P = P[:length]  
        pred_test[:,i] = P 
        
    return pred_test, ids

## Write the submission file
p = Pool(12)
res = p.map(process_subject, subjects)
pred_tot = [elt[0] for elt in res]
ids_tot = [elt[1] for elt in res]
submission_file = dirname+'/submission.csv'
submission = pd.DataFrame(index=np.concatenate(ids_tot),
                          columns=cols,
                          data=np.concatenate(pred_tot))
submission.to_csv(submission_file,index_label='id',float_format='%.5f')
