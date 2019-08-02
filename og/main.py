import numpy as np
from pydub import AudioSegment
import random
import sys
import io
import os
import glob
import matplotlib.pyplot as plt
from td_utils import *
from scipy.io import wavfile
from python_speech_features import mfcc
from python_speech_features import logfbank
import scipy.io.wavfile as wav

from keras.callbacks import ModelCheckpoint
from keras.models import Model, load_model, Sequential
from keras.layers import Dense, Activation, Dropout, Input, Masking, TimeDistributed, LSTM, Conv1D
from keras.layers import GRU, Bidirectional, BatchNormalization, Reshape
from keras.optimizers import Adam

#(rate,sig) = wav.read("/home/rohanposeidon/Documents/Honeywell Hackathon/1Aero/Train_Set_1/FS_P01_dev_001.wav")
#mfcc_feat = mfcc(sig,rate)
#print(mfcc_feat.sh,ape)

from pydub import AudioSegment

Tx = 1998 # 2000 # for 20 s sample
n_freq = 101
# Ty = 1375

X = np.ndarray(shape=(90, Tx, n_freq), dtype='int64')

for i in range(0,90):
    y = graph_spectrogram('../1/output{}.wav'.format(i))
    X[i] = y.T
print(X.shape)

op = np.load('processedoutput.npy')
op = np.reshape(op, (90, 2000, 1))
print(op.shape)

def model(input_shape):
    X_input = Input(shape = input_shape)
    # Step 1: CONV layer
    X = Conv1D(filters=196, kernel_size=15, strides=4)(X_input)
    X = BatchNormalization()(X)
    X = Activation('relu')(X)  
    X = Dropout(0.8)(X)        
    # Step 2: First GRU Layer
    X = GRU(units = 128, return_sequences = True)(X)
    X = Dropout(0.8)(X)      
    X = BatchNormalization()(X)
    # Step 3: Second GRU Layer
    X = GRU(units = 128, return_sequences = True)(X)
    X = Dropout(0.8)(X)                             
    X = BatchNormalization()(X)                     
    X = Dropout(0.8)(X)                             
    # Step 4: Time-distributed dense layer
    X = TimeDistributed(Dense(1, activation = "sigmoid"))(X[::4])
    model = Model(inputs = X_input, outputs = X)
    return model

model = model((Tx, n_freq))
# model.summary()
opt = Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, decay=0.01)
model.compile(loss='binary_crossentropy', optimizer=opt, metrics=["accuracy"])
model.fit(X, op, batch_size=5, epochs=200)

model.save('speech_mk1.h5')