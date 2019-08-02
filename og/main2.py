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
from pydub import AudioSegment
import math

full_audio = AudioSegment.from_file('../1Aero/Train_Set_1/FS_P01_dev_001.wav')
n = len(full_audio)//2
full_audio[:n+10].export('../1Aero/Train_Set_1/FS_P01_dev_001_half.wav', format='wav')

(rate,sig) = wav.read("../1Aero/Train_Set_1/FS_P01_dev_001_half.wav")
mfcc_feat = mfcc(sig,rate)
#mfcc_feat = np.append(mfcc_feat, [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
print(mfcc_feat.shape)

input('once wait...')

Tx = 13
n_freq = 1000

X = np.ndarray(shape=(90, Tx, n_freq), dtype='int64')

for i in range(0,90):
    X[i] = mfcc_feat[i*1000:(i+1)*1000].T

op = np.load('processedoutput.npy')
lenop= len(op)//2
op = op[:lenop]
op = np.reshape(op, (90, 1000, 1))
print(op.shape)

def model(input_shape):
    X_input = Input(shape = input_shape)
    # Step 1: CONV layer
    X = Conv1D(filters=196, kernel_size=13, strides=4)(X_input)
    X = BatchNormalization()(X)
    X = Activation('relu')(X)  
    X = Dropout(0.8)(X)        
    # Step 2: First GRU Layer
    X = GRU(units = 128, return_sequences = True)(X_input)
    X = Dropout(0.8)(X)      
    X = BatchNormalization()(X)
    # Step 3: Second GRU Layer
    X = GRU(units = 128, return_sequences = True)(X)
    X = Dropout(0.8)(X)                             
    X = BatchNormalization()(X)                     
    X = Dropout(0.8)(X)                             
    # Step 4: Time-distributed dense layer
    X = TimeDistributed(Dense(1, activation = "sigmoid"))(X[::7])
    model = Model(inputs = X_input, outputs = X)
    return model

model = model((Tx, n_freq))
# model.summary()
opt = Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, decay=0.01)
model.compile(loss='binary_crossentropy', optimizer=opt, metrics=["accuracy"])
model.fit(X, op, batch_size=5, epochs=200)

model.save('speech_mk1.h5')