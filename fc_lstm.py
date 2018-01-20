# -*- coding: utf-8 -*-
"""
Created on Fri Jan 19 10:31:44 2018

@author: mritunjay
"""
import numpy as np,pandas as pd
import importlib
from pydub import AudioSegment
importlib.import_module('music_main')
import librosa
import scipy.fftpack as sf
from scipy.io import wavfile

#sampFreq, snd = wavfile.read(PATH+str(26)+'.wav')

len_x = int(input_slice/dimensions)

for i in range(MIN,MIN+20):
    x_file , sr_file = librosa.load(PATH+str(i)+'.wav',sr=sampling_rate)
    x_file = np.fft.fft(x_file)
    x_file = x_file[:input_slice]
    x_file = np.reshape(x_file,(len_x,dimensions))
    x_file = np.append(x_file.real,x_file.imag,axis=1)
    final_matrix.append(x_file)

sample_matrix = np.array(final_matrix)
nb_samples = sample_matrix.shape[0]*sample_matrix.shape[1]
features = sample_matrix.shape[2]
sample_matrix = sample_matrix.reshape((nb_samples,features))
#sample_matrix.dump('new_array_20000')

#sample_matrix = np.load('new_array')

from keras.models import Sequential
from keras.layers import Dense,TimeDistributed
from keras.layers import LSTM
from keras.layers import Dropout


X_train = sample_matrix[:-TIMESTEPS]
Y_train = sample_matrix[TIMESTEPS:]
samples = int(X_train.shape[0]/TIMESTEPS)
X_train = np.reshape(X_train,(samples,TIMESTEPS,features))
Y_train = np.reshape(Y_train,(samples,TIMESTEPS,features))


reg = Sequential()
reg.add(TimeDistributed(Dense(2048,activation='softmax'),input_shape=(TIMESTEPS,features)))
reg.add(Dropout(0.2))
reg.add(LSTM(units = 50, return_sequences = True))
# Adding the output layer
reg.add(TimeDistributed(Dense(units = features)))
reg.compile(optimizer = 'rmsprop', loss = 'mean_squared_error')


reg.fit(X_train, Y_train, epochs = epochs, batch_size = bsize)
reg.save_weights('music_generation')



new = []
x_file , sr_file = librosa.load(PATH_T+str(0)+'.wav',sr=sampling_rate)
x_file = np.fft.fft(x_file)
x_file = x_file[:input_slice]
x_file = np.reshape(x_file,(len_x,dimensions))
x_file = np.append(x_file.real,x_file.imag,axis=1)
new.append(x_file)
new = np.array(new)
new = new.reshape(new.shape[0]*new.shape[1],2*dimensions)
new_dim = int(new.shape[0]/TIMESTEPS)
pred_samples = new_dim*TIMESTEPS
new = new[:pred_samples]
new = new.reshape((new_dim,TIMESTEPS,2*dimensions))
preds = reg.predict(new)


pred_arr = preds.reshape((new.shape[0]*new.shape[1],new.shape[2]))
pred_real = pred_arr[:,:dimensions]
pred_real = pred_real.reshape((pred_real.shape[0]*pred_real.shape[1]))
pred_imag = pred_arr[:,dimensions:]
pred_imag = pred_imag.reshape((pred_imag.shape[0]*pred_imag.shape[1]))

pred_complex = pred_real + 1j*pred_imag
pred_ifft = sf.irfft(pred_complex )

pred_arr = pred_ifft.real





librosa.output.write_wav(PATH_T+'pred5'+'.wav',pred_arr,sampling_rate)






