# -*- coding: utf-8 -*-
"""
Created on Fri Jan 19 10:31:44 2018

@author: mritunjay
"""
import numpy as np,pandas as pd
import importlib
import matplotlib.pyplot as plt
from pydub import AudioSegment
importlib.import_module('music_main')
import librosa
import scipy.fftpack as sf
from scipy.io import wavfile
from IPython.display import Audio
from sklearn.preprocessing import StandardScaler , MinMaxScaler
#sampFreq, snd = wavfile.read(PATH+str(26)+'.wav')

len_x = int(input_slice/dimensions)

for i in range(MIN,MIN+10):
    sr_file , x_file= wavfile.read(PATH+str(i)+'.wav')
    x_file = np.fft.fft(x_file)
    x_file = x_file[:input_slice]
    x_file = np.reshape(x_file,(len_x,dimensions))
    x_file = np.append(x_file.real,x_file.imag,axis=1)
    final_matrix.append(x_file)


sample_matrix = np.array(final_matrix)
nb_samples = sample_matrix.shape[0]*sample_matrix.shape[1]
features = sample_matrix.shape[2]
sample_matrix = sample_matrix.reshape((nb_samples,features))
#sample_matrix.dump('new_array_22050_scipy')
sample_matrix = np.load('new_array_22050_scipy')

from keras.models import Sequential
from keras.layers import Dense,TimeDistributed
from keras.layers import LSTM
from keras.layers import Dropout

features = 2*dimensions
X_train = sample_matrix[:-TIMESTEPS]
Y_train = sample_matrix[TIMESTEPS:]
samples = int(X_train.shape[0]/TIMESTEPS)

sc = MinMaxScaler(feature_range=(-1 , 1))
ab = sc.fit_transform(X_train)
cd = sc.fit_transform(Y_train)

X_train = np.reshape(ab,(samples,TIMESTEPS,features))
Y_train = np.reshape(cd,(samples,TIMESTEPS,features))


#training model
reg = Sequential()
reg.add(TimeDistributed(Dense(2048,activation='softmax'),input_shape=(TIMESTEPS,features)))
reg.add(Dropout(0.2))
reg.add(LSTM(units = 50, return_sequences = True))
# Adding the output layer
reg.add(TimeDistributed(Dense(units = features)))
reg.compile(optimizer = 'rmsprop', loss = 'mean_squared_error')


reg.fit(X_train, Y_train, epochs = epochs, batch_size = bsize)
reg.load_weights('music_generation_22050_scipy')
#reg.load_weights('')


new = []
x_file , sr_file = wavfile.read(PATH_T+str(3)+'.wav')
x_file = np.fft.fft(sr_file)
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


def to_time_domain(preds):
    pred_arr = preds
    pred_real = pred_arr[:,:dimensions]
    pred_real = pred_real.reshape((pred_real.shape[0]*pred_real.shape[1]))
    pred_imag = pred_arr[:,dimensions:]
    pred_imag = pred_imag.reshape((pred_imag.shape[0]*pred_imag.shape[1]))
    pred_complex = pred_real + 1j*pred_imag
    pred_ifft = sf.ifft(pred_complex)
    pred_int = pred_ifft.astype('float32')
    pred_int = np.array(pred_int)
    return pred_int
    




output = []
seed  = new[0]
seed = seed.reshape((1,seed.shape[0],seed.shape[1]))
for it in range(TIMESTEPS):
    seedNew = reg.predict(seed)
    
    if it==0:
        for i in range(seedNew.shape[1]):
            output.append(seedNew[0][i].copy())
    else:
        output.append(seedNew[0][seedNew.shape[1]-1].copy())
    
    newSeq = seedNew[0][seedNew.shape[1]-1]
    newSeq = newSeq.reshape(1,1,newSeq.shape[0])
    seed = np.concatenate((seed,newSeq),axis=1)
    seed = seed[:,1:,:]

predicted_output = np.array(output)
pred_output_denorm = sc.inverse_transform(predicted_output)
pred_x = to_time_domain(pred_output_denorm)
pred_x_librosa = to_time_domain(predicted_output)
pred_x_librosa = pred_x_librosa*60
wavfile.write(PATH_T+'pred8.wav',22050,np.array(va))
librosa.output.write_wav(PATH_T+'pred9.wav',pred_x_librosa,22050)
Audio(PATH_T+'pred7'+'.wav',autoplay = True)



bt , pred_X = wavfile.read(PATH_T+'pred8.wav')

plt.plot(sr_file)
plt.plot(pred_x_librosa)

wavfile.write(PATH_T+'pred9.wav',22050,var[1])

va = [x for x in pred_x_librosa if x>=0.001 or x<=-0.001]
