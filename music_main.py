# -*- coding: utf-8 -*-
"""
Created on Fri Jan 19 09:58:39 2018

@author: mritunjay
"""

import numpy as np

TIMESTEPS = 30
PATH = "music_train/rock_"
PATH_T = "music_test/rock_"
MIN = 21
MAX = 100
final_matrix = []
sample_matrix = []
sampling_rate = 16000
input_slice = 600000
dimensions = 4000
epochs = 100
bsize = 32


from pydub import AudioSegment

for i in range(MIN,MAX):
    song = AudioSegment.from_file('genres/rock/rock.000'+str(i)+'.au')
    song.export('music_train/rock_'+str(i)+'.wav',format='wav')


for i in range(10):
    song = AudioSegment.from_file('genres/rock/rock.0000'+str(i)+'.au')
    song.export('music_test/rock_'+str(i)+'.wav',format='wav')
    

import librosa.core as lc

val = []
for i in range(MIN,MAX):
    dur = lc.get_duration(filename='music_train/rock_'+str(i)+'.wav')
    val.append(dur)

val = np.array(val)
in_a_song = int(val.max()/10)
