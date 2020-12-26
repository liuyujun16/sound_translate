from PyQt5.QtWidgets import *
from PyQt5 import uic
from PyQt5.QtCore import QThread
from scipy.io.wavfile import write
from scipy import signal

import operator
import scipy.io as sio
import sys
import pyaudio
import wave
import numpy as np
import sounddevice as sd
import soundfile as sf
import matplotlib.pyplot as plt
import time



fs = 48000  # Record at 44100 samples per second
chunk = 1200  # Record in chunks of 1024 samples
seconds = 30
filename_1 = "output_1.wav"
filename_2 = "output_2.wav"
preamble = '01010101010101010101'

f = 4000
Ts = 1 / fs
symbol_duration = 0.025
N = fs * symbol_duration

preamble_signal = ','.join(list(preamble))
preamble_time_signal = N * len(preamble) / fs
preamble_t = np.arange(0, preamble_time_signal, Ts)
preamble_np_signal = np.fromstring(preamble_signal, dtype='int', sep=',')
preamble_sample = np.repeat(preamble_np_signal, N)
preamble_y = np.sin(2 * np.pi * (f + preamble_sample * 2000) * preamble_t)
write('first.wav', fs, preamble_y)
samplerate, data = sio.wavfile.read('first.wav')







fig = plt.figure(figsize=(10, 6))

bandpass1 = 3000
bandpass2 = 7000
read_signal, samplerate = sf.read(filename_1)
plt.subplot(411)

plt.plot(read_signal)
plt.grid()

wn1 = 2.0 * bandpass1 / samplerate
wn2 = 2.0 * bandpass2 / samplerate
b, a = signal.butter(8, [wn1, wn2], 'bandpass')  # 범위를 조금 더 넓게 해야할
filtedData = signal.filtfilt(b, a, read_signal)  # data为要过滤的信号
plt.subplot(412)
current_length = len(filtedData)
current_data = filtedData

plt.plot(filtedData)
plt.grid()
plt.show()
store_index = []
sum = 0
while 1:
    once_check = 0
    corr = []
    corr_index = dict()

    print('finding preamble')
    print('current_data length', len(current_data))
    for i in range(current_length - len(preamble_y)):
        corr.append(np.corrcoef(current_data[i:i + len(preamble_y)], preamble_y)[0, 1])
        # if once_check + 50000 == i and once_check != 0:
        #     print('corr 찾는거 ')
        #     break
        if i == 240000-len(preamble_y):
            break
        if corr[i] > 0.60:
            if once_check == 0:
                once_check = i
                print('once_check', once_check)
            print(i,'번째 친구의 corr',corr[i])
            corr_index[i] = corr[i]

    try:
        flag = max(corr_index.items(), key=operator.itemgetter(1))[0]
    except:
        print('decode 结束')
        break
    print(flag)
    store_index.append(flag)
    current_data = current_data[240000:]
    current_length = len(current_data)


print(store_index)
result_index = []
# for i in range(len(store_index)):
#     sum = 0
#     for j in range(i+1):
#         sum += store_index[j]
#     result_index.append(sum + i * len(preamble_y))
#
# print(result_index)
result_index.append( store_index[0])
result_index.append(240000+store_index[1])

print('aaaaa',result_index)


temp = []
for ele in result_index:
    temp.append(ele / 48000)

print('ta1:',temp[0])
print('ta3:',temp[1])

total = temp[1] - temp[0]

print('total:', total)

print('total:', total * 170)