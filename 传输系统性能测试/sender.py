import time
import datetime
import numpy as np
import scipy.io as sio
from scipy.io.wavfile import write
import sounddevice as sd



sampling_rate = 48000
symbol_duration  = 0.1
N = sampling_rate * symbol_duration

Ts = 1/sampling_rate
preamble = '0101010101010101'

f = 4000



time_vector = datetime.today()

h = time_vector.hour
m = time_vector.minute
s = time_vector.seconde

h = h*100
m = m*100
s = s*1000

h_bin = bin(h)
m_bin = bin(m)
s_bin = bin(s)


h_bin = h_bin.zfill(16)
m_bin = m_bin.zfill(16)
s_bin = s_bin.zfill(16)

time_bin = preamble + h_bin + m_bin + s_bin


time_signal = ','.join(list(time_bin))
time_signal_len = N * len(preamble) / sampling_rate
time_t = np.arange(0, time_signal_len, Ts)
time_np_signal = np.fromstring(time_signal, dtype='int', sep=',')
time_sample = np.repeat(time_np_signal, N)
time_y = np.sin(2 * np.pi * (f + time_sample * 1000) * time_t)

write('first.wav', sampling_rate, time_y)
samplerate, data = sio.wavfile.read('first.wav')
sd.play(data, samplerate)

