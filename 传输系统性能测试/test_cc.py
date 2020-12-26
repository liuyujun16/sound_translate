import operator
from scipy import signal
from datetime import datetime



import sys
import wave
from threading import Thread

from numpy import mean,zeros,arange,fromstring,repeat,sin,pi,corrcoef,zeros,reshape,mat
from scipy.fftpack import fft,ifft
import scipy.io as sio
from PyQt5.QtCore import QThread
from PyQt5.QtWidgets import *
from PyQt5 import uic
import pyaudio
from scipy.io.wavfile import write
import sounddevice as sd
import time
import soundfile as sf
import math
form_class = uic.loadUiType("gui_for_translate.ui")[0]



sampling_rate = 48000
symbol_duration  = 0.025
N = sampling_rate * symbol_duration
print(N)
Ts = 1/sampling_rate
preamble = '0101010101010101'

f = 4000
FILENAME = 'save.tmp'

preamble_signal = ','.join(list(preamble))
preamble_time_signal = N * len(preamble) / sampling_rate
preamble_t = arange(0, preamble_time_signal, Ts)
preamble_np_signal = fromstring(preamble_signal, dtype='int', sep=',')
preamble_sample = repeat(preamble_np_signal, N)
preamble_y = sin(2 * pi * (f + preamble_sample * 1000) * preamble_t)



class ThreadClass(QThread):
    def __init__(self):
        super().__init__()

    def run(self):


        global flag
        flag = 0
        FORMAT = pyaudio.paInt16
        CHANNELS = 1
        frames = []

        p = pyaudio.PyAudio()
        stream = p.open(format=FORMAT,
                        channels=CHANNELS,
                        rate=sampling_rate,
                        input=True,
                        output=True,
                        frames_per_buffer=int(N))

        while 1:
            if flag == 0:
                data = stream.read(int(N))
                frames.append(data)
            else:
                break
        stream.stop_stream()
        stream.close()
        p.terminate()
        wf = wave.open(FILENAME, "wb")
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(p.get_sample_size(FORMAT))
        wf.setframerate(sampling_rate)
        wf.writeframes(b"".join(frames))
        wf.close()



class WindowClass(QMainWindow, form_class):
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.encode_button.clicked.connect(self.encode)
        #self.decode_button.clicked.connect(self.decode)
        self.start_record.clicked.connect(self.recording)
        self.stop_record.clicked.connect(self.stop_recording)
        self.threadclass = ThreadClass()

    def stop_recording(self):
        global flag
        flag = 1
        print('stop!!!')
        time.sleep(1)
        self.decode()


    def recording(self):
        print('start')
        self.threadclass.start()

    def encode(self):
        time_vector = datetime.today()

        h = time_vector.hour
        m = time_vector.minute
        s = time_vector.second

        h = h * 100
        m = m * 100
        s = s * 1000

        h_bin = format(h,'b')
        m_bin = format(m,'b')
        s_bin = format(s,'b')

        h_bin = h_bin.zfill(16)
        m_bin = m_bin.zfill(16)
        s_bin = s_bin.zfill(16)

        time_bin = preamble + h_bin + m_bin + s_bin

        time_signal = ','.join(list(time_bin))
        print(time_signal)
        time_signal_len = N * len(time_bin) / sampling_rate
        print(time_signal_len)
        time_t = arange(0, time_signal_len, Ts)
        print(time_t)

        time_np_signal = fromstring(time_signal, dtype='int', sep=',')
        print(time_np_signal)

        time_sample = repeat(time_np_signal, N)
        print(time_sample)
        time_y = sin(2 * pi * (f + time_sample * 1000) * time_t)
        print(time_y)

        write('first.wav', sampling_rate, time_y)
        samplerate, data = sio.wavfile.read('first.wav')
        sd.play(data, samplerate)

    def decode(self):


        time_vector = datetime.today()

        start_h = time_vector.hour
        start_m = time_vector.minute
        start_s = time_vector.second



        bandpass1 = 3000
        bandpass2 = 7000
        read_signal, fs = sf.read(FILENAME)

        wn1 = 2.0 * bandpass1 / sampling_rate
        wn2 = 2.0 * bandpass2 / sampling_rate
        b, a = signal.butter(8, [wn1, wn2], 'bandpass')  # 범위를 조금 더 넓게 해야할
        filtedData = signal.filtfilt(b, a, read_signal)  # data为要过滤的信号

        current_length = len(filtedData)
        print(current_length)
        print(len(preamble_y))
        current_data = filtedData
        corr = []
        corr_index = dict()
        once_check = 0
        for i in range(current_length - len(preamble_y)):
            corr.append(corrcoef(current_data[i:i + len(preamble_y)], preamble_y)[0, 1])
            if (once_check + 76800) == i and once_check != 0:
                break

            if corr[i] > 0.5:
                if once_check == 0:
                    once_check = i
                    print('once_check', once_check)

                corr_index[i] = corr[i]

        try:
            flag = max(corr_index.items(), key=operator.itemgetter(1))[0]
        except:
            print('decode 结束')

        print(flag)
        current_data = current_data[flag + len(preamble_y):]
        current_length = len(current_data)
        sympling_length = 400
        data_index = round(f / fs * sympling_length)
        data_fft = []
        for i in range(current_length - sympling_length):
            abb = fft(current_data[i:i + sympling_length])
            y = abs(abb)
            data_fft[i] = max(y[data_index - 2:data_index + 2])
        data_fft_temp = data_fft
        for i in range(5, current_length - 5):
            data_fft_temp[i] = mean(data_fft[i - 5:i + 5])
        data_fft = data_fft_temp

        peak_index = []
        for i in range(800, current_length - 800):
            if data_fft[i] > 40 and data_fft[i] == max(data_fft[i - 800: i + 800]):
                peak_index.append(i)

        message_bin = zeros(54);
        for i in range(len(peak_index)):
            message_bin[math.ceil(peak_index / 4800)] = 1

        for i in range(6):
            if message_bin[i] == 1:
                last_one_index = 1

        real_message_start = last_one_index + 1

        real_message_bin = message_bin[real_message_start:real_message_start + 47]

        real_message_bin = reshape(mat(real_message_bin).H, mat([16, 3]).H);
        message_vector = int(real_message_bin, 2)
        bias = 0
        delta_time = start_h * 3600000 + start_m * 60000 + start_s * 1000 + flag / 48 - message_vector[1] * 36000 - \
                     message_vector[2] * 600 - message_vector[3] - bias;
        distance = delta_time * 0.34
        print(distance)
if __name__ == "__main__":
    app = QApplication(sys.argv)

    myWindow = WindowClass()

    myWindow.show()

    app.exec_()


