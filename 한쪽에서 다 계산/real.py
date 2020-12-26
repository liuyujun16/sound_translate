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





form_class = uic.loadUiType("real.ui")[0]

fs = 48000  # Record at 44100 samples per second
chunk = 1200  # Record in chunks of 1024 samples
seconds = 3
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


class ThreadClass(QThread):
    def __init__(self):
        super().__init__()

    def run(self):
        sample_format = pyaudio.paInt16  # 16 bits per sample
        channels = 1

        p = pyaudio.PyAudio()  # Create an interface to PortAudio

        print('Recording')

        stream = p.open(format=sample_format,
                        channels=channels,
                        rate=fs,
                        frames_per_buffer=chunk,
                        input=True)

        frames = []  # Initialize array to store frames

        # Store data in chunks for 3 seconds
        for i in range(0, int(fs / chunk * seconds)):
            data = stream.read(chunk)
            frames.append(data)

        # Stop and close the stream
        stream.stop_stream()
        stream.close()
        # Terminate the PortAudio interface
        p.terminate()
        wf = wave.open(filename_1, "wb")
        wf.setnchannels(channels)
        wf.setsampwidth(p.get_sample_size(sample_format))
        wf.setframerate(fs)
        wf.writeframes(b"".join(frames))
        wf.close()


        print('Finished recording')
        end = time.time()
        print(end)

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

        plt.plot(filtedData)
        plt.grid()
        plt.show()
        current_length = len(filtedData)
        current_data = filtedData
        once_check = 0
        corr_index = dict()

        print('finding preamble')
        print('current_data length', len(current_data))
        for i in range(current_length - len(preamble_y)):
            corr = np.corrcoef(current_data[i:i + len(preamble_y)], preamble_y)[0, 1]
            if once_check + 24000 == i and once_check != 0:
                print('corr 찾는거 ')
                break

            if corr > 0.7:
                if once_check == 0:
                    once_check = i
                    print('once_check', once_check)

                corr_index[i] = corr
                print(i,'번째친구의',corr)

        try:
            flag = max(corr_index.items(), key=operator.itemgetter(1))[0]
        except:
            print('decode 结束')
            return
        print(flag)
        minus_time = (current_length - flag)/fs
        time_a = end-minus_time
        print(time_a)



        time.sleep(10)

        sample_format = pyaudio.paInt16  # 16 bits per sample
        channels = 1

        p = pyaudio.PyAudio()  # Create an interface to PortAudio

        print('Recording')

        stream = p.open(format=sample_format,
                        channels=channels,
                        rate=fs,
                        frames_per_buffer=chunk,
                        input=True)

        frames = []  # Initialize array to store frames

        # Store data in chunks for 3 seconds
        for i in range(0, int(fs / chunk * seconds)):
            data = stream.read(chunk)
            frames.append(data)

        # Stop and close the stream
        stream.stop_stream()
        stream.close()
        # Terminate the PortAudio interface
        p.terminate()
        p.terminate()
        wf = wave.open(filename_2, "wb")
        wf.setnchannels(channels)
        wf.setsampwidth(p.get_sample_size(sample_format))
        wf.setframerate(fs)
        wf.writeframes(b"".join(frames))
        wf.close()

        print('Finished recording')
        end = time.time()
        print(end)

        fig = plt.figure(figsize=(10, 6))

        bandpass1 = 3000
        bandpass2 = 7000
        read_signal, samplerate = sf.read(filename_2)
        plt.subplot(411)

        plt.plot(read_signal)
        plt.grid()

        wn1 = 2.0 * bandpass1 / samplerate
        wn2 = 2.0 * bandpass2 / samplerate
        b, a = signal.butter(8, [wn1, wn2], 'bandpass')  # 범위를 조금 더 넓게 해야할
        filtedData = signal.filtfilt(b, a, read_signal)  # data为要过滤的信号
        plt.subplot(412)

        plt.plot(filtedData)
        plt.grid()
        plt.show()
        current_length = len(filtedData)
        current_data = filtedData
        once_check = 0
        corr_index = dict()

        print('finding preamble')
        print('current_data length', len(current_data))
        for i in range(current_length - len(preamble_y)):
            corr = np.corrcoef(current_data[i:i + len(preamble_y)], preamble_y)[0, 1]
            if once_check + 24000 == i and once_check != 0:
                print('corr 찾는거 ')
                break

            if corr > 0.7:
                if once_check == 0:
                    once_check = i
                    print('once_check', once_check)

                corr_index[i] = corr
                print(i, '번째친구의', corr)

        try:
            flag = max(corr_index.items(), key=operator.itemgetter(1))[0]
        except:
            print('decode 结束')
        print(flag)
        minus_time = (current_length - flag) / fs
        time_b = end - minus_time
        print(time_b)


        print('last_result ',time_b-time_a)



        return


class WindowClass(QMainWindow, form_class):
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.start_button.clicked.connect(self.start_measure)
        self.threadclass = ThreadClass()

    def start_measure(self):
        self.threadclass.start()
        time.sleep(1)
        sd.play(data, samplerate)
        return




if __name__ == "__main__":
    app = QApplication(sys.argv)

    myWindow = WindowClass()

    myWindow.show()

    app.exec_()
