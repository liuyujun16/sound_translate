from PyQt5.QtWidgets import *
from PyQt5 import uic
from PyQt5.QtCore import QThread
from scipy.io.wavfile import write
from scipy import signal
from scipy.signal import chirp
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

import math
from typing import Callable, Dict, List

import numpy as np
from librosa.core import time_to_samples, tone
from scipy.signal import correlate, find_peaks, hilbert

form_class = uic.loadUiType("real.ui")[0]

fs = 44100  # Record at 44100 samples per second
chunk = 2205  # Record in chunks of 1024 samples
seconds = 12
filename_1 = "output_1.wav"



arr = np.array([])

ttime = 0.05  # 持续时间
x = np.arange(0, 0.005, 1.0 / fs)
sine = np.sin(2 * np.pi * 2000 * x + (np.pi / 2))
arr = np.append(arr, sine)
t = np.arange(0, ttime, 1.0 / fs)
arr = np.append(arr, chirp(t, 2000, 6000, ttime, method='linear', phi=0, vertex_zero=True))

write('first.wav', fs, arr)
samplerate, data = sio.wavfile.read('first.wav')


target_hz = 2000
duration_ms = 50

detect = False

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
        end = time.time()
        print(end)
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
        aa = WindowClass
        aa.start_decoding(self)

class WindowClass(QMainWindow, form_class):
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.start_button.clicked.connect(self.start_measure)
        self.decode_button.clicked.connect(self.start_decoding)
        self.threadclass = ThreadClass()

    def start_measure(self):
        self.threadclass.start()
        time.sleep(1)
        sd.play(data, samplerate)




    def start_decoding(self):

        read_signal, samplerate = sf.read(filename_1)
        signal = tone(2000, 44100, duration=duration_ms / 1000.0)
        print(len(read_signal))
        result = []
        fig = plt.figure(figsize=(10, 6))
        plt.plot(read_signal)
        plt.grid()
        plt.show()

        length_file = int(len(read_signal)/20)
        print(length_file)
        for i in range(10):

        # find onset, this differs from the description in the paper which uses a sharpness and peak finding algorithm
            correlation = correlate(read_signal[i*length_file:(i+1)*length_file], signal, mode='valid', method='fft')
            print('from:',i*length_file,'to:',(i+1)*length_file)
            envelope = np.abs(hilbert(correlation))
            max_correlation = np.max(correlation)
            peaks, _ = find_peaks(envelope)
            filtered_peaks = peaks[(peaks < len(read_signal)).nonzero()]
            peaks = peaks[(envelope[filtered_peaks] > .85 * max_correlation).nonzero()]
            ratio = (np.max(signal) / np.max(correlation))
            correlation *= ratio
            envelope *= ratio

            if len(peaks) == 0:
                print('done!')

            else:
                print(peaks[0])
                result.append(peaks[0]+i*length_file)


        temp = result[9]+len(data)*3

        print(result[9]+len(data))
        correlation = correlate(read_signal[temp:temp+50000], signal, mode='valid',
                                method='fft')

        envelope = np.abs(hilbert(correlation))
        max_correlation = np.max(correlation)
        peaks, _ = find_peaks(envelope)
        filtered_peaks = peaks[(peaks < len(read_signal)).nonzero()]
        peaks = peaks[(envelope[filtered_peaks] > .85 * max_correlation).nonzero()]
        ratio = (np.max(signal) / np.max(correlation))
        correlation *= ratio
        envelope *= ratio


        print('peaks',peaks[0])
        index_middle = temp + peaks[0]
        result.append(index_middle)
        print(index_middle)
        index_middle += len(data)
        for i in range(9):

            # find onset, this differs from the description in the paper which uses a sharpness and peak finding algorithm
            correlation = correlate(read_signal[i* 22050 + index_middle:(i + 1) * 22050+index_middle], signal, mode='valid',
                                    method='fft')
            print('from:',i*length_file+index_middle,'to:',(i+1)*length_file+index_middle)

            envelope = np.abs(hilbert(correlation))
            max_correlation = np.max(correlation)
            peaks, _ = find_peaks(envelope)
            filtered_peaks = peaks[(peaks < len(read_signal)).nonzero()]
            peaks = peaks[(envelope[filtered_peaks] > .85 * max_correlation).nonzero()]
            ratio = (np.max(signal) / np.max(correlation))
            correlation *= ratio
            envelope *= ratio

            if len(peaks) == 0:
                print('done!')

            else:
                print(peaks[0])
                result.append(peaks[0] + i * 22050+index_middle)


        total = []
        time = []
        distance = []
        print(result)
        for i in range(10):
            total.append(result[i] - result[i+10])
            time.append((result[i] - result[i+10])/44100)
            distance.append((result[i] - result[i+10])/44100*170)

        print(total)
        print(time)
        print(distance)




if __name__ == "__main__":
    app = QApplication(sys.argv)

    myWindow = WindowClass()

    myWindow.show()

    app.exec_()
