import sys
import wave
import numpy as np
import scipy.io as sio
from PyQt5.QtCore import QThread
from PyQt5.QtWidgets import *
from PyQt5 import uic
import pyaudio
from scipy.io import wavfile
from scipy.io.wavfile import write
import sounddevice as sd
import time
import soundfile as sf
from Messages import msgButtonClick, showDialog
from scipy import signal
import operator
import csv
import noisereduce as nr
# load data
import matplotlib.pyplot as plt
import pandas as pd


form_class = uic.loadUiType("gui_all.ui")[0]

sampling_rate = 48000
symbol_duration = 0.025
f = 4000
N = sampling_rate * symbol_duration
Ts = 1 / sampling_rate
FILENAME = "tmp.wav"
preamble = '01010101010101010101'

preamble_signal = ','.join(list(preamble))
preamble_time_signal = N * len(preamble) / sampling_rate
preamble_t = np.arange(0, preamble_time_signal, Ts)
preamble_np_signal = np.fromstring(preamble_signal, dtype='int', sep=',')
preamble_sample = np.repeat(preamble_np_signal, N)
preamble_y = np.sin(2 * np.pi * (f + preamble_sample * 2000) * preamble_t)


def encode_c(s):
    return ''.join([bin(ord(c)).replace('0b', '') for c in s])


def decode_c(s):
    return ''.join([chr(i) for i in [int(b, 2) for b in s.split(' ')]])


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
        self.decode_button.clicked.connect(self.decode)

        self.threadclass = ThreadClass()


    def decode(self):

        start = time.time()

        onset_data = []
        length_data = []
        coma = ''
        for i in range(41):
            coma = coma + ','
        result_f= open('result.csv', 'w')
        csv_content = []
        ff = open('content.csv', 'r', encoding='utf-8')
        rdr = csv.reader(ff)
        i = 0
        for line in rdr:
            temp = ''
            once = 0
            i = i + 1
            if i < 6:
                pass
            else:
                onset_data.append(line[3])
                length_data.append(line[2])
                for i in range(int(line[2])):


                    temp = temp + line[4 + i]

                result_f.write(line[2]+','+','.join(temp)+'\n')
                csv_content.append(line[2]+temp)
        print(csv_content)
        result_f.write(coma+','+'ross data'+','+'ross data rate'+'\n')
        ff.close()
        print(length_data)
        print(onset_data)
        bandpass1 = 3000
        bandpass2 = 7000
        fig = plt.figure(figsize=(10, 6))

        read_signal, fs = sf.read('res.wav')


        wn1 = 2.0 * bandpass1 / sampling_rate
        wn2 = 2.0 * bandpass2 / sampling_rate
        b, a = signal.butter(8, [wn1, wn2], 'bandpass')  # 범위를 조금 더 넓게 해야할

        filtedData = signal.filtfilt(b, a, read_signal)  # data为要过滤的信号




        current_length = len(filtedData)
        current_data = filtedData
        preamble_index = 20*1200


        for index in range(len(onset_data)):


            data = current_data[int(onset_data[index]) +preamble_index:int(onset_data[index]) +preamble_index + (int(length_data[index])+8)*1200]

            target_fre = 6000
            n = len(data)
            window = 600
            impulse_fft = np.zeros(n)
            for i in range(int(n - window)):
                y = np.fft.fft(data[i:i + int(window) - 1])
                y = np.abs(y)
                index_impulse = round(target_fre / sampling_rate * window)
                impulse_fft[i] = max(y[index_impulse - 2:index_impulse + 2])

            sliding_window = 10
            impulse_fft_tmp = impulse_fft
            for i in range(1 + sliding_window, n - sliding_window):
                impulse_fft_tmp[i] = np.mean(impulse_fft[i - sliding_window:i + sliding_window])
            impulse_fft = impulse_fft_tmp

            temporary = []
            adjust = 0
            while 1:
                decode_length = ''

                for i in range(8):
                    bin = np.mean(impulse_fft[i * 1200:(i + 1) * 1200])
                    temporary.append(bin)
                    if adjust == 1:
                        bin +=20
                        temporary.append(bin)

                    if adjust == 2:
                        bin += 30
                        temporary.append(bin)

                    if bin < 70:
                        decode_length = decode_length + '0'
                    else:
                        decode_length = decode_length + '1'

                print(decode_length)
                decode_payload_length = int(decode_length, 2)
                print(decode_payload_length)
                if decode_payload_length < 30 or max(temporary) < 80 and adjust == 0:
                    adjust = 1
                    continue
                if max(temporary)<80 and adjust == 1:
                    adjust = 2

                else:
                    break




            decode_payload = ''
            for i in range(decode_payload_length):
                bin = np.mean(impulse_fft[(i + 8) * 1200:(i + 1 + 8) * 1200])
                if adjust == 1:
                    bin += 20
                if adjust == 2:
                    bin +=30
                if bin < 70:
                    decode_payload = decode_payload + '0'
                else:
                    decode_payload = decode_payload + '1'
                if index == 16 or index ==17 or index ==18 or index ==22 or index ==26 or index ==27 or index ==28 or index ==30 or index == 32 or index ==33 or index ==34:
                    print(bin)


            print('aaaaa',decode_payload)
            error_count = 0
            Target_data = csv_content[index][2:]
            print(Target_data)
            for i in range(len(Target_data)):
                if Target_data[i]!=decode_payload[i]:
                    error_count += 1
            coma_2 = ''
            for i in range(41-len(Target_data)):
                coma_2 = coma_2 + ','

            result_f.write(str(decode_payload_length) + ',' + ','.join(decode_payload)+coma_2+','+str(error_count)+','+str(error_count/len(Target_data)*100)+'%'+'\n')


        end = time.time()
        print(end-start)



    def recording(self):
        print('recording....')
        self.threadclass.start()

    def stop_recording(self):
        global flag
        flag = 1
        print('stop!!!')

    def encode(self):
        input_text = encode_c(self.encode_text.toPlainText())
        if input_text == '':
            msgBox = QMessageBox()
            msgBox.setIcon(QMessageBox.Information)
            msgBox.setText("不能是空的")
            msgBox.setWindowTitle("系统警告")
            msgBox.setStandardButtons(QMessageBox.Ok | QMessageBox.Cancel)
            msgBox.buttonClicked.connect(msgButtonClick)
            returnValue = msgBox.exec()
            if returnValue == QMessageBox.Ok:
                return
        length = 40
        temp = [input_text[i:i + length] for i in range(0, len(input_text), length)]
        for i in range(len(temp)):
            length_payload = str(format(len(temp[i]), 'b'))  # 7자리
            if len(length_payload) < 8:
                length_payload = length_payload.zfill(8)
            total = preamble + length_payload + temp[i]
            signal = ','.join(list(total))
            time_signal = N * len(total) / sampling_rate
            t = np.arange(0, time_signal, Ts)
            np_signal = np.fromstring(signal, dtype='int', sep=',')
            sample = np.repeat(np_signal, N)
            if (len(t) % 10) != 0:
                t = t[:len(t) - 1]
            y = np.sin(2 * np.pi * (f + sample * 2000) * t)
            write('first.wav', sampling_rate, y)
            samplerate, data = sio.wavfile.read('first.wav')
            sd.play(data, samplerate)
            time.sleep(time_signal + 1)

    def window(self):
        app = QApplication(sys.argv)
        win = QWidget()
        button1 = QPushButton(win)
        button1.setText("Show dialog!")
        button1.move(50, 50)
        button1.clicked.connect(showDialog)
        win.setWindowTitle("Click button")
        win.show()
        sys.exit(app.exec_())

    def msgButtonClick(self, i):
        print("Button clicked is:", i.text())


if __name__ == "__main__":
    app = QApplication(sys.argv)

    myWindow = WindowClass()

    myWindow.show()

    app.exec_()
