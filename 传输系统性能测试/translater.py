import sys
import wave
from threading import Thread

import np as np
import numpy as np
import scipy.io as sio
from PyQt5.QtCore import QThread
from PyQt5.QtWidgets import *
from PyQt5 import uic
import pyaudio
from scipy.io.wavfile import write
import sounddevice as sd
import time
import scipy.io.wavfile as wav
import soundfile as sf
from Messages import msgButtonClick, showDialog
from scipy import signal
import operator
import csv
import math
form_class = uic.loadUiType("gui_for_translate.ui")[0]

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
        self.encode_button.clicked.connect(self.encode)
        #self.decode_button.clicked.connect(self.decode)
        self.start_record.clicked.connect(self.recording)
        self.stop_record.clicked.connect(self.stop_recording)
        self.threadclass = ThreadClass()





    def decode(self):
        display_result = ''
        filtedData = []
        current_length = 0
        current_data = []
        data = []
        flag = 0
        impulse_fft = []
        impulse_fft_tmp = []
        bin = []
        real = []
        self.decode_text.setPlainText(''.join(real))


        bandpass1 = 3000
        bandpass2 = 7000
        read_signal, fs = sf.read(FILENAME)

        wn1 = 2.0 * bandpass1 / sampling_rate
        wn2 = 2.0 * bandpass2 / sampling_rate
        b, a = signal.butter(8, [wn1, wn2], 'bandpass')  # 범위를 조금 더 넓게 해야할
        filtedData = signal.filtfilt(b, a, read_signal)  # data为要过滤的信号

        current_length = len(filtedData)
        current_data = filtedData
        while 1:
            once_check = 0
            corr = []
            corr_index = dict()


            print('finding preamble')
            print('current_data length',len(current_data))
            for i in range(current_length - len(preamble_y)):
                corr.append(np.corrcoef(current_data[i:i + len(preamble_y)], preamble_y)[0, 1])
                if once_check + 24000 == i and once_check != 0:
                    print('corr 찾는거 ')
                    break

                if corr[i] > 0.5:
                    if once_check == 0:
                        once_check = i
                        print('once_check',once_check)

                    corr_index[i] = corr[i]



            try:
                flag = max(corr_index.items(), key=operator.itemgetter(1))[0]
            except:
                print('decode 结束')
                break


            print(flag)
            data = current_data[flag + len(preamble_y):flag + len(preamble_y)+60000]

            target_fre = 6000
            n = len(data)
            window = 600
            impulse_fft = np.zeros(n)
            for i in range(int(n - window)):
                y = np.fft.fft(data[i:i + int(window) - 1])
                y = np.abs(y)
                index_impulse = round(target_fre / sampling_rate * window)
                impulse_fft[i] = max(y[index_impulse - 2:index_impulse + 2])

            sliding_window = 5
            impulse_fft_tmp = impulse_fft
            for i in range(1 + sliding_window, n - sliding_window):
                impulse_fft_tmp[i] = np.mean(impulse_fft[i - sliding_window:i + sliding_window])
            impulse_fft = impulse_fft_tmp


            #
            #
            # position_impulse = [];
            # half_window = 800;
            #
            #
            #
            # for i in range(n-half_window*2):
            #     if impulse_fft[i+half_window] > 90 and impulse_fft[i+half_window] == max(impulse_fft[i - half_window: i + half_window]):
            #         position_impulse.append(i)
            # message_bin = np.zeros(230400)
            # for i in range(len(position_impulse)):
            #     message_bin[math.ceil(position_impulse / 4800)] = 1
            # real_message_start = 1
            # last_one_index = 1
            # for i in range(3):
            #     if message_bin[i] == 1:
            #         last_one_index = i
            #
            # real_message_start = last_one_index + 1
            #
            # real_message_bin = message_bin[real_message_start:230400]
            #
            # curr_package_index = 0
            # curr_bin_index = 1
            # real_message_bin = np.matrix.H(real_message_bin)


            plus = 0
            adjust = 0
            count =0
            while 1:
                decode_length = ''
                if adjust == 1:
                    plus += 0.1
                    print(plus)
                for i in range(8):

                    bin = np.mean(impulse_fft[i * 1200:(i + 1) * 1200])
                    bin += plus
                    print(bin)
                    if bin < 5:
                        decode_length = decode_length + '0'
                    else:
                        decode_length = decode_length + '1'

                print(decode_length)
                decode_payload_length = int(decode_length, 2)
                count += 1
                if count == 30:
                    break
                if decode_payload_length != 35:
                    adjust = 1
                else:
                    break

            if count == 30:
                decode_length = ''

                for i in range(8):
                    bin = np.mean(impulse_fft[i * 1200:(i + 1) * 1200])
                    print(bin)
                    if bin < 3:
                        decode_length = decode_length + '0'
                    else:
                        decode_length = decode_length + '1'

                print(decode_length)
                decode_payload_length = int(decode_length, 2)
                adjust = 0

                decode_payload = ''
                for i in range(decode_payload_length):
                    bin = np.mean(impulse_fft[(i + 8) * 1200:(i + 1 + 8) * 1200])

                    if bin < 3:
                        decode_payload = decode_payload + '0'
                    else:
                        decode_payload = decode_payload + '1'
                    print(bin)
            else:
                decode_payload = ''
                for i in range(decode_payload_length):
                    bin = np.mean(impulse_fft[(i + 8) * 1200:(i + 1 + 8) * 1200])

                    if adjust == 1:
                        bin += plus
                    if bin < 5:
                        decode_payload = decode_payload + '0'
                    else:
                        decode_payload = decode_payload + '1'
                    print(bin)

            print(decode_payload)
            while 1:
                if len(decode_payload) % 7 != 0:
                    decode_payload = decode_payload + '0'
                else:
                    break

            print(1200*(int(decode_length,2)+8))
            current_data = current_data[1200*(int(decode_length,2)+8+20)+flag:len(current_data)]
            current_length = len(current_data)
            display_result = display_result + decode_payload


        real = []
        for i in range(int(len(display_result) / 7)):
            real.append(decode_c(display_result[i * 7:(i + 1) * 7]))
        print(real)
        self.decode_text.setPlainText(''.join(real))

        print('result:',''.join(real))
        global start
        cost_time = "time:" + str(time.time() - start) + '\n'
        decode_payload = decode_payload + '\n'

        file = open("result_translate.txt", 'w')
        file.write(cost_time)
        file.write(decode_payload)
        file.write(''.join(real))
        file.close()

    def recording(self):
        global start
        start = time.time()

        print('recording....')
        self.threadclass.start()

    def stop_recording(self):
        global flag
        flag = 1
        print('stop!!!')
        time.sleep(1)
        self.decode()
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
        length = 35
        temp = [input_text[i:i + length] for i in range(0, len(input_text), length)]
        for i in range(len(temp)):
            length_payload = str(format(len(temp[i]), 'b'))  # 7자리
            if len(length_payload) < 8:
                length_payload = length_payload.zfill(8)
            print(temp[i])
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


