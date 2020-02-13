#!usr/bin/python3
import sys
import os
import time
import array
import configparser
import subprocess
from datetime import datetime
from datetime import timedelta
import socket

import numpy as np
# import scipy as sci

import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import style

from pydub import AudioSegment
from pydub import generators
from pydub.playback import play

import sounddevice as sd
import soundfile as sf

from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QDesktopWidget, QComboBox, QMainWindow,\
		QLineEdit, QGridLayout, QLabel, QCheckBox, QStackedLayout, QStackedWidget, QProgressBar, QTextEdit, QListWidget, QListWidgetItem, QRadioButton, QHBoxLayout, QVBoxLayout
from PyQt5.QtCore import QCoreApplication, QThread, pyqtSignal, QObject, pyqtSlot, QRegExp
from PyQt5.QtGui import QColor, QRegExpValidator


def get_current_time(mode=1):
	if mode == 1:
		return datetime.now().strftime("%d-%m-%Y %H_%M_%S")
	else:
		return datetime.now().strftime("%d-%m-%Y %H:%M:%S")

def normalized(raw_bytes, frame, norm=0):
	data = []
	bit_depth = 8*frame
	for i in raw_bytes:
		if i>=int((2**bit_depth-1)/2):
			data.append(i-(2**bit_depth-1)-1)
		else:
			data.append(i)
	return [i/(2**(frame*8)) for i in data] if norm else data

def get_raw_bytes(segment):
	data = list(segment.raw_data)
	if segment.channels == 1:
		#Returning only 1 channel
		if segment.frame_width==2:
			k, v = data[::2], data[1::2]
			return [(v[i]<<8)|k[i] for i in range(len(k))]
		else:
			return data
	else:
		#Returning both channels: Left and Right
		if segment.frame_width == 2:
			return data[::2],data[1::2]
		else:
			data = np.split(np.array(data), len(data)/2)
			L, R = data[::2], data[1::2]
			return [(i[1]<<8)|i[0] for i in L], [(i[1]<<8)|i[0] for i in R]

def get_freq_band(fft, sensor_freq, sample_rate):
	ret = []
	lower, upper = (sensor_freq-30)*len(fft)/sample_rate, (sensor_freq+30)*len(fft)/sample_rate
	low_i, upper_i = int(lower), int(upper)
	for i in range(int(len(fft)/2)):
		if low_i <= i <= upper_i:
			ret.append(fft[i])
		else:
			ret.append(0)
	return ret

class ChannelSensor(object):
	def __init__(self, input_lines, freqs, name):
		self.coeffs = [[float(s.text()) for s in i] for i in input_lines]
		self.freqs = freqs
		self.name = name
	
	def calc_polynoms(self, args):
		ret = [0,0]
		for q,s in enumerate(self.coeffs):
			for v,k in enumerate(s):
				ret[q] += k*args[q]**(5-v)
		return ret

	def process_channel_data(self, data, sample_rate, ret_type):
		FFT = np.fft.fft(data)

		# Get frequency bands from sensors
		# and iFFT from them
		bands = [np.fft.ifft(get_freq_band(FFT, freq, sample_rate)) for freq in self.freqs]

		# Get mean values from bands
		means = [np.mean(np.absolute(i)) for i in bands]

		# Convert discrete values
		values = self.calc_polynoms(means)
		if ret_type:
			return self.name+' CHANNEL REACTION:\n'+'\n'.join([' Hz - '.join(['\t'+str(f),str(v)]) for f, v in zip(self.freqs, values)])
		else:
			return values

class MeasureThread(QObject):

	log_signal = pyqtSignal(str, int)
	finished_signal = pyqtSignal()
	progress_bar_signal = pyqtSignal(int, int)
	back_signal = pyqtSignal()
	
	def __init__(self, args):
		super(MeasureThread, self).__init__()
		self.args = args
		self.processed_data = []

	def create_output(self, sens_freq):

		left_ch = generators.Sine(sens_freq[0], bit_depth = 16).to_audio_segment(duration = self.duration*1000)
		right_ch = generators.Sine(sens_freq[1], bit_depth = 16).to_audio_segment(duration = self.duration*1000)

		# print("max sin", max(left_ch.get_array_of_samples()))

		if self.channels == 1:
			# REWORK MONO MODE
			ovrl_ch = left_ch.overlay(right_ch)
			ad = AudioSegment.from_mono_audiosegments(ovrl_ch)
		else:
			#ovrl_ch = left_ch.overlay(right_ch)
			ad = AudioSegment.from_mono_audiosegments(left_ch, right_ch)

		out = np.array(ad.get_array_of_samples())

		return np.array(np.split(out, int(len(out)/2)), dtype = 'int16')

	def save_to_file(self, data, tm = get_current_time()):
		message = "Сохранение результатов в файл records/"+ tm +".txt"
		self.log_signal.emit(message, 0)

		if not os.path.exists("records"):
			os.system("mkdir records")

		file = open('records/'+tm+'.txt', 'w')
		for i in data:
			file.write(','.join([str(k) for k in i]) +'\n')
		file.close()

	@pyqtSlot()
	def single_mode(self):
		self.freq = int(self.args[0])
		self.channels = int(self.args[1])
		self.duration = int(self.args[2])
		self.is_txt = self.args[3]
		self.is_wav = self.args[4]
		self.sensors = self.args[5]

		sd.default.samplerate = self.freq
		sd.default.channels = self.channels
		sd.default.dtype = 'int16'

		message = "Проводятся измерения:\nЧастота дискретизации - %d Hz\nКол-во каналов - %d\nДлительность измерения - %d (секунды)" % (self.freq, self.channels, self.duration)
		self.log_signal.emit(message, 1)

		message = "Подготовка выходного сигнала, частота датчиков: %s Hz, %s Hz" % (tuple(self.sensors[0].freqs))
		self.log_signal.emit(message, 0)
		output_signal = self.create_output(self.sensors[0].freqs)
		
		self.progress_bar_signal.emit(10, 1)

		message = "Измерение..."
		self.log_signal.emit(message, 0)
		self.progress_bar_signal.emit(20, 1)

		input_signal = sd.playrec(output_signal)
		sd.wait()
		self.progress_bar_signal.emit(50, 1)
		
		message = "Вычисление показаний датчиков\n"
		self.log_signal.emit(message, 0)

		processed = []

		for d,s in zip(input_signal.T, self.sensors):
			processed.append(s.process_channel_data(d, self.freq, 1))

		for i in processed:
			self.log_signal.emit(i, 3)

		if self.is_wav:
			tm = get_current_time()
			message = "Сохранение записанного результата в файл "+tm+".wav"
			self.log_signal.emit(message, 0)

			sound = array.array('h', np.concatenate(input_signal))
			new_sound = AudioSegment(data = b'', sample_width = sd.default.sample_width, frame_rate = self.freq, channels = self.channels)._spawn(sound)
			new_sound.export('records/'+tm+'.wav', format = 'wav')

			self.progress_bar_signal.emit(60, 1)
			
			
		if self.is_txt:
			self.save_to_file(input_signal)
			self.progress_bar_signal.emit(70, 1)

		if not (self.is_txt  or self.is_wav):
			message = "Сохранение в файл не было выбрано, отображение полученных результатов на графике"
			self.log_signal.emit(message, 0)
			time.sleep(2)

			plt.figure()
			plt.plot([i[0] for i in input_signal])
			if self.channels==2:
				plt.plot([i[1] for i in input_signal])
			plt.show()

		# TERMINATE ?
		self.progress_bar_signal.emit(100, 1)

		self.back_signal.emit()

	@pyqtSlot()
	def cycled_mode(self):
		self.freq = int(self.args[0])
		self.channels = int(self.args[1])
		self.duration = int(self.args[2])
		self.period = int(self.args[3])
		self.live = self.args[4]
		self.quantity = int(self.args[5])
		# collecting starting time: hour, minute, second
		self.start_time = [int(i) for i in self.args[6]]
		self.sensors = self.args[7]

		sd.default.samplerate = self.freq
		sd.default.channels = self.channels
		sd.default.dtype = 'int16'

		self.progress_bar_signal.emit(self.quantity, 0)

		message = "Проводятся циклические измерения:\nЧастота дискретизации - %d Hz\nКол-во каналов - %d\nДлительность измерения - %d (сек.)\nПериодичность измерений - %d (сек.)" % (self.freq, self.channels, self.duration, self.period)
		self.log_signal.emit(message, 1)

		message = "Подготовка выходного сигнала, частота датчиков: %s Hz, %s Hz" % (tuple(self.sensors[0].freqs))
		self.log_signal.emit(message, 2)
		output_signal = self.create_output(self.sensors[0].freqs)

		message = "Циклическое измерение..."
		self.log_signal.emit(message, 0)

		slp = self.period-self.duration
        
		self.log_signal.emit('Подключение отображения графиков', 1)
		if self.live:
			sock = socket.socket()
			
			if self.channels==2:
				temp_name = 'L,R'
			else:
				temp_name = 'M'

			print(["plotter_v2.exe", "15098", temp_name, ','.join([str(i)+'Hz' for i in self.sensors[0].freqs])])
			subprocess.Popen(["plotter_v2.exe", "15098", temp_name, ','.join([str(i)+'Hz' for i in self.sensors[0].freqs])])
			for i in range(15):
				try:
					sock.connect(('localhost', 15098))
					self.log_signal.emit("Подключение к графикам установлено", 1)
					break
				except Exception as e:
					self.log_signal.emit(str(e)+". Переподключение %d/15"%(i), -1)
					time.sleep(1)
					continue

		message = "Ожидание времени начала измерений. Осталось ждать: "
		tm = get_current_time()
		time_now = datetime.now()
		time_start = datetime(time_now.year, time_now.month, time_now.day, self.start_time[0], self.start_time[1], self.start_time[2])
		time_delta = time_start - time_now
		
		secs = time_delta.seconds
		message+=str(secs//3600)+" ч., "
		secs = secs % 3600
		message+=str(secs//60)+" мин.,"
		secs = secs % 60
		message+=str(secs)+" c."
		self.log_signal.emit(message, 1)

		while datetime.now() < time_start:
			time.sleep(0.5)
			self.log_signal.emit('До начала измерений %s сек.' % (str((time_start-datetime.now()).seconds)), -1)

		for count in range(1,self.quantity+1):

			message = "Измерение №"+str(count)
			self.log_signal.emit(message, 0)

			input_signal = sd.playrec(output_signal)
			sd.wait()

			t1 = time.perf_counter()

			reaction = []
			for s, data in zip(self.sensors, input_signal.T):
				reaction += s.process_channel_data(data, self.freq, 0)

			self.processed_data.append(reaction)

			if self.live:
				reaction = [str(i) for i in reaction]
				data_to_send = (','.join(reaction)+'|').encode()
				print('send', data_to_send)
				sock.send(data_to_send)

			self.log_signal.emit("Calculation time elapsed: "+str(time.perf_counter()-t1), -1)
			
			self.progress_bar_signal.emit(count, 1)

			if count != self.quantity:
				time.sleep(slp-t1+time.perf_counter())

		message = "Измерения завершены успешно"
		self.log_signal.emit(message, 3)
		self.save_to_file(self.processed_data, tm+" __ "+get_current_time())

		sock.close()
		self.finished_signal.emit()


class Window(QMainWindow):
	def __init__(self):
		super().__init__()

		self.setWindowTitle("Sound Measure")
		self.resize(200,300)
		self.center()

		self.stored_configs = self.get_all_configs()
		self.dynamic_labels = []
		self.single_style_layout()
		self.cycled_style_layout()
		self.measurements_layout()
		self.config_sens_layout()
		if self.stored_configs != []:
			self.load_config(self.stored_configs[0])

		self.stacked_layout = QStackedLayout()
		self.stacked_layout.addWidget(self.main_widget)
		self.stacked_layout.addWidget(self.cycled_widget)
		self.stacked_layout.addWidget(self.msr_widget)
		self.stacked_layout.addWidget(self.config_central_widget)

		self.central_widget = QWidget()
		self.central_widget.setLayout(self.stacked_layout)
		self.setCentralWidget(self.central_widget)

	def single_style_layout(self):
		switch_button = QPushButton("Перейти к циклическому режиму")

		freq_combobox = QComboBox()
		freq_combobox.addItems(['44100','48000'])

		self.single_chan_combobox = QLineEdit()
		self.single_chan_combobox.setReadOnly(True)
		# self.single_chan_combobox.addItems(['2','1'])

		duration_combobox = QComboBox()
		duration_combobox.addItems(['1','2', '5', '10', '30'])

		save_as_txt = QCheckBox("txt")
		save_as_wav = QCheckBox("wav")
		start_button = QPushButton("Старт")
		config_button = QPushButton("Настройка")

		self.single_cfg_chooser = QComboBox()
		self.single_cfg_chooser.addItems(self.stored_configs)

		grid = QGridLayout()

		grid.addWidget(QLabel("Одиночный тип измерений"), 1, 0)
		grid.addWidget(switch_button, 1, 1, 1, 2)

		grid.addWidget(QLabel("Частота дискретизации"), 2,0)
		grid.addWidget(freq_combobox, 2, 1, 1, 2)

		grid.addWidget(QLabel("Кол-во каналов"), 3, 0)
		grid.addWidget(self.single_chan_combobox, 3, 1, 1, 2)

		grid.addWidget(QLabel("Длительность (секунды)"), 4, 0)
		grid.addWidget(duration_combobox, 4, 1, 1, 2)

		grid.addWidget(QLabel("Сохранить в"), 5, 0)
		grid.addWidget(save_as_txt, 5, 1)
		grid.addWidget(save_as_wav, 5, 2)

		grid.addWidget(self.single_cfg_chooser, 6, 0)
		grid.addWidget(config_button, 6, 1)
		grid.addWidget(start_button, 6, 2)

		self.main_widget = QWidget()
		self.main_widget.setLayout(grid)

		self.single_params = [freq_combobox, self.single_chan_combobox, duration_combobox, save_as_txt, save_as_wav]

		switch_button.clicked.connect(self.act_cycled)
		start_button.clicked.connect(self.start_msr_single)
		config_button.clicked.connect(self.act_config)
		self.single_cfg_chooser.currentIndexChanged[str].connect(self.load_config)

	def cycled_style_layout(self):

		switch_button = QPushButton("Перейти к одиночному режиму")

		freq_combobox = QComboBox()
		freq_combobox.addItems(['44100','48000'])

		self.cycled_chan_combobox = QLineEdit()
		self.cycled_chan_combobox.setReadOnly(True)
		# self.cycled_chan_combobox.addItems(['2','1'])

		duration_combobox = QComboBox()
		duration_combobox.addItems(['1','2', '5', '10', '30'])

		period_combobox = QComboBox()
		period_combobox.addItems(['10', '30', '60', '600', '3600'])

		quantity_line = QLineEdit()
		quantity_regex = QRegExp("^[0-9]+")
		input_validator = QRegExpValidator(quantity_regex, quantity_line)
		quantity_line.setValidator(input_validator)
		quantity_line.setText("1")

		time_regex = QRegExp("^([0-9]|[1-5][0-9])$")
		
		hour_line = QLineEdit()
		hour_validator = QRegExpValidator(time_regex, hour_line)
		hour_line.setValidator(hour_validator)
		hour_line.setText("0")

		minute_line = QLineEdit()
		minute_validator = QRegExpValidator(time_regex, minute_line)
		minute_line.setValidator(minute_validator)
		minute_line.setText("0")

		second_line = QLineEdit()
		second_validator = QRegExpValidator(time_regex, second_line)
		second_line.setValidator(second_validator)
		second_line.setText("0")

		live = QCheckBox("Отображать полученные результаты на графиках")
		live.setChecked(True)
		start_button = QPushButton("Старт")

		self.cycled_cfg_chooser = QComboBox()
		self.cycled_cfg_chooser.addItems(self.stored_configs)

		grid = QGridLayout()

		grid.addWidget(QLabel("Циклический тип измерений"), 1, 0)
		grid.addWidget(switch_button, 1, 1, 1, 3)

		grid.addWidget(QLabel("Частота дискретизации"), 2,0)
		grid.addWidget(freq_combobox, 2, 1)

		grid.addWidget(QLabel("Кол-во каналов"), 3, 0)
		grid.addWidget(self.cycled_chan_combobox, 3, 1)

		grid.addWidget(QLabel("Длительность (секунды)"), 4, 0)
		grid.addWidget(duration_combobox, 4, 1)

		grid.addWidget(QLabel("Периодичность (секунды)"), 5, 0)
		grid.addWidget(period_combobox, 5, 1)

		grid.addWidget(QLabel("Кол-во измерений"), 6, 0)
		grid.addWidget(quantity_line, 6, 1)

		grid.addWidget(QLabel("Начать измерения в (час., мин., сек.)"), 7,0)
		grid.addWidget(hour_line, 7,1)
		grid.addWidget(minute_line, 7,2)
		grid.addWidget(second_line, 7,3)

		grid.addWidget(live, 9, 0, 1, 2)

		grid.addWidget(self.cycled_cfg_chooser, 9, 2)
		grid.addWidget(start_button, 9, 3)

		self.cycled_widget = QWidget()
		self.cycled_widget.setLayout(grid)

		self.cycled_params = [freq_combobox, self.cycled_chan_combobox, duration_combobox, period_combobox, live, quantity_line, [hour_line, minute_line, second_line]]

		switch_button.clicked.connect(self.act_single)
		start_button.clicked.connect(self.start_msr_cycled)
		self.cycled_cfg_chooser.currentIndexChanged[str].connect(self.load_config)

	def measurements_layout(self):

		self.log_console = QListWidget()

		self.progressBar = QProgressBar()
		self.progressBar.setRange(0, 100)
		self.progressBar.setValue(0)

		self.stop_button = QPushButton("Выход")
		self.stop_button.setEnabled(True)
		self.stop_button.clicked.connect(self.act_exit)

		self.back_buttn = QPushButton("Назад")
		self.back_buttn.setEnabled(False)
		self.back_buttn.clicked.connect(self.act_single)
		
		grid = QGridLayout()

		grid.addWidget(QLabel("Ход измерений: "))
		grid.addWidget(self.log_console, 1, 0, 1, 4)
		grid.addWidget(self.progressBar, 2, 0, 1, 4)
		grid.addWidget(self.stop_button, 3, 3)
		grid.addWidget(self.back_buttn, 3, 2)

		self.msr_widget = QWidget()
		self.msr_widget.setLayout(grid)
		
	def config_sens_layout(self): 
		self.config_name = QLineEdit()

		second_sens_radio_buttn = QRadioButton("2")
		quarter_sens_radio_buttn = QRadioButton("4")
		self.radio_bttns = [second_sens_radio_buttn, quarter_sens_radio_buttn]

		self.sens_freq_first = QLineEdit()
		self.sens_freq_second = QLineEdit()

		reg_ex = QRegExp("^[0-9]+")
		input_validator = QRegExpValidator(reg_ex, self.sens_freq_first)
		self.sens_freq_first.setValidator(input_validator)
		self.sens_freq_second.setValidator(input_validator)

		self.RIGHT_CHANNEL_SENS = [[QLineEdit() for i in range(6)] for i in range(2)]
		
		self.LEFT_CHANNEL_SENS = [[QLineEdit() for i in range(6)] for i in range(2)]

		self.MONO_CHANNEL_SENS = [[QLineEdit() for i in range(6)] for i in range(2)]

		for a in [self.RIGHT_CHANNEL_SENS, self.LEFT_CHANNEL_SENS, self.MONO_CHANNEL_SENS]:
			for line in a:
				for obj in line:
					obj.setText('0')

		back_buttn = QPushButton("Назад")
		save_buttn = QPushButton("Сохранить")

		MAIN_BOX = QVBoxLayout()

		CONFIG_INFO_BOX = QHBoxLayout()
		info_layout = QVBoxLayout()
		info_layout.addWidget(QLabel("Название конфигурации"))
		info_layout.addWidget(QLabel("Кол-во параметров"))
		info_layout.addWidget(QLabel("Частоты датчиков (Hz)"))
		CONFIG_INFO_BOX.addLayout(info_layout)

		data_layout = QVBoxLayout()
		data_layout.addWidget(self.config_name)

		line_layout = QHBoxLayout()
		line_layout.addWidget(second_sens_radio_buttn)
		line_layout.addWidget(quarter_sens_radio_buttn)
		data_layout.addLayout(line_layout)

		line_layout = QHBoxLayout()
		line_layout.addWidget(self.sens_freq_first)
		line_layout.addWidget(self.sens_freq_second)
		data_layout.addLayout(line_layout)
		CONFIG_INFO_BOX.addLayout(data_layout)
		MAIN_BOX.addLayout(CONFIG_INFO_BOX)

		# STACKED LAYOUT #
		sens2_layout = QVBoxLayout()
		mono_widget = self.create_channel_widget("Mono ch.", ["Q1", "Q2"], self.MONO_CHANNEL_SENS)
		sens2_layout.addWidget(mono_widget)

		sens4_layout = QVBoxLayout()
		left_ch_widget = self.create_channel_widget("Left ch.", ["Y1", "Y2"], self.LEFT_CHANNEL_SENS)
		right_ch_widget = self.create_channel_widget("Right ch.", ["Z1", "Z2"], self.RIGHT_CHANNEL_SENS)
		sens4_layout.addWidget(left_ch_widget)
		sens4_layout.addWidget(right_ch_widget)

		sens2 = QWidget()
		sens2.setLayout(sens2_layout)

		sens4 = QWidget()
		sens4.setLayout(sens4_layout)

		self.config_stacked_layout = QStackedWidget()
		self.config_stacked_layout.addWidget(sens2)
		self.config_stacked_layout.addWidget(sens4)
		
		# STACKED LAYOUT COMPLETED #

		# ADDING STACKED WIDGET TO MAIN BOX
		MAIN_BOX.addWidget(self.config_stacked_layout)

		# ADDING BUTTONS TO THE BOTTOM
		line_layout = QHBoxLayout()
		line_layout.addWidget(back_buttn)
		line_layout.addWidget(save_buttn)
		MAIN_BOX.addLayout(line_layout)

		self.config_central_widget = QWidget()
		self.config_central_widget.setLayout(MAIN_BOX)

		back_buttn.clicked.connect(self.act_single)
		save_buttn.clicked.connect(self.create_config)

		second_sens_radio_buttn.toggled.connect(lambda:self.config_stacked_layout.setCurrentIndex(0))
		quarter_sens_radio_buttn.toggled.connect(lambda:self.config_stacked_layout.setCurrentIndex(1))
		
		self.sens_freq_first.textChanged[str].connect(self.set_info_1freqs)
		self.sens_freq_second.textChanged[str].connect(self.set_info_2freqs)

	def create_config(self):

		CFG = configparser.ConfigParser()

		CFG.add_section("MAIN_SETTINGS")
		CFG.add_section("MONO_SETTINGS")
		CFG.add_section("STEREO_SETTINGS")

		CFG.set("MAIN_SETTINGS", "sens_q", list(filter(lambda x: x.isChecked(), self.radio_bttns))[0].text())
		CFG.set("MAIN_SETTINGS", "sens_freq_first", self.sens_freq_first.text())
		CFG.set("MAIN_SETTINGS", "sens_freq_second", self.sens_freq_second.text())

		for _,line in enumerate(self.MONO_CHANNEL_SENS):
			for ind, k in enumerate(line):
				CFG.set("MONO_SETTINGS", "mono_coeff"+str(_)+"_pow"+str(5-ind), k.text())

		for _,line in enumerate(self.LEFT_CHANNEL_SENS):
			for ind, k in enumerate(line):
				CFG.set("STEREO_SETTINGS", "left_coeff"+str(_)+"_pow"+str(5-ind), k.text())

		for _,line in enumerate(self.RIGHT_CHANNEL_SENS):
			for ind, k in enumerate(line):
				CFG.set("STEREO_SETTINGS", "right_coeff"+str(_)+"_pow"+str(5-ind), k.text())

		if not os.path.isdir('configs'):
			os.system('mkdir configs')

		with open('configs/'+self.config_name.text()+'.ini', 'w') as cfg_file:
			CFG.write(cfg_file)
		

		cfg_name = self.config_name.text()+'.ini'

		if cfg_name not in self.stored_configs:
			self.stored_configs.append(cfg_name)
			self.single_cfg_chooser.addItem(cfg_name)
			self.cycled_cfg_chooser.addItem(cfg_name)

		self.statusBar().showMessage('Успешное сохранение файла '+cfg_name)

	def load_config(self, name):

		CFG = configparser.ConfigParser()
		CFG.read('configs/'+name)

		self.config_name.setText(name.split('.')[0])
		q = int(int(CFG.get("MAIN_SETTINGS", "sens_q"))/2)
		self.radio_bttns[q-1].setChecked(True)
		self.single_chan_combobox.setText(str(q))
		self.cycled_chan_combobox.setText(str(q))

		self.sens_freq_first.setText(CFG.get("MAIN_SETTINGS", "sens_freq_first"))
		self.sens_freq_second.setText(CFG.get("MAIN_SETTINGS", "sens_freq_second"))

		for _,line in enumerate(self.MONO_CHANNEL_SENS):
			for ind, k in enumerate(line):
				k.setText(CFG.get("MONO_SETTINGS", "mono_coeff"+str(_)+"_pow"+str(5-ind)))

		for _,line in enumerate(self.LEFT_CHANNEL_SENS):
			for ind, k in enumerate(line):
				k.setText(CFG.get("STEREO_SETTINGS", "left_coeff"+str(_)+"_pow"+str(5-ind)))

		for _,line in enumerate(self.RIGHT_CHANNEL_SENS):
			for ind, k in enumerate(line):
				k.setText(CFG.get("STEREO_SETTINGS", "right_coeff"+str(_)+"_pow"+str(5-ind)))

		self.statusBar().showMessage('Успешная загрузка файла '+name)

	def get_all_configs(self):
		try:
			return [i for i in os.listdir('configs') if '.ini' in i]
		except Exception as e:
			self.statusBar().showMessage(str(e))	
			return []

	def set_info_1freqs(self, t):

		# USE SELF.SENDER() TO GET INFO FROM SENDER

		for label in self.dynamic_labels:
			text = label[0].text().split(',')
			text[1] = t+" Hz: "
			new_text = ', '.join(text)
			label[0].setText(new_text)

	def set_info_2freqs(self, t):
		
		# USE SELF.SENDER() TO GET INFO FROM SENDER

		for label in self.dynamic_labels:
			text = label[1].text().split(',')
			text[1] = t+" Hz: "
			new_text = ', '.join(text)
			label[1].setText(new_text)

	def center(self):
		qr = self.frameGeometry()
		cp = QDesktopWidget().availableGeometry().center()
		qr.moveCenter(cp)
		self.move(qr.topLeft())

	def log(self, message, mode):
		new_log = QListWidgetItem('\n'+get_current_time(mode=0) + ' ' +message)
		#mode = 0 #TEMPORARY 
		if mode == 1:
			new_log.setBackground(QColor(50, 170, 200, alpha = 50))
		elif mode == 2:
			new_log.setBackground(QColor(255, 70, 50, alpha = 50))
		elif mode == 3:
			new_log.setBackground(QColor(30, 250, 30, alpha = 50))
			
		if mode == -1:
			self.statusBar().showMessage(message)
		else:
			self.log_console.addItem(new_log)

	def set_prog_bar(self, value, mode):
		if mode:
			self.progressBar.setValue(value)
		else:
			self.progressBar.setRange(0,value)
			self.progressBar.setValue(0)

	def start_msr_single(self):
		self.stacked_layout.setCurrentIndex(2)
		self.resize(500, 400)
		self.center()

		params = self.single_params

		freq = params[0].currentText()
		channels = params[1].text()
		duration = params[2].currentText()
		is_txt = params[3].isChecked()
		is_wav = params[4].isChecked()
		
		params_to_send = [freq, channels, duration, is_txt, is_wav]
		sensor_freqs = [int(self.sens_freq_first.text()), int(self.sens_freq_second.text())]

		if channels == '1':
			mono_sensor = ChannelSensor(self.MONO_CHANNEL_SENS, sensor_freqs, 'MONO')
			params_to_send.append([mono_sensor])
		else:
			stereo_sensors = [ChannelSensor(self.LEFT_CHANNEL_SENS, sensor_freqs, "LEFT"), ChannelSensor(self.RIGHT_CHANNEL_SENS, sensor_freqs, "RIGHT")]
			params_to_send.append(stereo_sensors)

		self.single_msr = MeasureThread(params_to_send)
		self.thread = QThread()

		self.single_msr.log_signal.connect(self.log)
		self.single_msr.progress_bar_signal.connect(self.set_prog_bar)
		self.single_msr.back_signal.connect(lambda: self.back_buttn.setEnabled(True))

		self.single_msr.moveToThread(self.thread)

		self.single_msr.finished_signal.connect(self.thread.quit)

		self.thread.started.connect(self.single_msr.single_mode)

		self.thread.start()

	def start_msr_cycled(self):
		self.stacked_layout.setCurrentIndex(2)
		self.resize(500, 400)
		self.center()

		params = self.cycled_params

		freq = params[0].currentText()
		channels = params[1].text()
		duration = params[2].currentText()
		period = params[3].currentText()
		live = params[4].isChecked()
		quantity = params[5].text()
		st_hour, st_min, st_sec = params[6][0].text(), params[6][1].text(), params[6][2].text()

		params_to_send = [freq, channels, duration, period, live, quantity, [st_hour, st_min, st_sec]]
		sensor_freqs = [int(self.sens_freq_first.text()), int(self.sens_freq_second.text())]

		if channels == '1':
			mono_sensor = ChannelSensor(self.MONO_CHANNEL_SENS, sensor_freqs, 'MONO')
			params_to_send.append([mono_sensor])
		else:
			stereo_sensors = [ChannelSensor(self.LEFT_CHANNEL_SENS, sensor_freqs, "LEFT"), ChannelSensor(self.RIGHT_CHANNEL_SENS, sensor_freqs, "RIGHT")]
			params_to_send.append(stereo_sensors)

		self.cycled_msr = MeasureThread(params_to_send)
		self.thread = QThread()

		self.cycled_msr.log_signal.connect(self.log)
		self.cycled_msr.progress_bar_signal.connect(self.set_prog_bar)

		self.cycled_msr.moveToThread(self.thread)

		self.cycled_msr.finished_signal.connect(self.thread.quit)

		self.thread.started.connect(self.cycled_msr.cycled_mode)
		
		self.thread.start()

	def create_channel_widget(self, channel_name, sensors_names, lines_input):
		WIDGET_BOX = QWidget()

		BOX = QGridLayout()
		info_layout = QVBoxLayout()
		first_label = QLabel("Полином датчика %s, - Hz: " % (channel_name))
		second_label = QLabel("Полином датчика %s, - Hz: " % (channel_name))
		self.dynamic_labels.append([first_label, second_label])

		info_layout.addWidget(first_label)
		info_layout.addWidget(second_label)
		BOX.addLayout(info_layout, 0, 0)

		splitters = ["x^5 +","x^4 +","x^3 +","x^2 +","x +",False]
		data_layout = QVBoxLayout()

		for sens, n in zip(lines_input, sensors_names):
			line_layout = QHBoxLayout()
			line_layout.addWidget(QLabel(n+"(x) = "))
			for sens_input, splitter in zip(sens, splitters):
				line_layout.addWidget(sens_input)
				if splitter:
					line_layout.addWidget(QLabel(splitter))
			data_layout.addLayout(line_layout)

		BOX.addLayout(data_layout, 0, 1)
		WIDGET_BOX.setLayout(BOX)

		return WIDGET_BOX

	def act_config(self):
		# USE SELF.SENDER() TO GET INFO FROM SENDER 
		self.stacked_layout.setCurrentIndex(3)

	def act_cycled(self):
		# USE SELF.SENDER() TO GET INFO FROM SENDER
		self.stacked_layout.setCurrentIndex(1)

	def act_single(self):
		# USE SELF.SENDER() TO GET INFO FROM SENDER
		self.stacked_layout.setCurrentIndex(0)
		self.back_buttn.setEnabled(False)
		try:
			self.thread.quit()
		except Exception as e:
			print(e)

	def act_exit(self):
		try:
			self.thread.quit()
		except Exception as e:
			print(e)
		app.exit()


### REDIRECT OUTPUT TO LOG FILE ###
# doesnt work
if not os.path.exists("logs"):
	os.system("mkdir logs")
sys.stdout = open("logs/"+get_current_time()+'.log', 'w')
###

app = QApplication(sys.argv)
ex  = Window()
ex.show()
sys.exit(app.exec_())
