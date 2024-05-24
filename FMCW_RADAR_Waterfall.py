#!/usr/bin/env python3
#  Must use Python 3
# Copyright (C) 2022 Analog Devices, Inc. 
#
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without modification,
# are permitted provided that the following conditions are met:
#     - Redistributions of source code must retain the above copyright
#       notice, this list of conditions and the following disclaimer.
#     - Redistributions in binary form must reproduce the above copyright
#       notice, this list of conditions and the following disclaimer in
#       the documentation and/or other materials provided with the
#       distribution.
#     - Neither the name of Analog Devices, Inc. nor the names of its
#       contributors may be used to endorse or promote products derived
#       from this software without specific prior written permission.
#     - The use of this software may or may not infringe the patent rights
#       of one or more patent holders.  This license does not release you
#       from the requirement that you obtain separate licenses from these
#       patent holders to use this software.
#     - Use of the software either in source or binary form, must be run
#       on or directly connected to an Analog Devices Inc. component.
#
# THIS SOFTWARE IS PROVIDED BY ANALOG DEVICES "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES,
# INCLUDING, BUT NOT LIMITED TO, NON-INFRINGEMENT, MERCHANTABILITY AND FITNESS FOR A
# PARTICULAR PURPOSE ARE DISCLAIMED.
#
# IN NO EVENT SHALL ANALOG DEVICES BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, INTELLECTUAL PROPERTY
# RIGHTS, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR
# BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
# STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF
# THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

'''FMCW Radar Demo with Phaser (CN0566)
   Jon Kraft, Jan 20 2024'''
# Imports

import sys
import time
import matplotlib.pyplot as plt
import numpy
import numpy as np
import pyqtgraph as pg
import scipy

from utils import terminal_reader
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import *
from pyqtgraph.Qt import QtCore, QtGui


reader = terminal_reader.TerminalDataReader(port="COM3", start_marker=65535, end_marker=65534)
reader.start()
sample_rate = 500e3
signal_freq = 0
num_slices = 600  # this sets how much time will be displayed on the waterfall plot
fft_size = 1024*8
plot_freq = 100e3  # x-axis freq range to plot
img_array = np.ones((num_slices, fft_size)) * (-100)

# Configure the ADF4159 Rampling PLL
output_freq = 4.8e9
freq = output_freq
BW = 500e6
ramp_time_s = 0.002

# Print config
print(
    """
CONFIG:
Sample rate: {sample_rate}MHz
Num samples: 2^{Nlog2}
Bandwidth: {BW}MHz
Output frequency: {output_freq}MHz
IF: {signal_freq}kHz
""".format(
        sample_rate=sample_rate / 1e6,
        Nlog2=int(np.log2(fft_size)),
        BW=BW / 1e6,
        output_freq=output_freq / 1e6,
        signal_freq=signal_freq / 1e3,
    )
)

fs = int(sample_rate)
N = 1024*8
c = 3e8
default_chirp_bw = 500e6
N_frame = fft_size
freq = np.linspace(-fs / 2, fs / 2, int(N_frame))
slope = BW / ramp_time_s
dist = (freq - signal_freq) * c / (2 * slope)

plot_dist = False


class Window(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Interactive FFT")
        self.setGeometry(0, 0, 400, 400)  # (x,y, width, height)
        # self.setFixedWidth(600)
        self.setWindowState(QtCore.Qt.WindowMaximized)
        self.num_rows = 13
        self.setWindowFlag(QtCore.Qt.WindowCloseButtonHint, False)  # remove the window's close button
        self.UiComponents()
        self.show()

    # method for components
    def UiComponents(self):
        widget = QWidget()

        global layout, signal_freq
        layout = QGridLayout()

        # Control Panel
        control_label = QLabel("PHASER Simple FMCW Radar")
        font = control_label.font()
        font.setPointSize(24)
        control_label.setFont(font)
        font.setPointSize(12)
        control_label.setAlignment(Qt.AlignHCenter)  # | Qt.AlignVCenter)
        layout.addWidget(control_label, 0, 0, 1, 2)

        # Check boxes
        self.x_axis_check = QCheckBox("Convert to Distance")
        font = self.x_axis_check.font()
        font.setPointSize(10)
        self.x_axis_check.setFont(font)

        self.x_axis_check.stateChanged.connect(self.change_x_axis)
        layout.addWidget(self.x_axis_check, 2, 0)

        # Range resolution
        # Changes with the Chirp BW slider
        self.range_res_label = QLabel(
            "B: %0.2f MHz - R<sub>res</sub>: %0.2f m"
            % (default_chirp_bw / 1e6, c / (2 * default_chirp_bw))
        )
        font = self.range_res_label.font()
        font.setPointSize(10)
        self.range_res_label.setFont(font)
        self.range_res_label.setAlignment(Qt.AlignLeft)
        self.range_res_label.setMaximumWidth(200)
        self.range_res_label.setMinimumWidth(100)
        layout.addWidget(self.range_res_label, 4, 1)

        # Chirp bandwidth slider
        self.bw_slider = QSlider(Qt.Horizontal)
        self.bw_slider.setMinimum(100)
        self.bw_slider.setMaximum(500)
        self.bw_slider.setValue(int(default_chirp_bw / 1e6))
        self.bw_slider.setTickInterval(50)
        self.bw_slider.setMaximumWidth(200)
        self.bw_slider.setTickPosition(QSlider.TicksBelow)
        self.bw_slider.valueChanged.connect(self.get_range_res)
        layout.addWidget(self.bw_slider, 4, 0)

        self.set_bw = QPushButton("Set Chirp Bandwidth")
        self.set_bw.setMaximumWidth(200)
        self.set_bw.pressed.connect(self.set_range_res)
        layout.addWidget(self.set_bw, 5, 0, 1, 1)

        self.quit_button = QPushButton("Quit")
        self.quit_button.pressed.connect(self.end_program)
        layout.addWidget(self.quit_button, 30, 0, 4, 4)

        # waterfall level slider
        self.low_slider = QSlider(Qt.Horizontal)
        self.low_slider.setMinimum(-100)
        self.low_slider.setMaximum(0)
        self.low_slider.setValue(-45)
        self.low_slider.setTickInterval(20)
        self.low_slider.setMaximumWidth(200)
        self.low_slider.setTickPosition(QSlider.TicksBelow)
        self.low_slider.valueChanged.connect(self.get_water_levels)
        layout.addWidget(self.low_slider, 8, 0)

        self.high_slider = QSlider(Qt.Horizontal)
        self.high_slider.setMinimum(-100)
        self.high_slider.setMaximum(0)
        self.high_slider.setValue(-25)
        self.high_slider.setTickInterval(20)
        self.high_slider.setMaximumWidth(200)
        self.high_slider.setTickPosition(QSlider.TicksBelow)
        self.high_slider.valueChanged.connect(self.get_water_levels)
        layout.addWidget(self.high_slider, 10, 0)

        self.water_label = QLabel("Waterfall Intensity Levels")
        self.water_label.setFont(font)
        self.water_label.setAlignment(Qt.AlignCenter)
        self.water_label.setMinimumWidth(100)
        self.water_label.setMaximumWidth(200)
        layout.addWidget(self.water_label, 7, 0, 1, 1)
        self.low_label = QLabel("LOW LEVEL: %0.0f" % (self.low_slider.value()))
        self.low_label.setFont(font)
        self.low_label.setAlignment(Qt.AlignLeft)
        self.low_label.setMinimumWidth(100)
        self.low_label.setMaximumWidth(200)
        layout.addWidget(self.low_label, 8, 1)
        self.high_label = QLabel("HIGH LEVEL: %0.0f" % (self.high_slider.value()))
        self.high_label.setFont(font)
        self.high_label.setAlignment(Qt.AlignLeft)
        self.high_label.setMinimumWidth(100)
        self.high_label.setMaximumWidth(200)
        layout.addWidget(self.high_label, 10, 1)

        self.steer_slider = QSlider(Qt.Horizontal)
        self.steer_slider.setMinimum(-80)
        self.steer_slider.setMaximum(80)
        self.steer_slider.setValue(0)
        self.steer_slider.setTickInterval(20)
        self.steer_slider.setMaximumWidth(200)
        self.steer_slider.setTickPosition(QSlider.TicksBelow)
        self.steer_slider.valueChanged.connect(self.get_steer_angle)
        layout.addWidget(self.steer_slider, 14, 0)
        self.steer_title = QLabel("Receive Steering Angle")
        self.steer_title.setFont(font)
        self.steer_title.setAlignment(Qt.AlignCenter)
        self.steer_title.setMinimumWidth(100)
        self.steer_title.setMaximumWidth(200)
        layout.addWidget(self.steer_title, 13, 0)
        self.steer_label = QLabel("%0.0f DEG" % (self.steer_slider.value()))
        self.steer_label.setFont(font)
        self.steer_label.setAlignment(Qt.AlignLeft)
        self.steer_label.setMinimumWidth(100)
        self.steer_label.setMaximumWidth(200)
        layout.addWidget(self.steer_label, 14, 1, 1, 2)

        # Signal plot
        self.signal_plot = pg.plot()
        self.signal_plot.setMinimumWidth(400)
        self.signal_curve = self.signal_plot.plot(freq, pen={'color': 'y', 'width': 2})
        title_style = {"size": "20pt"}
        label_style = {"color": "#FFF", "font-size": "14pt"}
        self.signal_plot.setLabel("bottom", text="Time", units="conv", **label_style)
        self.signal_plot.setLabel("left", text="Value", units="", **label_style)
        self.signal_plot.setTitle("Received Signal1", **title_style)
        layout.addWidget(self.signal_plot, self.num_rows, 2, 1, 1)
        self.signal_plot.setYRange(0, 4096)
        self.signal_plot.setXRange(0, fft_size)

        # FFT plot
        self.fft_plot = pg.plot()
        self.fft_plot.setMinimumWidth(400)
        self.fft_curve = self.fft_plot.plot(freq, pen={'color': 'y', 'width': 2})
        title_style = {"size": "20pt"}
        label_style = {"color": "#FFF", "font-size": "14pt"}
        self.fft_plot.setLabel("bottom", text="Frequency", units="Hz", **label_style)
        self.fft_plot.setLabel("left", text="Magnitude", units="dB", **label_style)
        self.fft_plot.setTitle("Received Signal - Frequency Spectrum", **title_style)
        layout.addWidget(self.fft_plot, 0, 2, self.num_rows, 1)
        self.fft_plot.setYRange(-60, 0)
        self.fft_plot.setXRange(signal_freq, signal_freq+plot_freq)

        # Waterfall plot
        self.waterfall = pg.PlotWidget()
        self.imageitem = pg.ImageItem()
        self.waterfall.addItem(self.imageitem)
        # Use a viridis colormap
        pos = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
        color = np.array(
            [[68, 1, 84, 255], [59, 82, 139, 255], [33, 145, 140, 255], [94, 201, 98, 255], [253, 231, 37, 255]],
            dtype=np.ubyte)
        lut = pg.ColorMap(pos, color).getLookupTable(0.0, 1.0, 256)
        self.imageitem.setLookupTable(lut)
        self.imageitem.setLevels([0, 1])
        # self.imageitem.scale(0.35, sample_rate / (N))  # this is deprecated -- we have to use setTransform instead
        tr = QtGui.QTransform()
        tr.translate(0, -sample_rate / 2)
        tr.scale(0.35, sample_rate / (N))
        self.imageitem.setTransform(tr)
        zoom_freq = 35e3
        self.waterfall.setRange(yRange=(signal_freq, signal_freq + zoom_freq))
        self.waterfall.setTitle("Waterfall Spectrum", **title_style)
        self.waterfall.setLabel("left", "Frequency", units="Hz", **label_style)
        self.waterfall.setLabel("bottom", "Time", units="sec", **label_style)
        layout.addWidget(self.waterfall, 0 + self.num_rows + 1, 2, self.num_rows, 1)
        self.img_array = np.ones((num_slices, fft_size)) * (-100)

        widget.setLayout(layout)
        # setting this widget as central widget of the main window
        self.setCentralWidget(widget)

    def get_range_res(self):
        """ Updates the slider bar label with RF bandwidth and range resolution
		Returns:
			None
		"""
        bw = self.bw_slider.value() * 1e6
        range_res = c / (2 * bw)
        self.range_res_label.setText(
            "B: %0.2f MHz - R<sub>res</sub>: %0.2f m"
            % (bw / 1e6, c / (2 * bw))
        )

    def get_water_levels(self):
        """ Updates the waterfall intensity levels
		Returns:
			None
		"""
        if self.low_slider.value() > self.high_slider.value():
            self.low_slider.setValue(self.high_slider.value())
        self.low_label.setText("LOW LEVEL: %0.0f" % (self.low_slider.value()))
        self.high_label.setText("HIGH LEVEL: %0.0f" % (self.high_slider.value()))

    def get_steer_angle(self):
        """ Updates the steering angle readout
		Returns:
			None
		"""
        self.steer_label.setText("%0.0f DEG" % (self.steer_slider.value()))
        phase_delta = (
                2
                * 3.14159
                * 10.25e9
                * 0.014
                * np.sin(np.radians(self.steer_slider.value()))
                / (3e8)
        )
        # my_phaser.set_beam_phase_diff(np.degrees(phase_delta))

    def set_range_res(self):
        """ Sets the Chirp bandwidth
		Returns:
			None
		"""
        global dist, slope, signal_freq, plot_freq
        bw = self.bw_slider.value() * 1e6
        slope = bw / ramp_time_s
        dist = (freq - signal_freq) * c / (2 * slope)
        if self.x_axis_check.isChecked() == True:
            plot_dist = True
            range_x = (plot_freq) * c / (2 * slope)
            self.fft_plot.setXRange(0, range_x)
        else:
            plot_dist = False
            self.fft_plot.setXRange(signal_freq, signal_freq + plot_freq)
        # my_phaser.freq_dev_range = int(bw / 4)  # frequency deviation range in Hz
        # my_phaser.enable = 0

    def end_program(self):
        """ Gracefully shutsdown the program and Pluto
		Returns:
			None
		"""
        # my_sdr.tx_destroy_buffer()
        self.close()

    def change_x_axis(self, state):
        """ Toggles between showing frequency and range for the x-axis
		Args:
			state (QtCore.Qt.Checked) : State of check box
		Returns:
			None
		"""
        global plot_dist, slope, signal_freq, plot_freq
        plot_state = win.fft_plot.getViewBox().state
        if state == QtCore.Qt.Checked:
            plot_dist = True
            range_x = (plot_freq) * c / (2 * slope)
            self.fft_plot.setXRange(0, range_x)
        else:
            plot_dist = False
            self.fft_plot.setXRange(signal_freq, signal_freq + plot_freq)


# create pyqt5 app
App = QApplication(sys.argv)

# create the instance of our Window
win = Window()
index = 0


def update():
    """ Updates the FFT in the window
	Returns:
		None
	"""
    global index, plot_dist, freq, dist
    label_style = {"color": "#FFF", "font-size": "14pt"}

    data = reader.read()  # my_sdr.rx()

    #data = scipy.signal.hilbert(data, 1024)
#    data = data[0] + data[1]
    if len(data) == 0:
        return
    average = numpy.average(data)
    data_raw = data
    data -= average
    win_funct = np.blackman(len(data))
    y = data * win_funct
    sp = np.absolute(np.fft.fft(y))
    sp = np.fft.fftshift(sp)
    s_mag = np.abs(sp) / np.sum(win_funct)
    s_mag = np.maximum(s_mag, 10 ** (-15))
    s_dbfs = 20 * np.log10(s_mag / (2 ** 11))
    win.signal_curve.setData(np.arange(fft_size), data_raw)
    if plot_dist:
        win.fft_curve.setData(dist, s_dbfs)
        win.fft_plot.setLabel("bottom", text="Distance", units="m", **label_style)
    else:
        win.fft_curve.setData(freq, s_dbfs)
        win.fft_plot.setLabel("bottom", text="Frequency", units="Hz", **label_style)

    win.img_array = np.roll(win.img_array, 1, axis=0)
    win.img_array[0] = s_dbfs
    win.imageitem.setLevels([win.low_slider.value(), win.high_slider.value()])
    win.imageitem.setImage(win.img_array, autoLevels=False)

    if index == 1:
        win.fft_plot.enableAutoRange("xy", False)
    index = index + 1


timer = QtCore.QTimer()
timer.timeout.connect(update)
timer.start(0)


# start the app
sys.exit(App.exec())
