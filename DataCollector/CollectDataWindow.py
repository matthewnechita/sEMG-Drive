"""
Data Collector GUI
This is the GUI that lets you connect to a base, scan via rf for sensors, and stream data from them in real time.
"""

import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple
from PySide6.QtCore import *
from PySide6.QtGui import *
from PySide6.QtWidgets import *
from DataCollector.CollectDataController import *
import tkinter as tk
from tkinter import filedialog
import numpy as np
from project_paths import STRICT_DATA_ROOT, strict_raw_dir

from DataCollector.CollectionMetricsManagement import CollectionMetricsManagement
from Plotter import GenericPlot as gp


@dataclass
class TrialConfig:
    gestures: List[str]
    gesture_duration: float
    neutral_duration: float
    repetitions: int
    subject: str
    session: str
    protocol_name: str = "standard"
    arm: str = "right"
    prep_duration: float = 3.0
    inter_gesture_rest_s: float = 0.0
    label_trim_s: float = 0.0
    rest_label_trim_s: Optional[float] = None
    recovery_neutral_lead_trim_s: float = 0.0
    recovery_neutral_trail_trim_s: float = 0.0
    calibrate: bool = False
    calibration_neutral_s: float = 5.0
    calibration_mvc_s: float = 5.0
    calibration_mvc_prep_s: float = 2.0  # countdown before MVC window starts
    calibration_min_ratio: float = 2.0   # warn if median MVC/neutral ratio below this


def resolve_rest_label_trim(
    label_trim_s: float,
    rest_label_trim_s: Optional[float],
    rest_duration_s: float,
) -> float:
    if rest_label_trim_s is not None:
        return max(0.0, float(rest_label_trim_s))
    if label_trim_s <= 0.0 or rest_duration_s <= 0.0:
        return 0.0
    return min(label_trim_s, rest_duration_s * 0.25)


class CollectDataWindow(QWidget):
    plot_enabled = False
    DEFAULT_WINDOW_SIZE = QSize(1280, 720)
    MINIMUM_WINDOW_SIZE = QSize(980, 660)
    SCREEN_MARGIN = 24

    def __init__(self, controller):
        QWidget.__init__(self)
        self.pipelinetext = "Off"
        self.controller = controller
        self._geometry_adjustment_pending = False
        self._geometry_adjustment_active = False
        self._pending_center_adjust = False
        self._initial_geometry_applied = False
        # Default identifiers for saving data; editable in UI
        self.default_subject = "subject01"
        self.default_session = "01"
        self.buttonPanel = self.ButtonPanel()
        self.plotPanel = None
        self.collectionLabelPanel = self.CollectionLabelPanel()
        self.instructionsPanel = self.InstructionPanel()
        self.channel_labels_layout = None
        self.channel_labels_container = None
        self.x_axis_label = None
        self.y_axis_label = None

        self.grid = QGridLayout(self)

        self.MetricsConnector = CollectionMetricsManagement()
        self.collectionLabelPanel.setFixedHeight(275)
        self.MetricsConnector.collectionmetrics.setFixedHeight(275)

        self.metricspanel = QWidget()
        self.metricspane = QHBoxLayout()
        self.metricspane.setAlignment(Qt.AlignmentFlag.AlignTop)
        self.metricspane.addWidget(self.collectionLabelPanel)
        self.metricspane.addWidget(self.MetricsConnector.collectionmetrics)
        self.metricspanel.setLayout(self.metricspane)
        self.metricspanel.setFixedWidth(400)
        self.grid.addWidget(self.buttonPanel, 0, 0)
        self.grid.addWidget(self.metricspanel, 0, 1)
        self.grid.addWidget(self.instructionsPanel, 1, 0, 1, 2)
        # Let the button/sensor column breathe so sensor settings aren't cut off
        self.grid.setColumnStretch(0, 3)
        self.grid.setColumnStretch(1, 2)
        self.grid.setColumnStretch(2, 4)

        self.setStyleSheet("background-color:#3d4c51;")
        self.setLayout(self.grid)
        self.setWindowTitle("Collect Data GUI")
        self.resize(self.DEFAULT_WINDOW_SIZE)
        self.setMinimumSize(self.MINIMUM_WINDOW_SIZE)
        self.pairing = False
        self.selectedSensor = None
        self.protocol_running = False
        self.protocol_abort = False

    def showEvent(self, event):
        super().showEvent(event)
        self._schedule_window_fit(center=not self._initial_geometry_applied)
        self._initial_geometry_applied = True

    def moveEvent(self, event):
        super().moveEvent(event)
        self._schedule_window_fit()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._schedule_window_fit()

    def _schedule_window_fit(self, center: bool = False):
        if not self.isVisible() or self.isMaximized() or self.isFullScreen():
            return
        if self._geometry_adjustment_active:
            self._pending_center_adjust = self._pending_center_adjust or center
            return
        if self._geometry_adjustment_pending:
            self._pending_center_adjust = self._pending_center_adjust or center
            return

        self._geometry_adjustment_pending = True
        self._pending_center_adjust = center
        QTimer.singleShot(0, self._fit_window_to_screen)

    def _available_screen_geometry(self):
        screen = self.screen()
        if screen is None:
            screen = QGuiApplication.primaryScreen()
        if screen is None:
            return None
        return screen.availableGeometry().adjusted(
            self.SCREEN_MARGIN,
            self.SCREEN_MARGIN,
            -self.SCREEN_MARGIN,
            -self.SCREEN_MARGIN,
        )

    def _fit_window_to_screen(self):
        self._geometry_adjustment_pending = False

        if not self.isVisible() or self.isMaximized() or self.isFullScreen():
            self._pending_center_adjust = False
            return

        available = self._available_screen_geometry()
        if available is None:
            self._pending_center_adjust = False
            return

        target_width = min(self.width(), available.width())
        target_height = min(self.height(), available.height())

        self._geometry_adjustment_active = True
        try:
            if target_width != self.width() or target_height != self.height():
                self.resize(target_width, target_height)

            if self._pending_center_adjust:
                centered = QStyle.alignedRect(
                    Qt.LayoutDirection.LeftToRight,
                    Qt.AlignmentFlag.AlignCenter,
                    self.size(),
                    available,
                )
                self.move(centered.topLeft())
            else:
                geometry = self.geometry()
                max_x = available.x() + max(0, available.width() - geometry.width())
                max_y = available.y() + max(0, available.height() - geometry.height())
                clamped_x = min(max(geometry.x(), available.x()), max_x)
                clamped_y = min(max(geometry.y(), available.y()), max_y)
                if clamped_x != geometry.x() or clamped_y != geometry.y():
                    self.move(clamped_x, clamped_y)
        finally:
            self._geometry_adjustment_active = False
            self._pending_center_adjust = False

    def AddPlotPanel(self):
        self.plotPanel = self.Plotter()
        self.grid.addWidget(self.plotPanel, 0, 2)

    def SetCallbackConnector(self):
        if self.plot_enabled:
            self.CallbackConnector = PlottingManagement(self, self.MetricsConnector, self.plotCanvas)
        else:
            self.CallbackConnector = PlottingManagement(self, self.MetricsConnector)

    # -----------------------------------------------------------------------
    # ---- GUI Components
    def ButtonPanel(self):
        buttonPanel = QWidget()
        buttonPanel.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        buttonLayout = QVBoxLayout()
        buttonLayout.setContentsMargins(0, 0, 0, 0)
        buttonLayout.setSpacing(8)
        findSensor_layout = QHBoxLayout()
        # ---- Pair Button
        self.pair_button = QPushButton('Pair', self)
        self.pair_button.setToolTip('Pair Sensors')
        self.pair_button.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)
        self.pair_button.objectName = 'Pair'
        self.pair_button.clicked.connect(self.pair_callback)
        self.pair_button.setStyleSheet('QPushButton {color: grey;}')
        self.pair_button.setEnabled(False)
        self.pair_button.setFixedHeight(50)
        findSensor_layout.addWidget(self.pair_button)

        # ---- Scan Button
        self.scan_button = QPushButton('Scan', self)
        self.scan_button.setToolTip('Scan for Sensors')
        self.scan_button.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)
        self.scan_button.objectName = 'Scan'
        self.scan_button.clicked.connect(self.scan_callback)
        self.scan_button.setStyleSheet('QPushButton {color: grey;}')
        self.scan_button.setEnabled(False)
        self.scan_button.setFixedHeight(50)
        findSensor_layout.addWidget(self.scan_button)

        buttonLayout.addLayout(findSensor_layout)

        triggerLayout = QHBoxLayout()

        self.starttriggerlabel = QLabel('Start Trigger', self)
        self.starttriggerlabel.setStyleSheet("color : grey")
        triggerLayout.addWidget(self.starttriggerlabel)
        self.starttriggercheckbox = QCheckBox()
        self.starttriggercheckbox.setEnabled(False)
        triggerLayout.addWidget(self.starttriggercheckbox)
        self.stoptriggerlabel = QLabel('Stop Trigger', self)
        self.stoptriggerlabel.setStyleSheet("color : grey")
        triggerLayout.addWidget(self.stoptriggerlabel)
        self.stoptriggercheckbox = QCheckBox()
        self.stoptriggercheckbox.setEnabled(False)
        triggerLayout.addWidget(self.stoptriggercheckbox)

        buttonLayout.addLayout(triggerLayout)

        # ---- Start Button
        self.start_button = QPushButton('Start', self)
        self.start_button.setToolTip('Start Sensor Stream')
        self.start_button.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)
        self.start_button.objectName = 'Start'
        self.start_button.clicked.connect(self.start_callback)
        self.start_button.setStyleSheet('QPushButton {color: grey;}')
        self.start_button.setEnabled(False)
        self.start_button.setFixedHeight(50)
        buttonLayout.addWidget(self.start_button)

        # ---- Stop Button
        self.stop_button = QPushButton('Stop', self)
        self.stop_button.setToolTip('Stop Sensor Stream')
        self.stop_button.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)
        self.stop_button.objectName = 'Stop'
        self.stop_button.clicked.connect(self.stop_callback)
        self.stop_button.setStyleSheet('QPushButton {color: grey;}')
        self.stop_button.setEnabled(False)
        self.stop_button.setFixedHeight(50)
        buttonLayout.addWidget(self.stop_button)

        # ---- Export CSV Button
        self.exportcsv_button = QPushButton('Export CSV', self)
        self.exportcsv_button.setToolTip('Export collected data to project root - data.csv')
        self.exportcsv_button.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)
        self.exportcsv_button.objectName = 'Export'
        self.exportcsv_button.clicked.connect(self.exportcsv_callback)
        self.exportcsv_button.setStyleSheet('QPushButton {color: grey;}')
        self.exportcsv_button.setEnabled(False)
        self.exportcsv_button.setFixedHeight(50)
        buttonLayout.addWidget(self.exportcsv_button)

        # --- Subject / Session inputs (inline instead of popups)
        idWidget = QWidget()
        idLayout = QFormLayout()
        idLayout.setFormAlignment(Qt.AlignLeft)
        idLayout.setLabelAlignment(Qt.AlignLeft)
        idLayout.setContentsMargins(0, 0, 0, 0)
        idLayout.setHorizontalSpacing(8)

        self.subject_input = QLineEdit(self.default_subject)
        self.subject_input.setPlaceholderText("subject01")
        self.subject_input.setStyleSheet("color: white; background:#6b6b6b;")
        self.subject_input.setMinimumWidth(160)

        self.session_input = QLineEdit(self.default_session)
        self.session_input.setPlaceholderText("01")
        self.session_input.setStyleSheet("color: white; background:#6b6b6b;")
        self.session_input.setMinimumWidth(120)

        subj_label = QLabel("Subject ID:")
        subj_label.setStyleSheet("color: white;")
        sess_label = QLabel("Session #:")
        sess_label.setStyleSheet("color: white;")

        idLayout.addRow(subj_label, self.subject_input)
        idLayout.addRow(sess_label, self.session_input)
        idWidget.setLayout(idLayout)
        buttonLayout.addWidget(idWidget)

        # ---- Arm toggle (Right / Left)
        armWidget = QWidget()
        armLayout = QHBoxLayout()
        armLayout.setContentsMargins(0, 0, 0, 0)
        arm_label = QLabel("Arm:")
        arm_label.setStyleSheet("color: white;")
        armLayout.addWidget(arm_label)
        self.arm_right_radio = QRadioButton("Right")
        self.arm_right_radio.setStyleSheet("color: white;")
        self.arm_right_radio.setChecked(True)
        self.arm_left_radio = QRadioButton("Left")
        self.arm_left_radio.setStyleSheet("color: white;")
        armLayout.addWidget(self.arm_right_radio)
        armLayout.addWidget(self.arm_left_radio)
        armLayout.addStretch()
        armWidget.setLayout(armLayout)
        buttonLayout.addWidget(armWidget)

        # ---- Labeled Protocol (NPZ) Button
        self.recordnpz_button = QPushButton('Run Protocol + Save NPZ', self)
        self.recordnpz_button.setToolTip('Run scripted gesture/rest protocol and save .npz with labels')
        self.recordnpz_button.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)
        self.recordnpz_button.clicked.connect(self.protocol_callback)
        self.recordnpz_button.setStyleSheet('QPushButton {color: grey;}')
        self.recordnpz_button.setEnabled(False)
        self.recordnpz_button.setFixedHeight(50)
        buttonLayout.addWidget(self.recordnpz_button)

        # ---- Neutral Recovery Protocol Button
        self.neutral_recovery_button = QPushButton('Run Neutral Recovery Protocol', self)
        self.neutral_recovery_button.setToolTip(
            'Collect repeated left/right gesture releases into neutral without changing the standard protocol'
        )
        self.neutral_recovery_button.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)
        self.neutral_recovery_button.clicked.connect(self.neutral_recovery_protocol_callback)
        self.neutral_recovery_button.setStyleSheet('QPushButton {color: grey;}')
        self.neutral_recovery_button.setEnabled(False)
        self.neutral_recovery_button.setFixedHeight(50)
        buttonLayout.addWidget(self.neutral_recovery_button)

        # ---- Drop-down menu of sensor modes
        self.SensorModeList = QComboBox(self)
        self.SensorModeList.setToolTip('Sensor Modes')
        self.SensorModeList.objectName = 'PlaceHolder'
        self.SensorModeList.setStyleSheet('QComboBox {color: white;background: #848482}')
        self.SensorModeList.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)
        # Widen the popup so long mode names aren't truncated
        popup_view = QListView()
        popup_view.setMinimumWidth(680)
        popup_view.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        popup_view.setWordWrap(True)
        popup_view.setTextElideMode(Qt.ElideNone)
        self.SensorModeList.setView(popup_view)
        self.SensorModeList.setMinimumContentsLength(45)
        self.SensorModeList.setMinimumWidth(520)
        self.SensorModeList.setFixedHeight(40)
        self.SensorModeList.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        buttonLayout.addWidget(self.SensorModeList)

        # ---- List of detected sensors
        self.SensorListBox = QListWidget(self)
        self.SensorListBox.setToolTip('Sensor List')
        self.SensorListBox.objectName = 'PlaceHolder'
        self.SensorListBox.setStyleSheet('QListWidget {color: white;background:#848482}')
        self.SensorListBox.itemClicked.connect(self.sensorList_callback)
        self.SensorListBox.setMinimumWidth(360)
        self.SensorListBox.setMinimumHeight(260)
        self.SensorListBox.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.SensorListBox.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        buttonLayout.addWidget(self.SensorListBox)
        buttonPanel.setLayout(buttonLayout)
        buttonPanel.setMinimumWidth(400)
        return buttonPanel

    def InstructionPanel(self):
        panel = QWidget()
        layout = QVBoxLayout()
        layout.setAlignment(Qt.AlignmentFlag.AlignTop)

        self.instruction_label = QLabel("Protocol idle.")
        self.instruction_label.setWordWrap(True)
        self.instruction_label.setStyleSheet("color:white; font-size:14pt; font-weight:bold;")

        self.next_label = QLabel("Next: --")
        self.next_label.setWordWrap(True)
        self.next_label.setStyleSheet("color:white; font-size:12pt;")

        self.timer_label = QLabel("Timer: --")
        self.timer_label.setStyleSheet("color:white; font-size:12pt;")

        self.reps_label = QLabel("Reps remaining: --")
        self.reps_label.setStyleSheet("color:white; font-size:12pt;")

        layout.addWidget(self.instruction_label)
        layout.addWidget(self.next_label)
        layout.addWidget(self.timer_label)
        layout.addWidget(self.reps_label)
        panel.setLayout(layout)
        return panel

    def Plotter(self):
        widget = QWidget()
        grid = QGridLayout()
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setHorizontalSpacing(8)
        grid.setVerticalSpacing(6)

        plot_mode = 'windowed'  # Select between 'scrolling' and 'windowed'
        pc = gp.GenericPlot(plot_mode)
        pc.native.objectName = 'vispyCanvas'
        pc.native.parent = self

        self.y_axis_label = QLabel("EMG amplitude")
        self.y_axis_label.setAlignment(Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignVCenter)
        self.y_axis_label.setStyleSheet("color:white; font-size:10pt; font-weight:bold;")
        self.y_axis_label.setWordWrap(True)
        self.y_axis_label.setMinimumWidth(28)

        self.channel_labels_container = QWidget()
        self.channel_labels_layout = QVBoxLayout()
        self.channel_labels_layout.setContentsMargins(0, 0, 0, 0)
        self.channel_labels_layout.setSpacing(0)
        self.channel_labels_layout.setAlignment(Qt.AlignmentFlag.AlignVCenter)
        self.channel_labels_container.setLayout(self.channel_labels_layout)
        self.channel_labels_container.setMinimumWidth(180)
        self.update_channel_labels([])

        grid.addWidget(self.y_axis_label, 0, 0, 2, 1, alignment=Qt.AlignmentFlag.AlignHCenter)
        grid.addWidget(self.channel_labels_container, 0, 1)
        grid.addWidget(pc.native, 0, 2)

        self.x_axis_label = QLabel("Time (s)")
        self.x_axis_label.setAlignment(Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignTop)
        self.x_axis_label.setStyleSheet("color:white; font-size:10pt; font-weight:bold;")
        self.x_axis_label.setFixedHeight(20)

        label = QLabel("*This Demo plots EMG Channels only")
        label.setStyleSheet('.QLabel { font-size: 8pt;}')
        label.setFixedHeight(20)
        label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignTop)

        grid.addWidget(label, 1, 1, alignment=Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignTop)
        grid.addWidget(self.x_axis_label, 1, 2, alignment=Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignTop)

        grid.setRowStretch(0, 1)
        grid.setColumnStretch(2, 1)
        widget.setLayout(grid)
        self.plotCanvas = pc

        return widget

    def update_channel_labels(self, labels: List[str]):
        if self.channel_labels_layout is None:
            return

        while self.channel_labels_layout.count():
            item = self.channel_labels_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()

        if not labels:
            placeholder = QLabel("Connect sensors to see channel names")
            placeholder.setStyleSheet("color: white; font-size:10pt;")
            placeholder.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)
            placeholder.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
            self.channel_labels_layout.addWidget(placeholder)
            self.channel_labels_layout.setStretch(0, 1)
            return

        for idx, text in enumerate(labels):
            channel_label = QLabel(text)
            channel_label.setStyleSheet("color: white; font-size:10pt;")
            channel_label.setAlignment(Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft)
            channel_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
            channel_label.setWordWrap(True)
            self.channel_labels_layout.addWidget(channel_label)
            self.channel_labels_layout.setStretch(idx, 1)

    def CollectionLabelPanel(self):
        collectionLabelPanel = QWidget()
        collectionlabelsLayout = QVBoxLayout()

        pipelinelabel = QLabel('Pipeline State:')
        pipelinelabel.setAlignment(Qt.AlignCenter | Qt.AlignRight)
        pipelinelabel.setStyleSheet("color:white")
        collectionlabelsLayout.addWidget(pipelinelabel)

        sensorsconnectedlabel = QLabel('Sensors Connected:', self)
        sensorsconnectedlabel.setAlignment(Qt.AlignCenter | Qt.AlignRight)
        sensorsconnectedlabel.setStyleSheet("color:white")
        collectionlabelsLayout.addWidget(sensorsconnectedlabel)

        totalchannelslabel = QLabel('Total Channels:', self)
        totalchannelslabel.setAlignment(Qt.AlignCenter | Qt.AlignRight)
        totalchannelslabel.setStyleSheet("color:white")
        collectionlabelsLayout.addWidget(totalchannelslabel)

        framescollectedlabel = QLabel('Frames Collected:', self)
        framescollectedlabel.setAlignment(Qt.AlignCenter | Qt.AlignRight)
        framescollectedlabel.setStyleSheet("color:white")
        collectionlabelsLayout.addWidget(framescollectedlabel)

        collectionLabelPanel.setFixedWidth(200)
        collectionLabelPanel.setLayout(collectionlabelsLayout)

        return collectionLabelPanel

    # -----------------------------------------------------------------------
    # ---- Callback Functions
    def getpipelinestate(self):
        self.pipelinetext = self.CallbackConnector.base.PipelineState_Callback()
        self.MetricsConnector.pipelinestatelabel.setText(self.pipelinetext)

    def connect_callback(self):
        self.CallbackConnector.base.Connect_Callback()

        self.pair_button.setEnabled(True)
        self.pair_button.setStyleSheet('QPushButton {color: white;}')
        self.scan_button.setEnabled(True)
        self.scan_button.setStyleSheet('QPushButton {color: white;}')
        self.starttriggerlabel.setStyleSheet("color : white")
        self.stoptriggerlabel.setStyleSheet("color : white")
        self.starttriggercheckbox.setEnabled(True)
        self.stoptriggercheckbox.setEnabled(True)
        self.getpipelinestate()
        self.MetricsConnector.pipelinestatelabel.setText(self.pipelinetext + " (Base Connected)")

    def pair_callback(self):
        """Pair button callback"""
        self.Pair_Window()
        self.getpipelinestate()
        self.exportcsv_button.setEnabled(False)
        self.exportcsv_button.setStyleSheet("color : gray")

    def Pair_Window(self):
        """Open pair sensor window to set pair number and begin pairing process"""
        pair_number, pressed = QInputDialog.getInt(QWidget(), "Input Pair Number", "Pair Number:",
                                                   1, 0, 100, 1)
        if pressed:
            self.pairing = True
            self.pair_canceled = False
            self.CallbackConnector.base.pair_number = pair_number
            self.PairThreadManager()

    def PairThreadManager(self):
        """Start t1 thread to begin pairing operation in DelsysAPI
           Start t2 thread to await result of CheckPairStatus() to return False
           Once threads begin, display awaiting sensor pair request window/countdown"""

        self.t1 = threading.Thread(target=self.CallbackConnector.base.Pair_Callback)
        self.t1.start()

        self.t2 = threading.Thread(target=self.awaitPairThread)
        self.t2.start()

        self.BeginPairingUISequence()


    def BeginPairingUISequence(self):
        """The awaiting sensor window will stay open until either:
           A) The pairing countdown timer completes (The end of the countdown will send a CancelPair request to the DelsysAPI)
           or...
           B) A sensor has been paired to the base (via self.pairing flag set by DelsysAPI CheckPairStatus() bool)

           If a sensor is paired, ask the user if they want to pair another sensor (No = start a scan for all previously paired sensors)
        """

        pair_success = False
        self.pair_countdown_seconds = 15

        awaitingPairWindow = QDialog()
        awaitingPairWindow.setWindowTitle(
            "Sensor (" + str(self.CallbackConnector.base.pair_number) + ") Awaiting sensor pair request. . . Cancel in: " + str(self.pair_countdown_seconds))
        awaitingPairWindow.setFixedWidth(500)
        awaitingPairWindow.setFixedHeight(80)
        awaitingPairWindow.show()

        while self.pair_countdown_seconds > 0:
            if self.pairing:
                time.sleep(1)
                self.pair_countdown_seconds -= 1
                self.UpdateTimerUI(awaitingPairWindow)
            else:
                pair_success = True
                break

        awaitingPairWindow.close()
        if not pair_success:
            self.CallbackConnector.base.TrigBase.CancelPair()
        else:
            self.ShowPairAnotherSensorDialog()

    def awaitPairThread(self):
        """ Wait for a sensor to be paired
        Once PairSensor() command is sent to the DelsysAPI, CheckPairStatus() will return True until a sensor has been paired to the base"""
        time.sleep(1)
        while self.pairing:
            pairstatus = self.CallbackConnector.base.CheckPairStatus()
            if not pairstatus:
                self.pairing = False

    def UpdateTimerUI(self, awaitingPairWindow):
        awaitingPairWindow.setWindowTitle(
            "Sensor (" + str(self.CallbackConnector.base.pair_number) + ") Awaiting sensor pair request. . . Cancel in: " + str(self.pair_countdown_seconds))

    def ShowPairAnotherSensorDialog(self):
        messagebox = QMessageBox()
        messagebox.setText("Pair another sensor?")
        messagebox.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
        messagebox.setIcon(QMessageBox.Question)
        button = messagebox.exec_()

        if button == QMessageBox.Yes:
            self.Pair_Window()
        else:
            self.scan_callback()

    def scan_callback(self):
        sensorList = self.CallbackConnector.base.Scan_Callback()

        self.set_sensor_list_box(sensorList)

        if len(sensorList) > 0:
            self.start_button.setEnabled(True)
            self.start_button.setStyleSheet("color : white")
            self.stop_button.setEnabled(True)
            self.stop_button.setStyleSheet("color : white")
            self.recordnpz_button.setEnabled(True)
            self.recordnpz_button.setStyleSheet("color : white")
            self.neutral_recovery_button.setEnabled(True)
            self.neutral_recovery_button.setStyleSheet("color : white")
            self.MetricsConnector.sensorsconnected.setText(str(len(sensorList)))
            self.starttriggercheckbox.setEnabled(True)
            self.stoptriggercheckbox.setEnabled(True)
        self.getpipelinestate()
        self.exportcsv_button.setEnabled(False)
        self.exportcsv_button.setStyleSheet("color : gray")

    def set_sensor_list_box(self, sensorList):
        self.SensorListBox.clear()

        number_and_names_str = []
        for i in range(len(sensorList)):
            base = getattr(self, "CallbackConnector", None)
            cur_mode = ""
            try:
                cur_mode = base.base.getCurMode(i) if base else ""
            except Exception:
                cur_mode = ""

            header = "(" + str(sensorList[i].PairNumber) + ") " + sensorList[i].FriendlyName
            if cur_mode:
                header += "\n    Mode: " + cur_mode
            number_and_names_str.append(header)
            for j in range(len(sensorList[i].TrignoChannels)):
                ch = sensorList[i].TrignoChannels[j]
                if not ch.IsEnabled:
                    continue
                ch_type = str(ch.Type)
                # Only show EMG channels to avoid ACC/IMU noise in the list
                if ch_type != "EMG":
                    continue
                number_and_names_str[i] += "\n     - " + ch.Name

        self.SensorListBox.addItems(number_and_names_str)

    def start_callback(self):
        self.CallbackConnector.base.Start_Callback(self.starttriggercheckbox.isChecked(),
                                                   self.stoptriggercheckbox.isChecked())
        self.CallbackConnector.resetmetrics()
        self.starttriggercheckbox.setEnabled(False)
        self.stoptriggercheckbox.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.exportcsv_button.setEnabled(False)
        self.exportcsv_button.setStyleSheet("color : gray")
        self.getpipelinestate()

    def stop_callback(self):
        if self.protocol_running:
            self.protocol_abort = True
            self.update_instruction("Stopping protocol...", None, 0, 0)
            try:
                self.CallbackConnector.base.Stop_Callback()
            except Exception:
                pass
            return

        self.CallbackConnector.base.Stop_Callback()
        self.getpipelinestate()
        self.exportcsv_button.setEnabled(True)
        self.exportcsv_button.setStyleSheet("color : white")

    def exportcsv_callback(self):
        export = None
        if self.CallbackConnector.streamYTData:
            export = self.CallbackConnector.base.csv_writer.exportYTCSV()
        else:
            export = self.CallbackConnector.base.csv_writer.exportCSV()
        self.getpipelinestate()
        print("CSV Export: " + str(export))

    def sensorList_callback(self):
        current_selected = self.SensorListBox.currentRow()
        if self.selectedSensor is None or self.selectedSensor != current_selected:
            if self.selectedSensor is not None:
                self.SensorModeList.currentIndexChanged.disconnect(self.sensorModeList_callback)
            self.selectedSensor = self.SensorListBox.currentRow()
            modeList = self.CallbackConnector.base.getSampleModes(self.selectedSensor)
            curMode = self.CallbackConnector.base.getCurMode(self.selectedSensor)

            if curMode is not None:
                self.resetModeList(modeList)
                if curMode in modeList:
                    self.SensorModeList.setCurrentText(curMode)
                elif len(modeList) > 0:
                    # If the current mode isn't available, default to the first option
                    self.SensorModeList.setCurrentText(modeList[0])
                self.starttriggercheckbox.setEnabled(True)
                self.stoptriggercheckbox.setEnabled(True)
                self.SensorModeList.currentIndexChanged.connect(self.sensorModeList_callback)

    def resetModeList(self, mode_list):
        self.SensorModeList.clear()
        self.SensorModeList.addItems(mode_list)

    def sensorModeList_callback(self):
        curItem = self.SensorListBox.currentRow()
        curMode = self.CallbackConnector.base.getCurMode(curItem)
        selMode = self.SensorModeList.currentText()
        if curMode != selMode:
            if selMode != '':
                self.CallbackConnector.base.setSampleMode(curItem, selMode)
                self.getpipelinestate()
                self.starttriggercheckbox.setEnabled(True)
                self.stoptriggercheckbox.setEnabled(True)

                sensorList = self.CallbackConnector.base.TrigBase.GetScannedSensorsFound()
                self.set_sensor_list_box(sensorList)
                self.SensorModeList.setCurrentText(selMode)
                self.SensorListBox.setCurrentRow(curItem)

    # -----------------------------------------------------------------------
    # ---- Labeled protocol runner (with plot)
    def _protocol_subject_session_arm(self) -> Tuple[str, str, str]:
        subject = self.subject_input.text().strip() or self.default_subject
        session = self.session_input.text().strip() or self.default_session
        arm = "right" if self.arm_right_radio.isChecked() else "left"
        return subject, session, arm

    def _build_standard_protocol_config(self, subject: str, session: str, arm: str) -> TrialConfig:
        return TrialConfig(
            gestures=["left_turn", "right_turn", "neutral", "signal_left", "signal_right", "horn"],
            gesture_duration=5.0,
            neutral_duration=5.0,
            repetitions=5,
            subject=subject.strip(),
            session=session.strip(),
            protocol_name="standard",
            arm=arm,
            prep_duration=5.0,
            inter_gesture_rest_s=1.0,
            label_trim_s=0.5,
            rest_label_trim_s=None,
            calibrate=True,
            calibration_neutral_s=5.0,
            calibration_mvc_s=5.0,
            calibration_mvc_prep_s=2.0,
            calibration_min_ratio=2.0,
        )

    def _build_neutral_recovery_protocol_config(self, subject: str, session: str, arm: str) -> TrialConfig:
        return TrialConfig(
            gestures=["left_turn", "right_turn"],
            gesture_duration=3.0,
            neutral_duration=5.0,
            repetitions=6,
            subject=subject.strip(),
            session=session.strip(),
            protocol_name="neutral_recovery",
            arm=arm,
            prep_duration=5.0,
            inter_gesture_rest_s=0.0,
            label_trim_s=0.5,
            rest_label_trim_s=None,
            recovery_neutral_lead_trim_s=1.0,
            recovery_neutral_trail_trim_s=0.0,
            calibrate=True,
            calibration_neutral_s=5.0,
            calibration_mvc_s=5.0,
            calibration_mvc_prep_s=2.0,
            calibration_min_ratio=2.0,
        )

    def _protocol_output_path(self, config: TrialConfig) -> Path:
        raw_dir = strict_raw_dir(STRICT_DATA_ROOT, config.arm, config.subject)
        raw_dir.mkdir(parents=True, exist_ok=True)
        if config.protocol_name == "standard":
            filename = f"{config.subject}_session{config.session}_raw.npz"
        else:
            filename = f"{config.subject}_session{config.session}_{config.protocol_name}_raw.npz"
        return raw_dir / filename

    def _total_instruction_steps(self, config: TrialConfig) -> int:
        if config.protocol_name == "neutral_recovery":
            return config.repetitions * len(config.gestures) * 2
        return config.repetitions * len(config.gestures)

    def protocol_callback(self):
        if not self.CallbackConnector or not hasattr(self, "CallbackConnector"):
            QMessageBox.critical(self, "Error", "CallbackConnector not ready. Connect and scan first.")
            return

        subject, session, arm = self._protocol_subject_session_arm()
        config = self._build_standard_protocol_config(subject, session, arm)
        output_path = self._protocol_output_path(config)

        try:
            self.run_protocol_with_plot(config, output_path)
            QMessageBox.information(self, "Saved", f"Saved to {output_path}")
            self.exportcsv_button.setEnabled(False)
            self.exportcsv_button.setStyleSheet("color : gray")
        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))

    def neutral_recovery_protocol_callback(self):
        if not self.CallbackConnector or not hasattr(self, "CallbackConnector"):
            QMessageBox.critical(self, "Error", "CallbackConnector not ready. Connect and scan first.")
            return

        subject, session, arm = self._protocol_subject_session_arm()
        config = self._build_neutral_recovery_protocol_config(subject, session, arm)
        output_path = self._protocol_output_path(config)

        try:
            self.run_protocol_with_plot(config, output_path)
            QMessageBox.information(self, "Saved", f"Saved to {output_path}")
            self.exportcsv_button.setEnabled(False)
            self.exportcsv_button.setStyleSheet("color : gray")
        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))

    def run_protocol_with_plot(self, config: TrialConfig, output_path: Path):
        # Use YT data (time,value) to keep timestamps
        self.CallbackConnector.streamYTData = True
        self.CallbackConnector.pauseFlag = False
        base = self.CallbackConnector.base
        base.start_trigger = False
        base.stop_trigger = False

        configured = base.ConfigureCollectionOutput()
        if not configured:
            raise RuntimeError("Failed to configure pipeline.")

        channel_count = base.channelcount
        plotter = self.plotCanvas if self.plot_enabled else None
        emg_idx = getattr(base, "emgChannelsIdx", [])
        self.protocol_abort = False
        self.protocol_running = True
        rest_trim_s = None
        if config.inter_gesture_rest_s > 0.0:
            rest_trim_s = resolve_rest_label_trim(
                config.label_trim_s,
                config.rest_label_trim_s,
                config.inter_gesture_rest_s,
            )

        base.TrigBase.Start(self.CallbackConnector.streamYTData)

        all_ts: List[List[float]] = []
        all_x: List[List[float]] = []
        all_labels: List[str] = []
        events = [{"event": "session_start", "t_wall": time.time()}]
        calib_neutral_ts: List[List[float]] = []
        calib_neutral_x: List[List[float]] = []
        calib_mvc_ts: List[List[float]] = []
        calib_mvc_x: List[List[float]] = []
        total_steps = self._total_instruction_steps(config)

        try:
            if config.prep_duration > 0:
                events.append({"event": "prep_start", "t_wall": time.time(), "duration_s": config.prep_duration})
                next_label = config.gestures[0] if config.gestures else None
                self.update_instruction("Get ready", next_label, total_steps, config.prep_duration)
                self.run_prep_buffer(config.prep_duration)

            if config.calibrate and not self.protocol_abort:
                events.append({"event": "calibration_neutral_start", "t_wall": time.time()})
                self.update_instruction(
                    f"Calibration: neutral rest",
                    "max contraction",
                    total_steps,
                    config.calibration_neutral_s,
                )
                seg_ts, seg_x, _ = self.collect_segment_with_plot(
                    self.CallbackConnector.DataHandler,
                    "calibration_neutral",
                    config.calibration_neutral_s,
                    channel_count,
                    plotter,
                    emg_idx,
                    stop_flag=self.protocol_abort_requested,
                )
                calib_neutral_ts = seg_ts
                calib_neutral_x = seg_x
                if self.protocol_abort:
                    events.append({"event": "session_abort", "t_wall": time.time()})
                    return

                # Brief prep countdown so the subject has time to brace
                if config.calibration_mvc_prep_s > 0 and not self.protocol_abort:
                    self.update_instruction(
                        "Prepare: SQUEEZE AS HARD AS POSSIBLE in...",
                        "max contraction",
                        total_steps,
                        config.calibration_mvc_prep_s,
                    )
                    self.run_prep_buffer(config.calibration_mvc_prep_s)

                events.append({"event": "calibration_mvc_start", "t_wall": time.time()})
                self.update_instruction(
                    "SQUEEZE AS HARD AS POSSIBLE — all arm/wrist muscles!",
                    config.gestures[0] if config.gestures else None,
                    total_steps,
                    config.calibration_mvc_s,
                )
                seg_ts, seg_x, _ = self.collect_segment_with_plot(
                    self.CallbackConnector.DataHandler,
                    "calibration_mvc",
                    config.calibration_mvc_s,
                    channel_count,
                    plotter,
                    emg_idx,
                    stop_flag=self.protocol_abort_requested,
                )
                calib_mvc_ts = seg_ts
                calib_mvc_x = seg_x

                # Quality gate: warn if MVC/neutral ratio is too low
                if calib_neutral_x and calib_mvc_x:
                    neutral_arr = np.asarray(calib_neutral_x, dtype=float)
                    mvc_arr = np.asarray(calib_mvc_x, dtype=float)
                    quality_neutral = neutral_arr
                    quality_mvc = mvc_arr
                    quality_scope = f"all {neutral_arr.shape[1]} channels"

                    # If both arms are connected in one stream, score only the selected arm.
                    # Sensor pairing contract is right arm channels first, then left arm channels.
                    if neutral_arr.shape[1] >= 30 and neutral_arr.shape[1] == mvc_arr.shape[1]:
                        split_idx = (neutral_arr.shape[1] + 1) // 2
                        if config.arm == "right":
                            quality_neutral = neutral_arr[:, :split_idx]
                            quality_mvc = mvc_arr[:, :split_idx]
                        else:
                            quality_neutral = neutral_arr[:, split_idx:]
                            quality_mvc = mvc_arr[:, split_idx:]
                        quality_scope = (
                            f"{config.arm} arm channels "
                            f"({quality_neutral.shape[1]}/{neutral_arr.shape[1]})"
                        )

                    neutral_rms = np.sqrt(np.mean(quality_neutral ** 2, axis=0))
                    mvc_rms = np.sqrt(np.mean(quality_mvc ** 2, axis=0))
                    ratio = np.where(neutral_rms < 1e-9, 1.0, mvc_rms / neutral_rms)
                    median_ratio = float(np.median(ratio))
                    n_weak = int(np.sum(ratio < config.calibration_min_ratio))
                    quality_msg = (
                        f"MVC quality ({quality_scope}): {median_ratio:.1f}x median "
                        f"({n_weak}/{len(ratio)} channels below {config.calibration_min_ratio:.0f}x)"
                    )
                    events.append({
                        "event": "calibration_quality",
                        "t_wall": time.time(),
                        "median_ratio": median_ratio,
                        "n_weak_channels": n_weak,
                        "scope": quality_scope,
                    })
                    if median_ratio < config.calibration_min_ratio:
                        self.update_instruction(
                            f"WARNING: Weak calibration ({median_ratio:.1f}x). Squeeze much harder next session.",
                            config.gestures[0] if config.gestures else None,
                            total_steps,
                            0,
                        )
                        QMessageBox.warning(
                            self,
                            "Weak MVC Calibration",
                            f"{quality_msg}\n\n"
                            "Your max contraction was too close to neutral rest.\n"
                            "This will degrade model accuracy.\n\n"
                            "For the next session:\n"
                            "  • Squeeze ALL arm and wrist muscles simultaneously\n"
                            "  • Grip tight as if squeezing something with full force\n"
                            "  • Sustain the effort for the full duration\n\n"
                            "Data will still be saved, but calibration may be unreliable.",
                        )
                    else:
                        self.update_instruction(
                            f"Calibration OK ({median_ratio:.1f}x). Starting protocol...",
                            config.gestures[0] if config.gestures else None,
                            total_steps,
                            0,
                        )
                if self.protocol_abort:
                    events.append({"event": "session_abort", "t_wall": time.time()})
                    return
                if config.inter_gesture_rest_s > 0.0:
                    rest_duration = config.inter_gesture_rest_s
                    self.update_instruction(
                        "Post-calibration rest: neutral buffer",
                        config.gestures[0] if config.gestures else None,
                        total_steps,
                        rest_duration,
                    )
                    events.append(
                        {
                            "event": "neutral_buffer_start",
                            "t_wall": time.time(),
                            "after": "calibration_mvc",
                        }
                    )
                    seg_ts, seg_x, seg_labels = self.collect_segment_with_plot(
                        self.CallbackConnector.DataHandler,
                        "neutral_buffer",
                        rest_duration,
                        channel_count,
                        plotter,
                        emg_idx,
                        label_trim_s=rest_trim_s or 0.0,
                        stop_flag=self.protocol_abort_requested,
                    )
                    all_ts.extend(seg_ts)
                    all_x.extend(seg_x)
                    all_labels.extend(seg_labels)
                    if self.protocol_abort:
                        events.append({"event": "session_abort", "t_wall": time.time()})
                        return

            if config.protocol_name == "neutral_recovery":
                for rep in range(config.repetitions):
                    for idx, gesture in enumerate(config.gestures):
                        if self.protocol_abort:
                            break
                        if idx + 1 < len(config.gestures):
                            next_gesture = config.gestures[idx + 1]
                        elif rep + 1 < config.repetitions:
                            next_gesture = config.gestures[0]
                        else:
                            next_gesture = None

                        segment_index = (rep * len(config.gestures) + idx) * 2
                        gesture_reps_left = max(0, total_steps - segment_index - 1)
                        recovery_reps_left = max(0, total_steps - segment_index - 2)

                        self.update_instruction(
                            f"Do: {gesture}",
                            "neutral",
                            gesture_reps_left,
                            config.gesture_duration,
                        )
                        events.append(
                            {
                                "event": f"{gesture}_start",
                                "t_wall": time.time(),
                                "rep": rep + 1,
                                "protocol": config.protocol_name,
                            }
                        )
                        seg_ts, seg_x, seg_labels = self.collect_segment_with_plot(
                            self.CallbackConnector.DataHandler,
                            gesture,
                            config.gesture_duration,
                            channel_count,
                            plotter,
                            emg_idx,
                            label_trim_s=config.label_trim_s,
                            stop_flag=self.protocol_abort_requested,
                        )
                        all_ts.extend(seg_ts)
                        all_x.extend(seg_x)
                        all_labels.extend(seg_labels)
                        if self.protocol_abort:
                            break

                        self.update_instruction(
                            "Recover: neutral",
                            next_gesture,
                            recovery_reps_left,
                            config.neutral_duration,
                        )
                        events.append(
                            {
                                "event": "neutral_recovery_start",
                                "t_wall": time.time(),
                                "rep": rep + 1,
                                "after": gesture,
                                "protocol": config.protocol_name,
                            }
                        )
                        seg_ts, seg_x, seg_labels = self.collect_segment_with_plot(
                            self.CallbackConnector.DataHandler,
                            "neutral",
                            config.neutral_duration,
                            channel_count,
                            plotter,
                            emg_idx,
                            label_trim_s=0.0,
                            leading_label_trim_s=config.recovery_neutral_lead_trim_s,
                            trailing_label_trim_s=config.recovery_neutral_trail_trim_s,
                            stop_flag=self.protocol_abort_requested,
                        )
                        all_ts.extend(seg_ts)
                        all_x.extend(seg_x)
                        all_labels.extend(seg_labels)
                        if self.protocol_abort:
                            break

                    if self.protocol_abort:
                        break
            else:
                for rep in range(config.repetitions):
                    for idx, gesture in enumerate(config.gestures):
                        if self.protocol_abort:
                            break
                        if idx + 1 < len(config.gestures):
                            next_gesture = config.gestures[idx + 1]
                        elif rep + 1 < config.repetitions:
                            next_gesture = config.gestures[0]
                        else:
                            next_gesture = None

                        duration = config.neutral_duration if gesture == "neutral" else config.gesture_duration
                        reps_left = (config.repetitions - rep - 1) * len(config.gestures) + (len(config.gestures) - idx - 1)
                        self.update_instruction(f"Do: {gesture}", next_gesture, reps_left, duration)
                        events.append({"event": f"{gesture}_start", "t_wall": time.time(), "rep": rep + 1})
                        seg_ts, seg_x, seg_labels = self.collect_segment_with_plot(
                            self.CallbackConnector.DataHandler,
                            gesture,
                            duration,
                            channel_count,
                            plotter,
                            emg_idx,
                            label_trim_s=config.label_trim_s,
                            stop_flag=self.protocol_abort_requested,
                        )
                        all_ts.extend(seg_ts)
                        all_x.extend(seg_x)
                        all_labels.extend(seg_labels)
                        if self.protocol_abort:
                            break

                        if config.inter_gesture_rest_s > 0.0 and (
                            idx + 1 < len(config.gestures) or rep + 1 < config.repetitions
                        ):
                            rest_duration = config.inter_gesture_rest_s
                            self.update_instruction(
                                "Rest: neutral buffer",
                                next_gesture,
                                reps_left,
                                rest_duration,
                            )
                            events.append(
                                {
                                    "event": "neutral_rest_start",
                                    "t_wall": time.time(),
                                    "rep": rep + 1,
                                    "after": gesture,
                                }
                            )
                            seg_ts, seg_x, seg_labels = self.collect_segment_with_plot(
                                self.CallbackConnector.DataHandler,
                                "neutral_buffer",
                                rest_duration,
                                channel_count,
                                plotter,
                                emg_idx,
                                label_trim_s=rest_trim_s or 0.0,
                                stop_flag=self.protocol_abort_requested,
                            )
                            all_ts.extend(seg_ts)
                            all_x.extend(seg_x)
                            all_labels.extend(seg_labels)
                            if self.protocol_abort:
                                break

            if self.protocol_abort:
                events.append({"event": "session_abort", "t_wall": time.time()})
        finally:
            base.Stop_Callback()
            events.append({"event": "session_stop", "t_wall": time.time()})
            self.protocol_running = False
            status_text = "Protocol complete." if not self.protocol_abort else "Protocol stopped early."
            self.update_instruction(status_text, None, 0, 0)

        X = np.asarray(all_x, dtype=float)
        timestamps = np.asarray(all_ts, dtype=float)
        y = np.asarray(all_labels, dtype=object)

        metadata = {
            "subject": config.subject,
            "arm":     config.arm,
            "session": config.session,
            "protocol_name": config.protocol_name,
            "data_root": str(STRICT_DATA_ROOT),
            "layout_mode": "strict",
            "gestures": config.gestures,
            "gesture_duration_s": config.gesture_duration,
            "neutral_duration_s": config.neutral_duration,
            "repetitions": config.repetitions,
            "channel_count": channel_count,
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "prep_duration_s": config.prep_duration,
            "inter_gesture_rest_s": config.inter_gesture_rest_s,
            "label_trim_s": config.label_trim_s,
            "rest_label_trim_s": rest_trim_s,
            "ramp_style": "ramp contractions (longer window for non-neutral gestures)",
            "emg_channel_labels": list(getattr(self.CallbackConnector.base, "emgChannelNames", [])),
        }
        if config.protocol_name == "neutral_recovery":
            metadata["neutral_recovery"] = {
                "lead_trim_s": float(config.recovery_neutral_lead_trim_s),
                "trail_trim_s": float(config.recovery_neutral_trail_trim_s),
                "sequence": [f"{gesture}->neutral" for gesture in config.gestures],
            }
        if config.calibrate:
            metadata["calibration"] = {
                "enabled": True,
                "neutral_duration_s": config.calibration_neutral_s,
                "mvc_duration_s": config.calibration_mvc_s,
            }

        if self.protocol_abort:
            if output_path.exists():
                try:
                    output_path.unlink()
                except Exception:
                    pass
            return

        output_path.parent.mkdir(parents=True, exist_ok=True)
        save_kwargs = {
            "X": X,
            "timestamps": timestamps,
            "y": y,
            "events": np.asarray(events, dtype=object),
            "metadata": metadata,
        }
        if config.calibrate:
            save_kwargs.update(
                {
                    "calib_neutral_X": np.asarray(calib_neutral_x, dtype=float),
                    "calib_neutral_timestamps": np.asarray(calib_neutral_ts, dtype=float),
                    "calib_mvc_X": np.asarray(calib_mvc_x, dtype=float),
                    "calib_mvc_timestamps": np.asarray(calib_mvc_ts, dtype=float),
                }
            )
        np.savez_compressed(output_path, **save_kwargs)
        print(f"Saved {X.shape[0]} samples to {output_path}")

    def collect_segment_with_plot(
        self,
        kernel,
        label: str,
        duration_s: float,
        channel_count: int,
        plotter=None,
        emg_idx: Optional[List[int]] = None,
        label_trim_s: float = 0.0,
        leading_label_trim_s: Optional[float] = None,
        trailing_label_trim_s: Optional[float] = None,
        stop_flag=None,
    ) -> Tuple[List[List[float]], List[List[float]], List[Optional[str]]]:
        ts_buffer: List[List[float]] = []
        x_buffer: List[List[float]] = []
        labels: List[Optional[str]] = []
        end_time = time.time() + duration_s
        segment_start_ts: Optional[float] = None
        lead_trim_s = label_trim_s if leading_label_trim_s is None else max(0.0, float(leading_label_trim_s))
        trail_trim_s = label_trim_s if trailing_label_trim_s is None else max(0.0, float(trailing_label_trim_s))
        apply_trim = (lead_trim_s > 0.0 or trail_trim_s > 0.0) and duration_s > (lead_trim_s + trail_trim_s)

        while time.time() < end_time:
            if stop_flag and stop_flag():
                break
            remaining = max(0, end_time - time.time())
            self.update_timer(remaining)
            out = kernel.GetYTData()
            if out is None:
                time.sleep(0.001)
                continue

            channel_times = []
            channel_values = []
            for channel in out:
                if not channel:
                    continue
                chan_array = np.asarray(channel[0], dtype=object)
                if chan_array.size == 0:
                    continue
                times, values = zip(*[self._pair_time_value(s) for s in chan_array])
                channel_times.append(list(times))
                channel_values.append(list(values))

            if len(channel_values) < channel_count:
                continue

            sample_count = min(len(c) for c in channel_values)
            if sample_count == 0:
                continue

            for idx in range(sample_count):
                ts_buffer.append([channel_times[ch][idx] for ch in range(channel_count)])
                x_buffer.append([channel_values[ch][idx] for ch in range(channel_count)])
                sample_label = label
                if apply_trim:
                    if segment_start_ts is None:
                        segment_start_ts = channel_times[0][idx]
                    elapsed = channel_times[0][idx] - segment_start_ts
                    if elapsed < lead_trim_s or elapsed > duration_s - trail_trim_s:
                        sample_label = None
                labels.append(sample_label)

            if plotter and emg_idx:
                try:
                    emg_data = [channel_values[ch] for ch in emg_idx]
                    next_vals = [vals[-1] for vals in emg_data]
                    plotter.plot_new_data(emg_data, next_vals)
                except Exception as e:
                    print(f"Plot update error: {e}")

            QApplication.processEvents()

        return ts_buffer, x_buffer, labels

    # --- UI helpers for protocol guidance
    def protocol_abort_requested(self) -> bool:
        return self.protocol_abort

    def run_prep_buffer(self, duration: float):
        """Small countdown before starting gesture collection."""
        end_time = time.time() + duration
        while time.time() < end_time and not self.protocol_abort:
            remaining = max(0, end_time - time.time())
            self.update_timer(remaining)
            QApplication.processEvents()
            time.sleep(0.05)

    def update_instruction(self, text: str, next_gesture: Optional[str], reps_left: int, duration: float):
        self.instruction_label.setText(text)
        if next_gesture:
            self.next_label.setText(f"Next: {next_gesture}")
        else:
            self.next_label.setText("Next: --")
        self.reps_label.setText(f"Reps remaining: {reps_left}")
        self.timer_label.setText(f"Timer: {duration:.1f}s")
        QApplication.processEvents()

    def update_timer(self, remaining: float):
        self.timer_label.setText(f"Timer: {remaining:.1f}s")

    def _pair_time_value(self, sample) -> Tuple[float, float]:
        if hasattr(sample, "Item1"):
            return float(sample.Item1), float(sample.Item2)
        return float(sample[0]), float(sample[1])
