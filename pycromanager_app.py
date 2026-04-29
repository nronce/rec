import os
import sys
import time
import json
import traceback
import datetime
from pathlib import Path
import numpy as np
import tifffile

from qtpy.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
    QPushButton, QLabel, QGroupBox, QSpinBox, QDoubleSpinBox, 
    QComboBox, QLineEdit, QFileDialog, QStackedWidget, QMessageBox,
    QPlainTextEdit, QCheckBox, QSlider, QSizePolicy
)
from qtpy.QtCore import Qt, QObject, Signal, QThread
from qtpy.QtGui import QPixmap, QImage
from pypylon import pylon

# Pycromanager imports
from pycromanager import Core, Acquisition, multi_d_acquisition_events

# Import hardware classes from your existing scripts
import config
from hardware import Microscope, ThorlabsLaser, BaslerCamera, ThorlabsLED

def apply_dark_theme(app):
    app.setStyle("Fusion")
    qss = """
    QMainWindow, QWidget { background-color: #121212; color: #ecf0f1; font-family: "Segoe UI", Arial, sans-serif; }
    QGroupBox { border: 1px solid #333333; border-radius: 6px; margin-top: 1ex; padding-top: 15px; font-weight: bold; color: #4fc3f7; }
    QGroupBox::title { subcontrol-origin: margin; subcontrol-position: top left; left: 10px; }
    QPushButton { background-color: #2c3e50; border-radius: 4px; padding: 8px; font-weight: bold; }
    QPushButton:hover { background-color: #34495e; }
    QPushButton:pressed { background-color: #1a252f; }
    QPushButton:disabled { background-color: #7f8c8d; color: #bdc3c7; }
    QSpinBox, QDoubleSpinBox, QLineEdit, QComboBox { background-color: #1e1e1e; color: #ffffff; border: 1px solid #444; border-radius: 3px; padding: 4px; }
    QPlainTextEdit { background-color: #000000; color: #00ff00; font-family: Consolas, monospace; font-size: 11px; }
    """
    app.setStyleSheet(qss)


# --- THREAD-SAFE MANAGERS ---
class AppSignals(QObject):
    log_signal = Signal(str)

class BaslerRecordingThread(QThread):
    """
    Thread for recording Basler frames during acquisition.
    Uses DAQmx Continuous Buffered Analog Input to guarantee zero missed pulses.
    """
    finished_signal = Signal()
    error_signal = Signal(str)
    
    def __init__(self, basler_camera, expected_frames=None):
        super().__init__()
        self.basler_camera = basler_camera
        self.expected_frames = expected_frames
        self.is_running = False
        self.graceful_stop = False 
        
    def run(self):
        self.is_running = True
        try:
            import nidaqmx
            from nidaqmx.constants import AcquisitionType
            from pypylon import pylon
            import numpy as np
            
            with nidaqmx.Task() as daq_task:
                daq_task.ai_channels.add_ai_voltage_chan("Dev1/ai0", min_val=0.0, max_val=5.0)
                
                # Tell DAQ hardware to sample perfectly at 10kHz in the background
                daq_task.timing.cfg_samp_clk_timing(rate=10000.0, sample_mode=AcquisitionType.CONTINUOUS)
                
                was_high = False
                pulse_count = 0
                frames_captured = 0
                
                # Start DAQ buffer BEFORE the loop
                daq_task.start()
                
                while self.is_running and self.basler_camera.is_recording:
                    # Blocks for 10ms, fetching 100 chronological voltage samples from the hardware buffer
                    try:
                        chunk = daq_task.read(number_of_samples_per_channel=100)
                    except Exception as e:
                        self.error_signal.emit(f"Buffer read error: {e}")
                        continue
                        
                    trigger_now = False
                    
                    # Process every microsecond chronologically so no edge is ever missed
                    for voltage in chunk:
                        is_high = voltage > self.basler_camera.trigger_threshold
                        if is_high and not was_high:
                            pulse_count += 1
                            if pulse_count >= self.basler_camera.n_divider:
                                trigger_now = True
                                pulse_count = 0
                        was_high = is_high
                        
                        # Fire exactly when the chronological edge is hit
                        if trigger_now:
                            if self.basler_camera.camera.WaitForFrameTriggerReady(200):
                                self.basler_camera.camera.ExecuteSoftwareTrigger()
                                res = self.basler_camera.camera.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)
                                
                                if res.GrabSucceeded():
                                    raw_img_safe = np.ascontiguousarray(res.Array.copy().T)
                                    if self.basler_camera.tiff_writer:
                                        self.basler_camera.tiff_writer.write(raw_img_safe)
                                        self.basler_camera.frames_written += 1
                                        frames_captured += 1
                                res.Release()
                            trigger_now = False # Reset so we don't double fire
                            
                    # End thread naturally if PycroManager finished AND we secured all our expected frames
                    if self.graceful_stop and (self.expected_frames is None or frames_captured >= self.expected_frames):
                        self.is_running = False
                        break
                        
        except Exception as e:
            self.error_signal.emit(f"DAQ or Grabbing Error: {e}")
        finally:
            self.finished_signal.emit()
            
    def initiate_graceful_stop(self):
        """Tells the thread to finish writing its final expected frames, then exit."""
        self.graceful_stop = True

    def stop(self):
        self.is_running = False
        self.wait()

class AcquisitionWorker(QThread):
    """
    Runs the blocking Pycro-Manager Acquisition in a background thread 
    so the PyQt GUI remains fully responsive.
    """
    finished_signal = Signal(str, str, str) # save_dir, save_name, mode
    error_signal = Signal(str)
    log_signal = Signal(str)

    def __init__(self, save_dir, save_name, events, hook_fn, mode):
        super().__init__()
        self.save_dir = save_dir
        self.save_name = save_name
        self.events = events
        self.hook_fn = hook_fn
        self.mode = mode

    def run(self):
        try:
            acq = Acquisition(directory=self.save_dir, name=self.save_name, 
                              pre_hardware_hook_fn=self.hook_fn)
            self.log_signal.emit("Constructor accepted. Acquiring sequence...")
            
            with acq:
                acq.acquire(self.events)
                
            # Signal the main thread that we are completely done
            self.finished_signal.emit(self.save_dir, self.save_name, self.mode)
        except Exception as e:
            self.error_signal.emit(traceback.format_exc())

class CameraThread(QThread):
    change_pixmap_signal = Signal(np.ndarray)
    fps_signal = Signal(float)

    def __init__(self, camera):
        super().__init__()
        self.camera = camera
        self._run_flag = True
        self.n_divider = 1
        
        self.is_recording = False
        self.tiff_writer = None
        
        # Initialize camera to 12-bit
        self.camera.Open()
        try:
            self.camera.PixelFormat.SetValue("Mono12")
        except:
            self.camera.PixelFormat.SetValue("Mono16")

    def run(self):
        task = None
        try:
            import nidaqmx
            task = nidaqmx.Task()
            task.ai_channels.add_ai_voltage_chan("Dev1/ai0")
        except Exception as e:
            print(f"DAQ Init Error: {e}")

        self.camera.TriggerSelector.SetValue("FrameStart")
        self.camera.TriggerMode.SetValue("On")
        self.camera.TriggerSource.SetValue("Software")
        self.camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)

        was_high = False
        pulse_count = 0
        last_time = time.time()
        frame_count = 0

        while self._run_flag:
            try:
                # 1. Pulse Detection
                trigger_now = False
                if task:
                    voltage = task.read()
                    is_high = voltage > 2.0
                    
                    if is_high and not was_high:
                        pulse_count += 1
                        if pulse_count >= self.n_divider:
                            trigger_now = True
                            pulse_count = 0
                    was_high = is_high
                else:
                    # Fallback to continuous if DAQ is disconnected for testing
                    trigger_now = True
                    time.sleep(0.01)

                # 2. Capture
                if trigger_now:
                    if self.camera.WaitForFrameTriggerReady(200):
                        self.camera.ExecuteSoftwareTrigger()
                        res = self.camera.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)
                        if res.GrabSucceeded():
                            raw_img = res.Array # 12-bit RAW data
                            
                            # FIX: Ensure memory is aligned for the writer
                            raw_img_safe = np.ascontiguousarray(raw_img)
                            
                            # Append to TIFF Stack
                            if self.is_recording and self.tiff_writer is not None:
                                self.tiff_writer.write(raw_img_safe)

                            # 4. Emit for UI
                            self.change_pixmap_signal.emit(raw_img.copy())
                            
                            # RESTORED: Use the local frame_count for the FPS display
                            frame_count += 1
                            if time.time() - last_time >= 1.0:
                                self.fps_signal.emit(frame_count / (time.time() - last_time))
                                frame_count = 0
                                last_time = time.time()
                                
                        res.Release()
            except Exception as e:
                print(f"Loop Error: {e}")

        # Cleanup
        if self.tiff_writer:
            self.tiff_writer.close()
        if task:
            task.close()
        self.camera.StopGrabbing()

    def start_recording(self, filepath):
        try:
            self.tiff_writer = tifffile.TiffWriter(filepath, bigtiff=True)
            self.is_recording = True
            print(f"Recording started: {filepath}")
        except Exception as e:
            print(f"Failed to start recording: {e}")

    def stop_recording(self):
        self.is_recording = False
        if self.tiff_writer:
            self.tiff_writer.close()
            self.tiff_writer = None
            print("Recording saved and closed.")

    def stop(self):
        self._run_flag = False
        self.wait()


class BaslerPreviewWindow(QMainWindow):
    def __init__(self, parent=None, camera=None):
        super().__init__(parent)
        self.camera = camera
        self.auto_contrast = False
        self.vmin, self.vmax = 0, 4095
        self.save_folder = ""
        
        self.init_ui()
        self.thread = CameraThread(self.camera)
        self.thread.change_pixmap_signal.connect(self.process_display)
        self.thread.fps_signal.connect(lambda f: self.fps_label.setText(f"FPS: {f:.1f}"))
        
        # Read initial exposure from camera to update UI
        try:
            initial_exp = self.camera.ExposureTime.GetValue() / 1000.0
            self.exp_input.setValue(initial_exp)
        except: pass

        self.thread.start()

    def init_ui(self):
        self.setWindowTitle("Basler Camera Preview")
        # Ensure the whole window opens at a good size
        self.resize(1024, 800) 
        
        main = QWidget()
        self.setCentralWidget(main)
        layout = QVBoxLayout(main)

        # Image Display - ensuring it has a generous minimum size
        self.img_label = QLabel("Waiting for Trigger...")
        self.img_label.setAlignment(Qt.AlignCenter)
        self.img_label.setStyleSheet("background: black; color: white; font-size: 16px;")
        self.img_label.setMinimumSize(800, 600)
        self.img_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        layout.addWidget(self.img_label, 1) # '1' makes it take up remaining stretch space

        # Top Bar (N and Exposure)
        top = QHBoxLayout()
        top.addWidget(QLabel("Capture 1 in N frames:"))
        self.n_spin = QSpinBox()
        self.n_spin.setRange(1, 1000)
        self.n_spin.valueChanged.connect(self.update_n)
        top.addWidget(self.n_spin)

        top.addSpacing(30)

        top.addWidget(QLabel("Exposure Time:"))
        self.exp_input = QDoubleSpinBox()
        self.exp_input.setRange(0.01, 2000.0) # Up to 2 seconds
        self.exp_input.setDecimals(2)
        self.exp_input.setSuffix(" ms")
        self.exp_input.setKeyboardTracking(False) # Only updates when Enter is pressed or focus lost
        self.exp_input.valueChanged.connect(self.update_exposure)
        top.addWidget(self.exp_input)
        
        top.addStretch()
        layout.addLayout(top)

        # Contrast
        contrast = QHBoxLayout()
        self.auto_cb = QCheckBox("Auto-Contrast (Display Only)")
        self.auto_cb.toggled.connect(self.toggle_auto)
        contrast.addWidget(self.auto_cb)
        
        contrast.addWidget(QLabel("Min (Black):"))
        self.min_s = QSlider(Qt.Horizontal)
        self.min_s.setRange(0, 4000)
        self.min_s.valueChanged.connect(self.update_contrast)
        contrast.addWidget(self.min_s)

        contrast.addWidget(QLabel("Max (White):"))
        self.max_s = QSlider(Qt.Horizontal)
        self.max_s.setRange(100, 4095)
        self.max_s.setValue(4095)
        self.max_s.valueChanged.connect(self.update_contrast)
        contrast.addWidget(self.max_s)
        layout.addLayout(contrast)

        # Bottom Bar (Folder and Recording)
        bot = QHBoxLayout()
        self.btn_folder = QPushButton("📁 Set Folder")
        self.btn_folder.clicked.connect(self.set_folder)
        bot.addWidget(self.btn_folder)
        
        self.folder_label = QLabel("No folder selected")
        self.folder_label.setStyleSheet("color: #555; font-style: italic;")
        bot.addWidget(self.folder_label)

        bot.addSpacing(20)

        self.btn_record = QPushButton("⏺ START TIFF STACK")
        self.btn_record.setCheckable(True)
        self.btn_record.clicked.connect(self.handle_record)
        # Make the record button a bit thicker/obvious
        self.btn_record.setMinimumHeight(40)
        bot.addWidget(self.btn_record)

        self.fps_label = QLabel("FPS: 0.0")
        self.fps_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.fps_label.setMinimumWidth(80)
        bot.addWidget(self.fps_label)
        
        layout.addLayout(bot)

    def update_n(self): 
        self.thread.n_divider = self.n_spin.value()
        
    def toggle_auto(self, checked): 
        self.auto_contrast = checked
        self.min_s.setEnabled(not checked)
        self.max_s.setEnabled(not checked)
        
    def update_contrast(self): 
        self.vmin, self.vmax = self.min_s.value(), self.max_s.value()
    
    def update_exposure(self):
        ms = self.exp_input.value()
        try: 
            # Basler expects microseconds
            self.camera.ExposureTime.SetValue(ms * 1000.0)
        except Exception as e: 
            print(f"Exposure set failed: {e}")

    def set_folder(self):
        path = QFileDialog.getExistingDirectory(self, "Select Folder for TIFF Stacks")
        if path:
            self.save_folder = path
            self.folder_label.setText(f".../{os.path.basename(path)}")

    def handle_record(self):
        if self.btn_record.isChecked():
            if not self.save_folder:
                self.set_folder()
                if not self.save_folder:  # If user canceled folder selection
                    self.btn_record.setChecked(False)
                    return
            
            # Auto-generate a timestamped filename
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = os.path.join(self.save_folder, f"Basler_Stack_{timestamp}.tif")
            
            self.thread.start_recording(filepath)
            self.btn_record.setText("⏹ STOP RECORDING")
            self.btn_record.setStyleSheet("background: #d32f2f; color: white; font-weight: bold;")
        else:
            self.thread.stop_recording()
            self.btn_record.setText("⏺ START TIFF STACK")
            self.btn_record.setStyleSheet("")

    def process_display(self, raw_img):
        # Transpose to match Kuro orientation
        raw_img = raw_img.T

        if self.auto_contrast:
            c_min, c_max = np.min(raw_img), np.max(raw_img)
        else:
            c_min, c_max = self.vmin, self.vmax

        span = max(1, c_max - c_min)
        display_img = np.clip((raw_img.astype(np.float32) - c_min) / span * 255, 0, 255).astype(np.uint8)

        # 1. Ensure the array is laid out contiguously in memory after the math/transpose
        display_img = np.ascontiguousarray(display_img)
        
        h, w = display_img.shape
        
        # 2. FIX: Use .tobytes() instead of .data to satisfy PyQt's type requirements
        qimg = QImage(display_img.tobytes(), w, h, w, QImage.Format_Grayscale8)
        
        self.img_label.setPixmap(QPixmap.fromImage(qimg).scaled(
            self.img_label.width(), self.img_label.height(), Qt.KeepAspectRatio))

    def closeEvent(self, event):
        self.thread.stop()
        # Notify parent that preview window closed
        if self.parent() and hasattr(self.parent(), '_on_preview_window_closed'):
            self.parent()._on_preview_window_closed()
        event.accept()


class PycromanagerAcquisitionApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Pycromanager Acquisition Control")
        self.resize(650, 900)
        
        # Thread-safe logging connection
        self.signals = AppSignals()
        self.signals.log_signal.connect(self._append_log)
        
        self.hw = Microscope()
        self.laser = None
        self.basler_camera = None  # Will be initialized on connect
        self.thorlabs_led = None   # Thorlabs 780nm LED control
        
        self._uv_state = False
        self._wl_state = False
        self._hardware_sequence = {}
        
        # Live Power & Thread Tracking
        self._is_acquiring = False
        self._acq_start_time = 0
        self._uv_power_history = []
        self._live_uv_power = config.DEFAULTS.get("laser_ma", 25.0)
        self._acq_worker = None
        
        # Basler Preview Window
        self.preview_window = None
        self._preview_active = False
        
        self.setup_ui()

    def setup_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)

        # --- 1. System & Hardware ---
        box_sys = QGroupBox("1. Hardware Connection")
        l_sys = QVBoxLayout()
        self.btn_connect = QPushButton("Connect Microscope & Laser")
        self.btn_connect.setStyleSheet("background-color: #2980b9;")
        self.btn_connect.clicked.connect(self.connect_hardware)
        l_sys.addWidget(self.btn_connect)
        self.lbl_hw_status = QLabel("Status: Disconnected")
        self.lbl_hw_status.setStyleSheet("color: #f39c12; font-weight: bold;")
        l_sys.addWidget(self.lbl_hw_status)
        
        # Basler Preview Button
        self.btn_basler_preview = QPushButton("Open Basler Preview")
        self.btn_basler_preview.setStyleSheet("background-color: #3498db;")
        self.btn_basler_preview.setCheckable(True)
        self.btn_basler_preview.toggled.connect(self._toggle_basler_preview)
        self.btn_basler_preview.setEnabled(False)
        l_sys.addWidget(self.btn_basler_preview)
        
        box_sys.setLayout(l_sys)
        layout.addWidget(box_sys)

        # --- 2. Illumination, Exposures & Filters ---
        box_illum = QGroupBox("2. Hardware Configuration")
        l_illum = QVBoxLayout()
        
        h_uv = QHBoxLayout()
        h_uv.addWidget(QLabel("UV Laser Power (mA):"))
        self.spin_laser_ma = QDoubleSpinBox()
        self.spin_laser_ma.setRange(0, 200)
        self.spin_laser_ma.setValue(self._live_uv_power)
        self.spin_laser_ma.valueChanged.connect(self._live_update_laser)
        h_uv.addWidget(self.spin_laser_ma)
        
        self.btn_uv_toggle = QPushButton("UV Laser: OFF")
        self.btn_uv_toggle.setStyleSheet("background-color: black; color: white; font-weight: bold;")
        self.btn_uv_toggle.setCheckable(True)
        self.btn_uv_toggle.toggled.connect(self.toggle_uv)
        h_uv.addWidget(self.btn_uv_toggle)
        l_illum.addLayout(h_uv)
        
        h_wl = QHBoxLayout()
        h_wl.addWidget(QLabel("Thorlabs 780nm LED Status:"))
        
        self.btn_wl_toggle = QPushButton("Thorlabs LED: OFF")
        self.btn_wl_toggle.setStyleSheet("background-color: black; color: white; font-weight: bold;")
        self.btn_wl_toggle.setCheckable(True)
        self.btn_wl_toggle.toggled.connect(self.toggle_wl)
        h_wl.addWidget(self.btn_wl_toggle)
        l_illum.addLayout(h_wl)

        h_exp = QHBoxLayout()
        h_exp.addWidget(QLabel("Main / SMLM Exp (ms):"))
        self.spin_smlm_exp = QDoubleSpinBox()
        self.spin_smlm_exp.setRange(0.1, 10000.0)
        self.spin_smlm_exp.setValue(config.DEFAULTS.get("smlm_exposure_ms", 30.0))
        self.spin_smlm_exp.valueChanged.connect(self._live_update_exposure)
        h_exp.addWidget(self.spin_smlm_exp)
        
        h_exp.addWidget(QLabel("WL Exp (ms):"))
        self.spin_wl_exp = QDoubleSpinBox()
        self.spin_wl_exp.setRange(0.1, 10000.0)
        self.spin_wl_exp.setValue(config.DEFAULTS.get("wl_exposure_ms", 50.0))
        h_exp.addWidget(self.spin_wl_exp)
        l_illum.addLayout(h_exp)

        h_filter = QHBoxLayout()
        h_filter.addWidget(QLabel("Microscope Filter Block:"))
        self.combo_filter_block = QComboBox()
        self.combo_filter_block.addItem("Empty", 1)  
        self.combo_filter_block.addItem("568 LP", 2)  
        self.combo_filter_block.addItem("635 LP", 3)  
        self.combo_filter_block.setCurrentIndex(1)
        self.combo_filter_block.activated.connect(self._live_update_filter)
        h_filter.addWidget(self.combo_filter_block)
        l_illum.addLayout(h_filter)
        
        box_illum.setLayout(l_illum)
        layout.addWidget(box_illum)

        # --- 3. Mode Selection & Parameters ---
        box_acq = QGroupBox("3. Acquisition Parameters")
        l_acq = QVBoxLayout()
        
        h_mode = QHBoxLayout()
        h_mode.addWidget(QLabel("Acquisition Mode:"))
        self.combo_mode = QComboBox()
        self.combo_mode.addItems(["Normal", "Interleaved PALM", "Z-Stack", "Dual Imaging"])
        self.combo_mode.currentTextChanged.connect(self._on_mode_changed)
        h_mode.addWidget(self.combo_mode)
        l_acq.addLayout(h_mode)

        self.stack_params = QStackedWidget()
        
        w_normal = QWidget()
        l_normal = QVBoxLayout(w_normal)
        l_normal.setContentsMargins(0, 0, 0, 0)
        h_norm1 = QHBoxLayout()
        h_norm1.addWidget(QLabel("Total Frames:"))
        self.spin_normal_frames = QSpinBox()
        self.spin_normal_frames.setRange(1, 200000)
        self.spin_normal_frames.setValue(config.DEFAULTS.get("target_frames_normal", 1000))
        h_norm1.addWidget(self.spin_normal_frames)
        l_normal.addLayout(h_norm1)
        self.stack_params.addWidget(w_normal)

        w_palm = QWidget()
        l_palm = QVBoxLayout(w_palm)
        l_palm.setContentsMargins(0, 0, 0, 0)
        h_palm1 = QHBoxLayout()
        h_palm1.addWidget(QLabel("Num Cycles:"))
        self.spin_palm_cycles = QSpinBox()
        self.spin_palm_cycles.setValue(10)
        h_palm1.addWidget(self.spin_palm_cycles)
        
        h_palm1.addWidget(QLabel("SMLM Frames:"))
        self.spin_palm_smlm = QSpinBox()
        self.spin_palm_smlm.setRange(1, 100000)
        self.spin_palm_smlm.setValue(1000)
        h_palm1.addWidget(self.spin_palm_smlm)
        l_palm.addLayout(h_palm1)
        
        h_palm2 = QHBoxLayout()
        h_palm2.addWidget(QLabel("PA Pulse (ms):"))
        self.spin_palm_pa = QDoubleSpinBox()
        self.spin_palm_pa.setRange(0, 10000)
        self.spin_palm_pa.setValue(100.0)
        h_palm2.addWidget(self.spin_palm_pa)
        
        h_palm2.addWidget(QLabel("WL Frames:"))
        self.spin_palm_wl = QSpinBox()
        self.spin_palm_wl.setRange(1, 1000)
        self.spin_palm_wl.setValue(10)
        h_palm2.addWidget(self.spin_palm_wl)
        l_palm.addLayout(h_palm2)
        self.stack_params.addWidget(w_palm)
        
        w_zstack = QWidget()
        l_zstack = QVBoxLayout(w_zstack)
        l_zstack.setContentsMargins(0, 0, 0, 0)
        h_z1 = QHBoxLayout()
        h_z1.addWidget(QLabel("Number of Planes:"))
        self.spin_z_steps = QSpinBox()
        self.spin_z_steps.setRange(1, 1000)
        self.spin_z_steps.setValue(17) 
        h_z1.addWidget(self.spin_z_steps)
        h_z1.addWidget(QLabel("Step Size (µm):"))
        self.spin_z_step_um = QDoubleSpinBox()
        self.spin_z_step_um.setRange(0.01, 100.0)
        self.spin_z_step_um.setSingleStep(0.1)
        self.spin_z_step_um.setValue(0.5) 
        h_z1.addWidget(self.spin_z_step_um)
        l_zstack.addLayout(h_z1)
        self.stack_params.addWidget(w_zstack)
        
        w_dual = QWidget()
        l_dual = QVBoxLayout(w_dual)
        l_dual.setContentsMargins(0, 0, 0, 0)
        h_dual1 = QHBoxLayout()
        h_dual1.addWidget(QLabel("Total Frames:"))
        self.spin_dual_frames = QSpinBox()
        self.spin_dual_frames.setRange(1, 200000)
        self.spin_dual_frames.setValue(1000)
        h_dual1.addWidget(self.spin_dual_frames)
        
        l_dual.addLayout(h_dual1)
        self.stack_params.addWidget(w_dual)
        
        l_acq.addWidget(self.stack_params)
        
        h_basler_acq = QHBoxLayout()
        self.chk_basler_record_acq = QCheckBox("Record Basler during acquisition")
        self.chk_basler_record_acq.setChecked(False)
        self.chk_basler_record_acq.setToolTip("Save Basler camera frames to the same directory as SMLM data")
        h_basler_acq.addWidget(self.chk_basler_record_acq)
        
        h_basler_acq.addWidget(QLabel("1 in N frames:"))
        self.spin_basler_n = QSpinBox()
        self.spin_basler_n.setRange(1, 1000)
        self.spin_basler_n.setValue(10)  # Default to every 10th frame
        self.spin_basler_n.setToolTip("Capture one Basler frame for every N main acquisition frames")
        h_basler_acq.addWidget(self.spin_basler_n)
        
        h_basler_acq.addStretch()
        l_acq.addLayout(h_basler_acq)
        
        box_acq.setLayout(l_acq)
        layout.addWidget(box_acq)

        # --- 4. Output & Actions ---
        box_out = QGroupBox("4. Saving Destination")
        l_out = QVBoxLayout()
        
        h_dir = QHBoxLayout()
        h_dir.addWidget(QLabel("Directory:"))
        self.txt_dir = QLineEdit(str(config.DEFAULT_SAVE_DIR))
        h_dir.addWidget(self.txt_dir)
        self.btn_dir = QPushButton("Browse")
        self.btn_dir.clicked.connect(self.browse_dir)
        h_dir.addWidget(self.btn_dir)
        l_out.addLayout(h_dir)

        h_name = QHBoxLayout()
        h_name.addWidget(QLabel("Base Name:"))
        self.txt_name = QLineEdit("experiment_01")
        h_name.addWidget(self.txt_name)
        l_out.addLayout(h_name)
        
        lbl_auto = QLabel("<i>Note: Date and Mode will be auto-prepended to the folder.</i>")
        lbl_auto.setStyleSheet("color: #7f8c8d; font-size: 11px;")
        l_out.addWidget(lbl_auto)
        
        box_out.setLayout(l_out)
        layout.addWidget(box_out)

        self.btn_run = QPushButton("START PYCROMANAGER ACQUISITION")
        self.btn_run.setStyleSheet("background-color: #27ae60; font-size: 14px; padding: 15px;")
        self.btn_run.clicked.connect(self.run_acquisition)
        layout.addWidget(self.btn_run)

        # --- Debug Console ---
        self.console = QPlainTextEdit()
        self.console.setReadOnly(True)
        self.console.setMaximumHeight(200)
        layout.addWidget(self.console)

    def log(self, text):
        self.signals.log_signal.emit(text)

    def _append_log(self, text):
        timestamp = time.strftime("%H:%M:%S")
        self.console.appendPlainText(f"[{timestamp}] {text}")
        self.console.verticalScrollBar().setValue(self.console.verticalScrollBar().maximum())

    def _on_mode_changed(self, text):
        if text == "Normal": self.stack_params.setCurrentIndex(0)
        elif text == "Interleaved PALM": self.stack_params.setCurrentIndex(1)
        elif text == "Z-Stack": self.stack_params.setCurrentIndex(2)
        elif text == "Dual Imaging": self.stack_params.setCurrentIndex(3)

    def browse_dir(self):
        d = QFileDialog.getExistingDirectory(self, "Select Save Directory", self.txt_dir.text())
        if d: self.txt_dir.setText(d)

    def connect_hardware(self):
        self.log("Attempting to connect to hardware...")
        try:
            self.hw.connect()
            if ThorlabsLaser is not None:
                self.laser = ThorlabsLaser()
                self.laser.connect()
            
            # Initialize Basler camera
            try:
                self.basler_camera = BaslerCamera()
                if self.basler_camera.connect():
                    self.btn_basler_preview.setEnabled(True)
                    self.log("Basler camera connected.")
                else:
                    self.basler_camera = None
                    self.log("Basler camera connection failed.")
            except Exception as e:
                self.basler_camera = None
                self.log(f"Basler camera unavailable: {e}")
            
            # Initialize Thorlabs 780nm LED
            try:
                self.thorlabs_led = ThorlabsLED()
                if self.thorlabs_led.connect():
                    self.log("Thorlabs 780nm LED connected via DAQ AO1.")
                else:
                    self.thorlabs_led = None
                    self.log("Warning: Thorlabs LED connection failed.")
            except Exception as e:
                self.thorlabs_led = None
                self.log(f"Thorlabs LED unavailable: {e}")
                
            self.lbl_hw_status.setText("Status: Connected")
            self.lbl_hw_status.setStyleSheet("color: #2ecc71; font-weight: bold;")
            self.log("Hardware connected successfully.")
            
            self._live_update_filter()
            self._live_update_laser()
            self._live_update_exposure()
        except Exception as e:
            self.lbl_hw_status.setText(f"Status: Error - {e}")
            self.lbl_hw_status.setStyleSheet("color: #e74c3c; font-weight: bold;")
            self.log(f"Connection Error: {str(e)}")

    def toggle_uv(self, checked):
        if not self.laser or not getattr(self.laser, 'is_connected', False):
            self.btn_uv_toggle.setChecked(False)
            self.log("Cannot toggle UV: Laser not connected.")
            return
            
        if checked:
            self.btn_uv_toggle.setText("UV Laser: ON")
            self.btn_uv_toggle.setStyleSheet("background-color: #9b59b6; color: white; font-weight: bold;") 
            self.laser.set_power_ma(self._live_uv_power)
            self.laser.set_emission(True)
            self.log("UV Laser toggled ON")
        else:
            self.btn_uv_toggle.setText("UV Laser: OFF")
            self.btn_uv_toggle.setStyleSheet("background-color: black; color: white; font-weight: bold;") 
            self.laser.set_emission(False)
            self.log("UV Laser toggled OFF")

    def toggle_wl(self, checked):
        if not self.thorlabs_led:
            self.btn_wl_toggle.setChecked(False)
            self.log("Cannot toggle WL: Thorlabs LED not connected.")
            return
            
        self._wl_state = not self._wl_state
        was_running = False
        try:
            if self.hw.core.is_sequence_running():
                was_running = True
                self.hw.core.stop_sequence_acquisition()
        except: pass

        if checked:
            self.btn_wl_toggle.setText("Thorlabs LED: ON")
            self.btn_wl_toggle.setStyleSheet("background-color: white; color: black; font-weight: bold;") 
            self.thorlabs_led.set_state(True)
            self.log("Thorlabs 780nm LED toggled ON")
        else:
            self.btn_wl_toggle.setText("Thorlabs LED: OFF")
            self.btn_wl_toggle.setStyleSheet("background-color: black; color: white; font-weight: bold;") 
            self.thorlabs_led.set_state(False)
            self.log("Thorlabs 780nm LED toggled OFF")

        try:
            if was_running:
                self.hw.core.start_continuous_sequence_acquisition(0)
        except: pass

    def _live_update_filter(self):
        if getattr(self.hw, 'core', None):
            filter_id = self.combo_filter_block.currentData()
            self.hw.set_filter_block(filter_id)
            try:
                self.hw.core.wait_for_device(config.DEVICE_FILTER_BLOCK)
            except Exception as e:
                self.log(f"Filter block wait warning: {e}")

    def _live_update_laser(self):
        self._live_uv_power = self.spin_laser_ma.value()
        
        if self.laser and getattr(self.laser, 'is_connected', False) and self.btn_uv_toggle.isChecked():
            self.laser.set_power_ma(self._live_uv_power)
            
        if self._is_acquiring:
            timestamp = round(time.time() - self._acq_start_time, 2)
            self._uv_power_history.append((timestamp, self._live_uv_power))
            self.log(f"[Live Update] Next UV Pulse will use {self._live_uv_power} mA")

    def _live_update_exposure(self):
        if getattr(self.hw, 'core', None):
            try:
                self.hw.core.set_exposure(float(self.spin_smlm_exp.value()))
            except Exception: pass

    def _toggle_basler_preview(self, checked):
        """Open/close the standalone Basler preview window."""
        if not self.basler_camera:
            self.btn_basler_preview.setChecked(False)
            self.log("Cannot preview: Basler camera not connected.")
            return

        if checked:
            self.btn_basler_preview.setText("Close Basler Preview")
            self.btn_basler_preview.setStyleSheet("background-color: #e74c3c;")
            self._create_basler_preview_window()
            self.log("Basler camera preview opened.")
        else:
            self.btn_basler_preview.setText("Open Basler Preview")
            self.btn_basler_preview.setStyleSheet("background-color: #3498db;")
            self._destroy_basler_preview_window()
            self.log("Basler camera preview closed.")

    def _create_basler_preview_window(self):
        if self.preview_window is None:
            self.preview_window = BaslerPreviewWindow(parent=self, camera=self.basler_camera.camera)
            self.preview_window.show()
            self.preview_window.raise_()
            self.preview_window.activateWindow()

    def _destroy_basler_preview_window(self):
        if self.preview_window:
            self.preview_window.close()
            self.preview_window = None

    def _on_preview_window_closed(self):
        """Called when the preview window is closed by the user."""
        self.btn_basler_preview.setChecked(False)
        self.btn_basler_preview.setText("Open Basler Preview")
        self.btn_basler_preview.setStyleSheet("background-color: #3498db;")
        self.log("Basler preview window closed.")

    def run_acquisition(self):
        self.log("\n===========================================")
        self.log("--- Preparing Pycromanager Engine ---")
        
        # NEW: Auto-close preview to free the DAQ
        if self.btn_basler_preview.isChecked():
            self.log("-> Auto-closing Basler preview to release DAQ...")
            self.btn_basler_preview.setChecked(False) # Triggers safe shutdown
            time.sleep(0.2) # Brief pause to ensure DAQ task fully unbinds
            
        if not getattr(self.hw, 'core', None):
            QMessageBox.warning(self, "Hardware Not Connected", "Please connect hardware first.")
            return

        # 1. AUTO-PAUSE LIVE STREAM
        try:
            if self.hw.core.is_sequence_running():
                self.hw.core.stop_sequence_acquisition()
                self.log("-> Auto-Paused existing Micro-Manager live stream.")
        except Exception as e:
            self.log(f"Warning checking sequence state: {e}")

        # Initialize Metadata & Live Tracking
        self._live_uv_power = self.spin_laser_ma.value()
        self._acq_start_time = time.time()
        self._uv_power_history = [(0.0, self._live_uv_power)]

        mode = self.combo_mode.currentText()
        root_save_dir = str(self.txt_dir.text()).strip()
        user_base_name = str(self.txt_name.text()).strip()
        
        mode_str = mode.replace(" ", "_")
        save_dir = Path(root_save_dir) / mode_str
        save_dir.mkdir(parents=True, exist_ok=True)
        
        date_str = datetime.datetime.now().strftime("%Y_%m_%d")
        if not user_base_name.startswith(date_str):
            base_save_name = f"{date_str}_{user_base_name}"
        else:
            base_save_name = user_base_name

        save_path = save_dir / base_save_name
        counter = 1
        while save_path.exists():
            save_path = save_dir / f"{base_save_name}_{counter}"
            counter += 1
        save_path.mkdir(parents=True, exist_ok=True)
        
        exp_smlm = float(self.spin_smlm_exp.value())
        exp_wl = float(self.spin_wl_exp.value())
        
        # Batch-Breaker
        if abs(exp_wl - exp_smlm) < 0.0001:
            exp_smlm_actual = exp_smlm + 0.001
            self.log(f"[WARNING] WL and SMLM exposures are identical ({exp_wl}ms).")
            self.log(f"[FIX] Setting SMLM exposure to {exp_smlm_actual}ms to break PycroManager batching.")
        else:
            exp_smlm_actual = exp_smlm
            
        events = []
        self._hardware_sequence.clear()
        active_laser_power = self._live_uv_power
        
        try:
            if mode == "Normal":
                frames = int(self.spin_normal_frames.value())
                self.log(f"[Queuing] {frames} frames in Normal mode.")
                events = multi_d_acquisition_events(num_time_points=frames)
                for e in events: e['exposure'] = exp_smlm

            elif mode == "Interleaved PALM":
                cycles = int(self.spin_palm_cycles.value())
                smlm_frames = int(self.spin_palm_smlm.value())
                wl_frames = int(self.spin_palm_wl.value())
                pa_ms = float(self.spin_palm_pa.value())
                
                # ALWAYS USE 568 LP (State 2) FOR SMLM FRAMES
                smlm_filter = 2
                
                total_frames = cycles * (smlm_frames + wl_frames)
                self.log(f"[Queuing] Interleaved PALM: {cycles} Cycles, {total_frames} Total Frames")
                events = multi_d_acquisition_events(num_time_points=total_frames)
                
                t_idx = 0
                for c in range(cycles):
                    for f in range(wl_frames):
                        evt = events[t_idx]
                        evt['exposure'] = exp_wl
                        self._hardware_sequence[t_idx] = {
                            'phase': 'WL', 
                            'frame_idx': f,
                            'cycle': c + 1,
                            'total_cycles': cycles
                        }
                        t_idx += 1
                        
                    for f in range(smlm_frames):
                        evt = events[t_idx]
                        evt['exposure'] = exp_smlm_actual 
                        self._hardware_sequence[t_idx] = {
                            'phase': 'SMLM', 
                            'frame_idx': f, 
                            'cycle': c + 1,
                            'total_cycles': cycles,
                            'pa_ms': pa_ms if f == 0 else 0, 
                            'smlm_filter': smlm_filter,
                            'laser_ma': active_laser_power
                        }
                        t_idx += 1
                        
            elif mode == "Z-Stack":
                steps = int(self.spin_z_steps.value())
                step_size = float(self.spin_z_step_um.value())
                self.log(f"[Queuing] Z-Stack: {steps} planes, step size: {step_size}µm.")
                
                current_z = 0.0
                z_dev = self.hw.core.get_focus_device()
                if z_dev: current_z = float(self.hw.core.get_position(z_dev))
                    
                start_z = current_z - ((steps / 2.0) * step_size)
                events = multi_d_acquisition_events(
                    num_time_points=1, z_start=start_z, z_end=start_z + ((steps - 1) * step_size), z_step=step_size
                )
                for e in events: e['exposure'] = exp_smlm

            elif mode == "Dual Imaging":
                frames = int(self.spin_dual_frames.value())
                self.log(f"[Queuing] {frames} frames in Dual Imaging mode (SMLM + Basler).")
                events = multi_d_acquisition_events(num_time_points=frames)
                for e in events: e['exposure'] = exp_smlm

        except Exception as e:
            self.log(f"Failed to build events: {e}")
            return

        # Define the Hook
        def pre_hardware_hook(event_or_events):
            try:
                is_list = isinstance(event_or_events, list)
                events_list = event_or_events if is_list else [event_or_events]
                
                count = len(events_list)
                t_start = events_list[0]['axes'].get('time', '?')
                
                first_t_idx = t_start
                if first_t_idx == '?': return event_or_events
                
                hw_params = self._hardware_sequence.get(first_t_idx)
                if hw_params and hw_params['frame_idx'] == 0:
                    # Give DAQ thread time to start listening on frame 0
                    time.sleep(0.2)
                    phase = hw_params['phase']
                    
                    # --- NEW CYCLE LOGGING ---
                    cycle = hw_params.get('cycle', 1)
                    tot_cycles = hw_params.get('total_cycles', 1)
                    self.log(f"** [Cycle {cycle}/{tot_cycles}] Phase Transition -> [{phase} Mode] **")
                    # -------------------------
                    
                    if phase == 'WL':
                        self.hw.set_filter_block(1)
                        time.sleep(0.8)  
                        if self.thorlabs_led:
                            self.thorlabs_led.set_state(True)
                        time.sleep(2.0)  
                            
                    elif phase == 'SMLM':
                        if self.thorlabs_led:
                            self.thorlabs_led.set_state(False)
                        self.hw.set_filter_block(1) 
                        time.sleep(0.3)
                        
                        pa_ms = hw_params.get('pa_ms', 0)
                        laser_ma = self._live_uv_power # Pulls dynamic value
                        
                        if pa_ms > 0:
                            self.log(f"  -> Pulsing UV Laser ({laser_ma}mA) for {pa_ms}ms")
                            if self.laser and getattr(self.laser, 'is_connected', False):
                                self.laser.set_power_ma(laser_ma)
                                self.laser.set_emission(True)
                                time.sleep(pa_ms / 1000.0)
                                self.laser.set_emission(False)
                            else:
                                time.sleep(pa_ms / 1000.0)
                                    
                        smlm_filt = hw_params.get('smlm_filter', 2)
                        self.hw.set_filter_block(smlm_filt)
                        time.sleep(0.8) 
                        
            except Exception as e:
                self.log(f"Hardware hook error: {e}")
                
            return event_or_events

# 2. START BASLER RECORDING IF ENABLED (Normal & Dual mode)
        self._basler_recording_started = False
        self._basler_thread = None
        if (mode == "Normal" and self.chk_basler_record_acq.isChecked() and self.basler_camera) or (mode == "Dual Imaging" and self.basler_camera):
            try:
                n_divider = self.spin_basler_n.value()
                self.basler_camera.set_n_divider(n_divider)
                
                if not self.basler_camera.start_acquisition():
                    self.log("Warning: Failed to start Basler camera acquisition")
                else:
                    basler_filename = save_path / f"{save_path.name}_basler.tif"
                    
                    if self.basler_camera.start_recording(str(basler_filename)):
                        # Calculate exact target frames so the camera knows when to shut off
                        expected_b_frames = None
                        if mode == "Normal":
                            expected_b_frames = int(self.spin_normal_frames.value()) // n_divider
                        elif mode == "Dual Imaging":
                            expected_b_frames = int(self.spin_dual_frames.value()) // n_divider

                        self._basler_thread = BaslerRecordingThread(self.basler_camera, expected_frames=expected_b_frames)
                        self._basler_thread.error_signal.connect(lambda e: self.log(f"Basler thread error: {e}"))
                        self._basler_thread.start()
                        
                        self._basler_recording_started = True
                        self.log(f"Basler recording started: {basler_filename} (Target: {expected_b_frames} frames)")
                        
                        # --- STARTUP FIX ---
                        # Give DAQ hardware 0.5s to start buffering BEFORE Micro-Manager fires the first pulse
                        time.sleep(0.5) 
                    else:
                        self.log("Warning: Failed to start Basler recording")
            except Exception as e:
                self.log(f"Failed to start Basler recording: {e}")

        # 3. DISPATCH BACKGROUND THREAD (Prevents GUI Freeze)
        self.log(f"\nAttempting Acquisition to {save_path}...")
        
        self._is_acquiring = True
        self.btn_run.setEnabled(False)
        self.btn_run.setText("ACQUIRING... (GUI is Active)")
        self.btn_run.setStyleSheet("background-color: #e67e22; font-size: 14px; padding: 15px; font-weight: bold;")
        
        self._acq_worker = AcquisitionWorker(str(save_path), save_path.name, events, pre_hardware_hook, mode)
        self._acq_worker.log_signal.connect(self.log)
        self._acq_worker.finished_signal.connect(self._on_acq_finished)
        self._acq_worker.error_signal.connect(self._on_acq_error)
        self._acq_worker.start()

    # --- THREAD CALLBACKS ---

    def _on_acq_finished(self, save_dir, save_name, mode):
        # Stop Basler recording safely without the Guillotine
        if hasattr(self, '_basler_recording_started') and self._basler_recording_started:
            try:
                self.log("-> Micro-Manager finished. Allowing Basler to catch final frames...")
                if self._basler_thread:
                    self._basler_thread.initiate_graceful_stop()
                    
                    # Give thread up to 2.5 seconds to naturally finish writing to disk
                    self._basler_thread.wait(2500) 
                    
                    if self._basler_thread.isRunning():
                        self.log("-> Forcing Basler stop (Expected frames not reached or aborted).")
                        self._basler_thread.stop()
                        
                    self._basler_thread = None
                    
                if self.basler_camera:
                    self.basler_camera.stop_recording()
                    self.basler_camera.stop_acquisition()
                    self.log("-> Basler recording safely closed.")
                    
                self._basler_recording_started = False
            except Exception as e:
                self.log(f"Error stopping Basler recording: {e}")
        
        self.log(f"\n===========================================")
        self.log(f"Acquisition successfully finished.")
        
        # 3. AUTO-RESET FILTER AT THE END OF SEQUENCE
        if mode == "Interleaved PALM":
            try:
                self.log("-> Sequence Complete. Auto-resetting Filter Block to Empty (1).")
                self.hw.set_filter_block(1)
            except Exception as e:
                self.log(f"Failed to reset filter: {e}")
        
        self._write_protocol_json(save_dir, save_name, mode) 
        self._is_acquiring = False
        
        self.btn_run.setEnabled(True)
        self.btn_run.setText("START PYCROMANAGER ACQUISITION")
        self.btn_run.setStyleSheet("background-color: #27ae60; font-size: 14px; padding: 15px;")

    def _on_acq_error(self, err_trace):
        self.log(f"CRASH in background Acquisition:\n{err_trace}")
        QMessageBox.critical(self, "Acquisition Error", f"Unhandled Error:\n{err_trace}")
        
        # Stop Basler recording if it was started
        if hasattr(self, '_basler_recording_started') and self._basler_recording_started:
            try:
                if self.basler_camera:
                    self.basler_camera.stop_recording()
                    self.basler_camera.stop_acquisition()
                if self._basler_thread:
                    self._basler_thread.stop()
                    self._basler_thread = None
                self._basler_recording_started = False
                self.log("Basler recording stopped due to acquisition error.")
            except Exception as e:
                self.log(f"Error stopping Basler recording: {e}")
        
        self._is_acquiring = False
        
        self.btn_run.setEnabled(True)
        self.btn_run.setText("START PYCROMANAGER ACQUISITION")
        self.btn_run.setStyleSheet("background-color: #27ae60; font-size: 14px; padding: 15px;")

    def _write_protocol_json(self, save_dir, save_name, mode):
        acq_dir = Path(save_dir)
        if not acq_dir.exists():
            acq_dir.mkdir(parents=True, exist_ok=True)
            
        mm_state = {}
        if getattr(self.hw, 'core', None):
            try:
                state_cache = self.hw.core.get_system_state_cache()
                for i in range(state_cache.size()):
                    prop = state_cache.get_setting(i)
                    dev = prop.get_device_label()
                    name = prop.get_property_name()
                    val = prop.get_property_value()
                    if dev not in mm_state: mm_state[dev] = {}
                    mm_state[dev][name] = str(val)
            except Exception as e:
                pass
            
        saved_filter_name = "568 LP (Forced)" if mode == "Interleaved PALM" else self.combo_filter_block.currentText()
            
        protocol = {
            "mode": mode, "save_dir": str(acq_dir), "save_name": acq_dir.name,
            "hardware_state": {
                "laser_ma": self._live_uv_power, 
                "thorlabs_led_state": self.thorlabs_led.get_state() if self.thorlabs_led else False,
                "thorlabs_led_daq_channel": "Dev1/ao1",
                "filter_block": saved_filter_name,
                "smlm_exposure_ms": self.spin_smlm_exp.value(),
                "wl_exposure_ms": self.spin_wl_exp.value()
            },
            "live_uv_power_changes_mA": self._uv_power_history, 
            "micromanager_parameters": mm_state
        }
        
        if mode == "Normal":
            protocol["params"] = {"total_frames": self.spin_normal_frames.value()}
        elif mode == "Interleaved PALM":
            protocol["params"] = {
                "cycles": self.spin_palm_cycles.value(), "smlm_frames": self.spin_palm_smlm.value(),
                "wl_frames": self.spin_palm_wl.value(), "pa_pulse_ms": self.spin_palm_pa.value()
            }
        elif mode == "Z-Stack":
            protocol["params"] = {
                "z_steps": self.spin_z_steps.value(), "z_step_um": self.spin_z_step_um.value()
            }
        elif mode == "Dual Imaging":
            protocol["params"] = {
                "total_frames": self.spin_dual_frames.value(),
                "basler_skip_frames": self.spin_basler_n.value() # FIX: Updated to match global variable
            }
            
        json_path = acq_dir / "protocol.json"
        with open(json_path, "w") as f: json.dump(protocol, f, indent=4)
        self.log(f"Saved protocol.json securely to {acq_dir.name}")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    apply_dark_theme(app)
    window = PycromanagerAcquisitionApp()
    window.show()
    sys.exit(app.exec_())