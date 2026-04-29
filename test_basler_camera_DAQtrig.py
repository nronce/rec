import sys
import os
import time
from datetime import datetime
import cv2
import numpy as np
import nidaqmx
import tifffile
from pypylon import pylon
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QLabel, QPushButton, QSpinBox, 
                             QDoubleSpinBox, QFileDialog, QCheckBox, QSlider, QSizePolicy)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap

# --- Hardware Config ---
DAQ_CHANNEL = "Dev1/ai0"
TRIGGER_THRESHOLD = 2.0 

class CameraThread(QThread):
    change_pixmap_signal = pyqtSignal(np.ndarray)
    fps_signal = pyqtSignal(float)

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
            task = nidaqmx.Task()
            task.ai_channels.add_ai_voltage_chan(DAQ_CHANNEL)
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
                    is_high = voltage > TRIGGER_THRESHOLD
                    
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
                            
                            # 3. Append to TIFF Stack
                            if self.is_recording and self.tiff_writer is not None:
                                self.tiff_writer.write(raw_img, contiguous=True)

                            # 4. Emit for UI
                            self.change_pixmap_signal.emit(raw_img.copy())
                            
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

class WL_Viewer(QMainWindow):
    def __init__(self, camera):
        super().__init__()
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
        self.setWindowTitle("WL Sync - TIFF Stack Mode")
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
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = os.path.join(self.save_folder, f"WL_Stack_{timestamp}.tif")
            
            self.thread.start_recording(filepath)
            self.btn_record.setText("⏹ STOP RECORDING")
            self.btn_record.setStyleSheet("background: #d32f2f; color: white; font-weight: bold;")
        else:
            self.thread.stop_recording()
            self.btn_record.setText("⏺ START TIFF STACK")
            self.btn_record.setStyleSheet("")

    def process_display(self, raw_img):
        if self.auto_contrast:
            c_min, c_max = np.min(raw_img), np.max(raw_img)
        else:
            c_min, c_max = self.vmin, self.vmax

        span = max(1, c_max - c_min)
        display_img = np.clip((raw_img.astype(np.float32) - c_min) / span * 255, 0, 255).astype(np.uint8)

        h, w = display_img.shape
        qimg = QImage(display_img.data, w, h, w, QImage.Format_Grayscale8)
        self.img_label.setPixmap(QPixmap.fromImage(qimg).scaled(
            self.img_label.width(), self.img_label.height(), Qt.KeepAspectRatio))

    def closeEvent(self, event):
        self.thread.stop()
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    try:
        factory = pylon.TlFactory.GetInstance()
        cam = pylon.InstantCamera(factory.CreateDevice(factory.EnumerateDevices()[0]))
        window = WL_Viewer(cam)
        window.show()
        sys.exit(app.exec_())
    except Exception as e:
        print(f"Could not start application: {e}")