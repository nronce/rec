import os
import sys
import time
import json
import yaml
import numpy as np
import tifffile
from pathlib import Path
from pycromanager import Core
from datetime import datetime

# Import config first so we have the KINESIS_PATH
import config

# --- .NET Kinesis Setup ---
# 1. CRITICAL: Tell Windows where to find the underlying C++ USB drivers
if config.KINESIS_PATH not in os.environ["PATH"]:
    os.environ["PATH"] = config.KINESIS_PATH + os.pathsep + os.environ["PATH"]

# Optional fallback for newer Python versions on Windows
try:
    os.add_dll_directory(config.KINESIS_PATH)
except AttributeError:
    pass

# 2. Tell Python where to find the .NET wrappers
sys.path.append(config.KINESIS_PATH)

# 3. Now it is safe to load the CLR
import clr
from System import Decimal, Enum

try:
    clr.AddReference("Thorlabs.MotionControl.DeviceManagerCLI")
    clr.AddReference("Thorlabs.MotionControl.TCube.LaserDiodeCLI")
    from Thorlabs.MotionControl.DeviceManagerCLI import DeviceManagerCLI
    from Thorlabs.MotionControl.TCube.LaserDiodeCLI import TCubeLaserDiode
    KINESIS_OK = True
except Exception as e:
    KINESIS_OK = False
    print(f"Laser DLL Load Error: {e}")

def ensure_dir(path: Path):
    if not path.exists():
        path.mkdir(parents=True, exist_ok=True)
    return path

class Microscope:
    def __init__(self):
        self.core = None
        self.camera_device = None
        self.camera_ready = False
        self.z_device = None
        self.wl_device = None  # Keep for backward compatibility, but will be replaced by LED
        self.thorlabs_led = None  # New LED control
        self.wl_desired_state = None  # Track desired WL state to restore after sequence restart
        self.PROP_STATE = "State"
        self.PROP_INTENSITY = "Intensity"
        self.PROP_COMP_CONTROL = "ComputerControl"
        self.pfs_requested = False

    def connect(self, port=4827):
        if self.core is None:
            try:
                self.core = Core(port=port)
                # FIX: Removed the call to _check_camera_ready as it was deleted
                
                self._detect_z_device()
                # Initialize Thorlabs LED instead of detecting WL device
                self._init_thorlabs_led()
            except Exception as e:
                print(f"[Hardware] Connection Failed: {e}")
        return self.core
    
    def _init_thorlabs_led(self):
        """Initialize Thorlabs 780nm LED control via DAQ."""
        try:
            self.thorlabs_led = ThorlabsLED()
            if self.thorlabs_led.connect():
                print("[Hardware] Thorlabs 780nm LED initialized via DAQ AO1")
            else:
                print("[Hardware] Warning: Thorlabs LED initialization failed")
        except Exception as e:
            print(f"[Hardware] LED initialization error: {e}")
            self.thorlabs_led = None
    
    def _set_safe(self, device, prop, value):
        """
        FIX: Restored this method. It is required by set_wl_intensity.
        Since the camera is not in MM, the 'is_sequence_running' check will 
        usually be False, but this wrapper is still needed to set properties.
        """
        if not self.core: return
        try:
            # If for some reason a sequence is running in MM, stop it safely
            if self.core.is_sequence_running():
                self.core.stop_sequence_acquisition()
                time.sleep(0.1)
            
            self.core.set_property(device, prop, str(value))
        except Exception as e:
            print(f"[Hardware] FAILED to set {prop} on {device}. Error: {e}")

    def set_wl_intensity(self, value):
        """Set white light intensity using Thorlabs LED via DAQ."""
        if self.thorlabs_led:
            self.thorlabs_led.set_intensity(int(value))
        else:
            print("[Hardware] Warning: Thorlabs LED not available")

    def _detect_z_device(self):
        if not self.core:
            self.z_device = None
            return None
        try:
            z_dev = self.core.get_focus_device()
            if z_dev and str(z_dev).strip() not in ("", "None", "null"):
                self.z_device = z_dev
                print(f"[Hardware] Z device: '{self.z_device}'")
                return self.z_device
        except Exception:
            pass
        self.z_device = None
        print("[Hardware] Warning: No focus (Z) device detected.")
        return None

    def _get_z_device(self):
        if not self.core:
            return None
        if self.pfs_requested:
            try:
                devices = self.core.get_loaded_devices()
                dev_list = [devices.get(i) for i in range(devices.size())]
                if "TIPFSOffset" in dev_list:
                    return "TIPFSOffset"
            except Exception:
                pass
        if self.z_device is None:
            self._detect_z_device()
        return self.z_device

    def _get_offset_property(self, device):
        for prop in ("Position", "Offset", "FocusOffset", "PFSOffset"):
            try:
                if self.core.has_property(device, prop):
                    return prop
            except Exception:
                continue
        return "Position"

    def _detect_wl_device(self):
        try:
            devices = self.core.get_loaded_devices()
            dev_list = [devices.get(i) for i in range(devices.size())]
            candidates = ["lamp", "dia", "tl", "led", "shutter"]
            
            for dev in dev_list:
                if any(c in dev.lower() for c in candidates):
                    try:
                        if self.core.has_property(dev, self.PROP_STATE) and \
                           self.core.has_property(dev, self.PROP_INTENSITY):
                            self.wl_device = dev
                            print(f"[Hardware] Auto-detected WL Device: '{dev}'")
                            return
                    except:
                        continue
            print("[Hardware] Warning: No suitable White Light device found.")
        except Exception as e:
            print(f"[Hardware] Detection Error: {e}")

    def get_z_position(self):
        if not self.core: return None
        try:
            device = self._get_z_device()
            if not device: return None
            try:
                return float(self.core.get_position(device))
            except Exception:
                prop = self._get_offset_property(device)
                return float(self.core.get_property(device, prop))
        except Exception as e:
            print(f"[Hardware] Z position read error: {e}")
            return None

    def set_z_position(self, z_um: float, relative: bool = True):
        if not self.core: return
        try:
            device = self._get_z_device()
            if not device: return
            if relative:
                current = self.get_z_position()
                target = float(z_um) if current is None else float(current + z_um)
            else:
                target = float(z_um)
            try:
                self.core.set_position(device, target)
            except Exception:
                prop = self._get_offset_property(device)
                self.core.set_property(device, prop, str(target))
        except Exception as e:
            print(f"[Hardware] Z position set error: {e}")

    def set_pfs_enabled(self, enabled: bool):
        if not self.core: return
        try:
            self.core.set_property("Core", "AutoFocus", "TIPFSStatus")
            self.core.set_property("TIPFSStatus", "State", "On" if enabled else "Off")
            self.pfs_requested = bool(enabled)
        except Exception as e:
            print(f"[Hardware] PFS enable error: {e}")

    def is_pfs_enabled(self):
        if not self.core: return False
        try:
            status = self.core.get_property("TIPFSStatus", "Status")
            return str(status).strip().lower() in ("locked", "locked in focus")
        except Exception as e:
            print(f"[Hardware] PFS status error: {e}")
            return False

    def set_filter_block(self, state_value):
        if self.core:
            try:
                self.core.set_property("TIFilterBlock1", "State", str(state_value))
                print(f"[Hardware] Filter block set to state {state_value}")
            except Exception as e:
                print(f"[Hardware] Filter block error: {e}")

    def set_wl_state(self, on: bool, force=False):
        """Set white light state using Thorlabs LED via DAQ."""
        if self.thorlabs_led:
            self.thorlabs_led.set_state(on)
            self.wl_desired_state = on
        else:
            print("[Hardware] Warning: Thorlabs LED not available")

    def get_wl_state(self):
        """Get white light state from Thorlabs LED."""
        if self.thorlabs_led:
            return self.thorlabs_led.get_state()
        return False

class ThorlabsLaser:
    def __init__(self):
        self.serial_no = config.LASER_SERIAL
        self.ldc = None
        self.is_connected = False
        self.last_setpoint_ma = config.DEFAULTS.get("laser_ma", 0.0)

    def connect(self):
        if not KINESIS_OK:
            print("[Laser] Kinesis DLLs not loaded. Dummy mode.")
            return

        try:
            # 1. Ask Windows to scan the USB bus
            DeviceManagerCLI.BuildDeviceList()
            time.sleep(0.2)
            
            # 2. Verify the device is actually available to python
            available_devices = list(DeviceManagerCLI.GetDeviceList())
            
            if not available_devices:
                print(f"[Laser] WARNING: No devices found on USB bus. Close Kinesis GUI if open.")
                return
                
            if self.serial_no not in available_devices:
                print(f"[Laser] WARNING: Found {available_devices}, but not {self.serial_no}.")
                return

            # 3. Safe to connect
            self.ldc = TCubeLaserDiode.CreateTCubeLaserDiode(self.serial_no)
            
            # Prevent crashes if a ghost connection exists
            if not self.ldc.IsConnected:
                self.ldc.Connect(self.serial_no)
                
            self.ldc.StartPolling(100)
            
            # Initialize Settings & Force Software Control
            self.ldc.RequestSettings()
            time.sleep(0.5) 
            
            source_type = self.ldc.GetControlSource().GetType()
            self.ldc.SetControlSource(Enum.ToObject(source_type, 0)) # Software
            self.ldc.SetOpenLoop() # Constant Current
            
            try:
                self.set_power_ma(self.last_setpoint_ma)
            except Exception as e:
                print(f"[Laser] Initial setpoint error: {e}")
            
            self.is_connected = True
            print(f"[Laser] Connected and Initialized: {self.serial_no}")
            
        except Exception as e:
            print(f"[Laser] Connection Error: {e}")
            self.is_connected = False

    def set_power_ma(self, value):
        if not self.is_connected: return
        try:
            self.ldc.RequestSettings()
            time.sleep(0.1)
            target = Decimal(float(value))
            self.ldc.SetLaserSetPoint(target)
            time.sleep(0.1)
            self.last_setpoint_ma = float(value)
        except Exception as e:
            print(f"[Laser] Set Power Error: {e}")

    def set_emission(self, on: bool):
        if not self.is_connected: return
        try:
            if on:
                if self.last_setpoint_ma is None or self.last_setpoint_ma <= 0:
                    self.last_setpoint_ma = float(config.DEFAULTS.get("laser_ma", 25.0))
                try: self.set_power_ma(self.last_setpoint_ma)
                except Exception: pass
                
                self.ldc.EnableDevice()
                time.sleep(0.1)
                self.ldc.SetOn()
                
                try: self.set_power_ma(self.last_setpoint_ma)
                except Exception: pass
            else:
                self.ldc.SetOff()
        except Exception as e:
            print(f"[Laser] Emission Error: {e}")

    def close(self):
        if self.ldc:
            self.set_emission(False)
            if self.is_connected:
                self.ldc.Disconnect(True)


# --- BASLER CAMERA WITH DAQ TRIGGER ---
try:
    import nidaqmx
    from pypylon import pylon
    BASLER_AVAILABLE = True
except ImportError:
    BASLER_AVAILABLE = False
    print("[Basler] pypylon or nidaqmx not available. Basler camera disabled.")


class ThorlabsLED:
    """
    Controls Thorlabs 780nm LED via DAQ analog output in modulation mode.
    Uses AO1 (analog output channel 1) for intensity control.
    """
    
    def __init__(self, daq_channel="Dev1/ao1", max_voltage=5.0):
        self.daq_channel = daq_channel
        self.max_voltage = max_voltage
        self.current_intensity = 0  # 0-10 scale
        self.is_connected = False
        self.daq_task = None
        
    def connect(self):
        """Initialize DAQ analog output for LED control."""
        try:
            self.daq_task = nidaqmx.Task()
            self.daq_task.ao_channels.add_ao_voltage_chan(
                self.daq_channel, 
                min_val=0.0, 
                max_val=self.max_voltage
            )
            # Set initial voltage to 0 (LED off)
            self.daq_task.write(0.0)
            self.is_connected = True
            print(f"[LED] Connected to {self.daq_channel} (max {self.max_voltage}V)")
            return True
        except Exception as e:
            print(f"[LED] Connection Error: {e}")
            self.is_connected = False
            return False
    
    def disconnect(self):
        """Close DAQ task and turn off LED."""
        if self.daq_task:
            try:
                self.daq_task.write(0.0)  # Ensure LED is off
                self.daq_task.close()
                self.daq_task = None
            except Exception as e:
                print(f"[LED] Disconnect error: {e}")
        self.is_connected = False
    
    def set_intensity(self, intensity):
        """
        Set LED intensity (0-10 scale).
        Converts to voltage: intensity * max_voltage / 10
        """
        if not self.is_connected or not self.daq_task:
            print("[LED] Not connected")
            return
        
        # Clamp intensity to 0-10 range
        intensity = max(0, min(10, intensity))
        self.current_intensity = intensity
        
        # Convert to voltage (0-10 scale -> 0-max_voltage)
        voltage = (intensity / 10.0) * self.max_voltage
        
        try:
            self.daq_task.write(voltage)
            print(f"[LED] Set intensity {intensity}/10 -> {voltage:.2f}V")
        except Exception as e:
            print(f"[LED] Set intensity error: {e}")
    
    def set_state(self, on):
        """Turn LED on/off using fixed DAQ output voltage."""
        if not self.daq_task:
            print("[LED] Not connected")
            return

        try:
            if on:
                self.current_intensity = self.max_voltage
                self.daq_task.write(self.max_voltage)
                print("[LED] Turned ON")
            else:
                self.current_intensity = 0
                self.daq_task.write(0.0)
                print("[LED] Turned OFF")
        except Exception as e:
            print(f"[LED] Set state error: {e}")
    
    def get_intensity(self):
        """Get current intensity setting (0-10)."""
        return self.current_intensity
    
    def get_state(self):
        """Get current state (True if intensity > 0)."""
        return self.current_intensity > 0


class BaslerCamera:
    """
    Manages Basler camera with DAQ trigger for synchronized frame acquisition.
    Supports N-frame divider to pick 1 in N frames from trigger pulses.
    """
    
    def __init__(self, daq_channel="Dev1/ai0", trigger_threshold=2.0, exposure_ms=30.0):
        if not BASLER_AVAILABLE:
            raise RuntimeError("Basler camera dependencies not available")
        
        self.daq_channel = daq_channel
        self.trigger_threshold = trigger_threshold
        self.exposure_ms = exposure_ms
        self.camera = None
        self.is_running = False
        self.n_divider = 1  # Pick 1 in N frames
        
        # Recording state
        self.is_recording = False
        self.tiff_writer = None
        self.record_path = None
        
        # Stats
        self.frame_count = 0
        self.frames_written = 0
        
    def connect(self):
        """Initialize and open the Basler camera."""
        if self.camera is not None:
            return
        
        try:
            # Create pylon TlFactory object
            tl_factory = pylon.TlFactory.GetInstance()
            devices = tl_factory.EnumerateDevices()
            
            if len(devices) == 0:
                raise RuntimeError("No Basler cameras found")
            
            # Connect to first available camera
            self.camera = pylon.InstantCamera(tl_factory.CreateDevice(devices[0]))
            self.camera.Open()
            
            # Configure pixel format to 12-bit
            try:
                self.camera.PixelFormat.SetValue("Mono12")
            except:
                try:
                    self.camera.PixelFormat.SetValue("Mono16")
                except:
                    pass
            
            print(f"[Basler] Connected: {self.camera.GetDeviceInfo().GetModelName()}")
            return True
            
        except Exception as e:
            print(f"[Basler] Connection Error: {e}")
            self.camera = None
            return False
    
    def disconnect(self):
        """Close and cleanup the camera."""
        if self.camera:
            try:
                self.camera.Close()
            except:
                pass
            self.camera = None
    
    def set_exposure(self, exposure_ms):
        """Set camera exposure time in milliseconds."""
        if not self.camera:
            return
        try:
            self.camera.ExposureTime.SetValue(exposure_ms * 1000.0)  # Convert to µs
            self.exposure_ms = exposure_ms
        except Exception as e:
            print(f"[Basler] Set exposure error: {e}")
    
    def set_n_divider(self, n):
        """Set frame divider (pick 1 in N frames from triggers)."""
        self.n_divider = max(1, int(n))
    
    def start_acquisition(self):
        """Configure camera for acquisition and start grabbing."""
        if not self.camera:
            print("[Basler] Camera not connected")
            return False
        
        if self.is_running:
            return True
        
        try:
            self.camera.TriggerSelector.SetValue("FrameStart")
            self.camera.TriggerMode.SetValue("On")
            self.camera.TriggerSource.SetValue("Software")
            try:
                self.camera.TriggerActivation.SetValue("RisingEdge")
            except Exception:
                pass

            # FIX: Actually tell the camera engine to start grabbing
            if not self.camera.IsGrabbing():
                self.camera.StartGrabbing(pylon.GrabStrategy_OneByOne)

            self.is_running = True
            print("[Basler] Camera configured and grabbing for acquisition")
            return True
            
        except Exception as e:
            print(f"[Basler] Configure acquisition error: {e}")
            return False
    
    def stop_acquisition(self):
        """Stop camera acquisition configuration."""
        if self.is_running:
            self.is_running = False
            # FIX: Stop the camera grabbing engine
            try:
                if self.camera and self.camera.IsGrabbing():
                    self.camera.StopGrabbing()
            except Exception:
                pass
            print("[Basler] Acquisition configuration stopped")
    
    def start_recording(self, filepath):
        """Start recording frames to TIFF file."""
        try:
            self.tiff_writer = tifffile.TiffWriter(filepath, bigtiff=True)
            self.record_path = filepath
            self.is_recording = True
            self.frames_written = 0
            print(f"[Basler] Recording started: {filepath}")
            return True
        except Exception as e:
            print(f"[Basler] Failed to start recording: {e}")
            return False
    
    def stop_recording(self):
        """Stop recording and close TIFF file."""
        if self.tiff_writer:
            try:
                self.tiff_writer.close()
                self.tiff_writer = None
                print(f"[Basler] Recording saved ({self.frames_written} frames)")
            except Exception as e:
                print(f"[Basler] Error closing recording: {e}")
        self.is_recording = False
    
    def grab_frame_with_daq_trigger(self, daq_task=None):
        """
        Grab a single frame triggered by DAQ input.
        Returns frame array if successful, None otherwise.
        """
        if not self.camera or not self.is_running:
            return None
        
        try:
            was_high = False
            pulse_count = 0
            
            # FIX: Use self.is_running instead of True so the thread can exit
            while self.is_running:
                # Check DAQ trigger
                trigger_now = False
                if daq_task:
                    voltage = daq_task.read()
                    is_high = voltage > self.trigger_threshold
                    
                    if is_high and not was_high:
                        pulse_count += 1
                        if pulse_count >= self.n_divider:
                            trigger_now = True
                            pulse_count = 0
                    was_high = is_high
                else:
                    # Fallback: continuous triggering if DAQ unavailable
                    trigger_now = True
                    time.sleep(0.01)
                
                if trigger_now:
                    # Wait for camera ready and trigger
                    if self.camera.WaitForFrameTriggerReady(200):
                        self.camera.ExecuteSoftwareTrigger()
                        res = self.camera.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)
                        
                        if res.GrabSucceeded():
                            frame = res.Array.copy()  # 12-bit RAW data
                            
                            # Transpose to match Kuro orientation
                            frame = frame.T
                            
                            # FIX: Force the memory to be contiguous after transposing
                            frame = np.ascontiguousarray(frame)
                            
                            # Write to recording if active
                            if self.is_recording and self.tiff_writer:
                                # FIX: Removed contiguous=True to prevent ValueError crashes
                                self.tiff_writer.write(frame)
                                self.frames_written += 1
                            
                            # --- THESE LINES MUST BE OUTDENTED ---
                            self.frame_count += 1
                            res.Release()
                            return frame
                        else:
                            res.Release()
                            return None
                        
        except Exception as e:
            print(f"[Basler] Frame grab error: {e}")
            return None
    
    def grab_frame_batch(self, num_frames, daq_task=None, on_frame_callback=None):
        """
        Grab multiple frames with DAQ trigger.
        
        Args:
            num_frames: Number of frames to grab
            daq_task: nidaqmx task for trigger input
            on_frame_callback: Optional callback function called for each frame
        
        Returns:
            List of frame arrays
        """
        frames = []
        for i in range(num_frames):
            frame = self.grab_frame_with_daq_trigger(daq_task)
            if frame is not None:
                frames.append(frame)
                if on_frame_callback:
                    on_frame_callback(i + 1, num_frames, frame)
            else:
                print(f"[Basler] Failed to grab frame {i + 1}")
                break
        return frames


class Recorder:
    def __init__(self, root_dir, experiment_name, params):
        mode_root = ensure_dir(Path(root_dir))
        run_folder = mode_root / experiment_name
        iteration = 1
        while run_folder.exists():
            run_folder = mode_root / f"{experiment_name}_{iteration}"
            iteration += 1

        self.root = ensure_dir(run_folder)
        self.params = dict(params or {})
        self.initial_params = dict(self.params)
        self.param_changes = []
        self.mode = params.get("mode", "standard")
        self.smlm_writer = None
        self.wl_writer = None
        self.frames_written = 0
        self.wl_written = 0
        self.start_time = datetime.now()
        self._closed = False
        self._status = "recording"
        self._error_message = None
        self._last_checkpoint_total = 0
        self.checkpoint_interval_frames = max(1, int(params.get("checkpoint_interval_frames", 200)))
        self.protocol_path = self.root / "protocol.json"
        self.protocol_in_progress_path = self.root / "protocol.in_progress.json"
        
        base_name = experiment_name
        file_iteration = 1
        file_name = base_name
        smlm_path = self.root / f"{file_name}_smlm.tif"
        while smlm_path.exists():
            file_name = f"{base_name}_{file_iteration}"
            smlm_path = self.root / f"{file_name}_smlm.tif"
            file_iteration += 1
        
        if self.mode in ["standard", "interleaved", "interleaved_palm", "pfs_zstack"]:
            self.smlm_path = self.root / f"{file_name}_smlm.tif"
            self.smlm_writer = tifffile.TiffWriter(self.smlm_path, bigtiff=True) 
        
        if self.mode in ["interleaved", "interleaved_palm"]:
            self.wl_path = self.root / f"{file_name}_wl.tif"
            self.wl_writer = tifffile.TiffWriter(self.wl_path, bigtiff=True) 

        self._write_protocol(in_progress=True)

    def _flush_writer(self, writer):
        if writer is None: return
        fh = getattr(writer, "filehandle", None)
        if fh is None: fh = getattr(writer, "_fh", None)
        if fh is None: return
        try: fh.flush()
        except Exception: pass
        try:
            base_fh = getattr(fh, "_fh", None)
            if base_fh is None: base_fh = fh
            os.fsync(base_fh.fileno())
        except Exception: pass

    def _write_json_atomic(self, path: Path, data: dict):
        tmp_path = path.with_suffix(path.suffix + ".tmp")
        with open(tmp_path, "w") as f:
            json.dump(data, f, indent=2)
            f.flush()
            try: os.fsync(f.fileno())
            except: pass
        tmp_path.replace(path)

    def _build_protocol(self, end_time=None):
        final_time = end_time if end_time is not None else datetime.now()
        duration = (final_time - self.start_time).total_seconds()
        return {
            "params": self.params,
            "start_time": self.start_time.isoformat(),
            "end_time": final_time.isoformat() if end_time is not None else None,
            "duration_seconds": duration,
            "counts": {"smlm_frames": self.frames_written, "wl_frames": self.wl_written},
            "mode": self.mode,
            "status": self._status,
            "paths": {"smlm": str(getattr(self, 'smlm_path', None)), "wl": str(getattr(self, 'wl_path', None))}
        }

    def _write_protocol(self, in_progress=False):
        meta = self._build_protocol(end_time=None if in_progress else datetime.now())
        target = self.protocol_in_progress_path if in_progress else self.protocol_path
        self._write_json_atomic(target, meta)

    def _checkpoint_if_needed(self):
        total = self.frames_written + self.wl_written
        if (total - self._last_checkpoint_total) < self.checkpoint_interval_frames: return
        self._last_checkpoint_total = total
        self._flush_writer(self.smlm_writer)
        self._flush_writer(self.wl_writer)
        self._write_protocol(in_progress=True)

    def write_smlm(self, image):
        if self.smlm_writer is None: raise RuntimeError("SMLM writer not initialized!")
        self.smlm_writer.write(image, photometric='minisblack') 
        self.frames_written += 1
        self._checkpoint_if_needed()

    def write_wl(self, image, metadata=None):
        if self.wl_writer is None: raise RuntimeError("WL writer not initialized!")
        self.wl_writer.write(image, photometric='minisblack') 
        self.wl_written += 1
        self._checkpoint_if_needed()

    def close(self, status="completed", error_message=None):
        if self._closed: return
        self._closed = True
        self._status = status
        self._error_message = error_message
        try:
            self._flush_writer(self.smlm_writer)
            self._flush_writer(self.wl_writer)
        except Exception: pass
        try:
            if self.smlm_writer:
                self.smlm_writer.close()
                self.smlm_writer = None
        except Exception: pass
        try:
            if self.wl_writer:
                self.wl_writer.close()
                self.wl_writer = None
        except Exception: pass

        end_time = datetime.now()
        meta = self._build_protocol(end_time=end_time)
        try:
            self._write_json_atomic(self.protocol_path, meta)
            if self.protocol_in_progress_path.exists():
                self.protocol_in_progress_path.unlink()
        except Exception: pass

class KuroCamera:
    def __init__(self):
        self.cam = None
        self.is_connected = False
        # Absolute coordinates of current ROI for sub-selection
        self.offset_x = 0
        self.offset_y = 0

    def connect(self):
        if self.is_connected and self.cam is not None:
            print("[Kuro Camera] Already connected.")
            return

        try:
            import os
            from pylablib import par
            picam_path = r"C:\Program Files\Princeton Instruments\PICam\Runtime"
            par['devices/dlls/picam'] = picam_path
            original_dir = os.getcwd()
            os.chdir(picam_path)
            try:
                from pylablib.devices import PrincetonInstruments
                self.cam = PrincetonInstruments.PicamCamera()
                print(f"[Kuro Camera] Connected successfully.")
                self.is_connected = True
            finally:
                os.chdir(original_dir)
        except Exception as e:
            print(f"[Kuro Camera] Connection failed: {e}")
            self.is_connected = False

    def get_temperature(self):
        if not self.is_connected: return None
        try: 
            # FIXED: Added the exact spaces discovered by the debug script
            return self.cam.get_attribute_value("Sensor Temperature Reading")
        except: 
            return None

    def apply_settings(self, settings: dict):
        if not self.is_connected: return
        try:
            was_running = self.cam.acquisition_in_progress()
            if was_running: self.cam.stop_acquisition()
            if "fan_off" in settings:
                self.cam.set_attribute_value("Disable Cooling Fan", bool(settings["fan_off"]))
            if "gain" in settings:
                self.cam.set_attribute_value("ADC Analog Gain", settings["gain"])
            if "temp_setpoint" in settings:
                self.cam.set_attribute_value("Sensor Temperature Set Point", float(settings["temp_setpoint"]))
            if "readout_speed" in settings:
                self.cam.set_attribute_value("ADC Speed", float(settings["readout_speed"]))
            if was_running: self.start_sequence()
        except Exception as e: print(f"[Kuro Camera] Settings error: {e}")

    def set_exposure(self, ms):
        if not self.is_connected: return
        try: self.cam.set_exposure(float(ms) / 1000.0)
        except Exception as e: print(f"[Kuro Camera] Exposure error: {e}")

    def start_sequence(self):
        if not self.is_connected: return False
        try:
            if self.cam.acquisition_in_progress():
                self.cam.stop_acquisition()
            self.cam.clear_acquisition()
            
            # CRITICAL: This line must be present to allocate the 100-frame buffer
            self.cam.setup_acquisition(mode="sequence", nframes=100) 
            
            self.cam.start_acquisition()
            return True
        except Exception as e:
            print(f"[Kuro Camera] FATAL ERROR starting sequence: {e}")
            return False

    def stop_sequence(self):
        if not self.is_connected: return
        try:
            if self.cam.acquisition_in_progress():
                self.cam.stop_acquisition()
            self.cam.clear_acquisition()
        except Exception: pass

    def pop_images(self, max_count=20):
        """Standardized version of pop_images with Error 27 resilience."""
        if not self.is_connected or self.cam is None: 
            return []
        try:
            # Only poll if hardware is actually running to prevent Error 27
            if not self.cam.acquisition_in_progress():
                return []
                
            frames = self.cam.read_multiple_images(return_info=False)
            if frames and len(frames) > max_count:
                frames = frames[-max_count:] 
            return frames
        except Exception as e:
            # Ignore Error 27 (Acquisition Not In Progress) during hardware swaps
            if "AcquisitionNotInProgress" in str(e) or "error 27" in str(e).lower():
                return [] 
            raise RuntimeError(f"Kuro dropped connection: {e}")

    def set_roi(self, x, y, w, h):
        if not self.is_connected: return
        try:
            was_running = self.cam.acquisition_in_progress()
            if was_running: self.cam.stop_acquisition()
            self.cam.set_roi(int(x), int(x + w), int(y), int(y + h))
            self.offset_x, self.offset_y = int(x), int(y)
            if was_running: self.start_sequence()
        except Exception as e: print(f"[Kuro Camera] ROI error: {e}")

    def reset_roi(self):
        if not self.is_connected: return
        try:
            was_running = self.cam.acquisition_in_progress()
            if was_running: self.cam.stop_acquisition()
            self.cam.set_roi() # Reset to full sensor
            self.offset_x, self.offset_y = 0, 0
            if was_running: self.start_sequence()
        except Exception as e: print(f"[Kuro Camera] Reset ROI error: {e}")

    def close(self):
        if self.is_connected and self.cam:
            self.stop_sequence()
            self.cam.close()