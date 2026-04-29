from pathlib import Path

# --- Device Names ---
DEVICE_FILTER_BLOCK = "TIFilterBlock1"
DEVICE_WL_LAMP_CANDIDATES = ["DiaLamp", "TLLamp", "LED", "Dia"] 

# Property Names
PROP_STATE = "State"
PROP_INTENSITY = "Intensity"
PROP_COMPUTER_CONTROL = "ComputerControl"

# --- Defaults ---
DEFAULT_SAVE_DIR = Path("D:/Nathan/arabidopsis")
CHANNEL_GROUP = "Channel"

DEFAULTS = {
    "smlm_exposure_ms": 30.0,
    "wl_exposure_ms": 50.0,
    "photoactivation_ms": 100.0,
    "target_frames_normal": 1000,
    "smlm_frames_per_cycle": 1000,
    "wl_frames_per_cycle": 10,
    "num_cycles": 10,
    "wl_intensity": 4,
    "preview_max_fps": 15,
    "save_folder_name": "experiment",
    "save_experiment_name": "experiment_01",
    "save_comment": "",
    "interleaved_pause_ms": 1500.0,
    "preview_fps_throttle": 0.5,
    "pop_timer_ms": 5,
    "max_pop_per_tick": 20, # Increased slightly
    "n_frames": 2000,
    "laser_ma": 20.0,  # (ADD THIS LINE)
    "camera_offset_adu": 100.0,
    "camera_gain_adu_per_photon": 1.4,
}

# Hard safety limit for UV laser current (mA)
LASER_MAX_MA = 45.0

# --- Laser Settings --- (ADD THIS SECTION)
LASER_SERIAL = "64850466"
KINESIS_PATH = r"C:\Program Files\Thorlabs\Kinesis"