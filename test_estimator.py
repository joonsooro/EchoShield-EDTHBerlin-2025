import numpy as np
from edge_bearing.estimator import estimate_bearing

fs = 16000
duration = 0.5
t = np.linspace(0, duration, int(fs * duration), endpoint=False)
freq = 300
delay_samples = 3

ch0 = np.sin(2 * np.pi * freq * t)
ch1 = np.roll(ch0, delay_samples)

audio = np.stack([ch0, ch1], axis=1)

bearing, conf, state = estimate_bearing(audio, fs, {"orientation_deg": 0.0})
print("bearing:", bearing, "conf:", conf)
