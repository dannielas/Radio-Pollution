import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf
from scipy.signal import spectrogram
from matplotlib.colors import Normalize 

# Load the WAV RF64 file
file_path = 'C:/Users/jakym/gqrx_20240724_173541_156700000.wav'
#file_path = "C:/Users/jakym/Downloads/sdrsharp-x86/IQ/2024_07_17/11-02-56_96400000Hz.wav"
data, samplerate = sf.read(file_path, always_2d=True)

# Extract I/Q data
I = data[:, 0]
Q = data[:, 1]
IQ = I + 1j * Q

#identify and ignore zeros
non_zero_indices = np.where(IQ != 0)
IQ_non_zero = IQ[non_zero_indices]

#identify and ignore zeros and negative values
#valid_indices = np.where((np.abs(IQ) > 0) & (I >= 0) & (Q >= 0))
#IQ_valid = IQ[valid_indices]

# Compute the spectrogram (waterfall display) #IQ_valid
f, t, Sxx = spectrogram(IQ_non_zero, fs=samplerate, nperseg=1024, noverlap=512)

# Convert to dB
Sxx_dB = 10 * np.log10(Sxx) 

#Define normalization with reversed color scale
norm = Normalize(vmin=Sxx_dB.max(), vmax=Sxx_dB.min())

#Replace -inf values with NaN
#Sxx_dB[Sxx_dB == -np.inf] = np.nan

# Plot the waterfall display
plt.figure(figsize=(10, 6))
plt.imshow(Sxx_dB, aspect='auto', extent=[t.min(), t.max(), f.min(), f.max()], cmap='jet', origin='lower', norm=norm)
plt.colorbar(label='Power (dB)')
plt.xlabel('Time (s)')
plt.ylabel('Frequency (Hz)')
plt.title('Waterfall Display')
plt.show()
