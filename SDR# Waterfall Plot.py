import numpy as np
import matplotlib.pyplot as plt
from rtlsdr import RtlSdr
import matplotlib.animation as animation
import time

# Function to compute the FFT and get the power spectrum
def get_spectrum(sdr, fft_size=256):
    samples = sdr.read_samples(fft_size)
    window = np.hanning(len(samples))  # Apply a Hanning window
    samples_windowed = samples * window
    spectrum = np.fft.fftshift(np.fft.fft(samples_windowed, n=fft_size))
    power = np.abs(spectrum) ** 2
    return 10 * np.log10(power)

# Set up the SDR
sdr = RtlSdr()
sdr.sample_rate = 2.5e6  # Hz OG:2.048E6
sdr.center_freq = 96.5e6    # Hz
sdr.freq_correction = 60   # PPM
sdr.gain = 'auto'

# Parameters for the waterfall plot
fft_size = 3000
time_duration = 10  # Duration in seconds for the waterfall plot

# Initialize data storage
waterfall_data = np.zeros((0, fft_size))

# Start time for the plot
start_time = time.time()

# Function to update the waterfall plot
def update_plot(frame):
    global waterfall_data
    spectrum = get_spectrum(sdr, fft_size=fft_size)
    waterfall_data = np.vstack((waterfall_data, spectrum[np.newaxis, :]))
    
    # Only keep the last 'time_duration' seconds of data
    elapsed_time = time.time() - start_time
    if elapsed_time > time_duration:
        waterfall_data = waterfall_data[-int(sdr.sample_rate * time_duration / fft_size):, :]
    
    ax.clear()
    ax.imshow(waterfall_data, aspect='auto', cmap='turbo',
              extent=[-sdr.sample_rate / (2*1e6), sdr.sample_rate / (2*1e6),
                      elapsed_time, 0])
    ax.set_title("Waterfall Plot of FM Radio Band with FFT")
    ax.set_ylabel("Time (s)")
    ax.set_xlabel("Δ Frequency (MHz)")

# Set up the plot
fig, ax = plt.subplots(figsize=(10, 6))
ani = animation.FuncAnimation(fig, update_plot, interval=100)

# Display the plot
plt.show()

# Close the SDR when done
sdr.close()