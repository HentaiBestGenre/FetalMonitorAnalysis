import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf
from scipy.signal import butter, sosfilt, find_peaks
from io import BytesIO
import base64
from typing import Union


FILEPATH = "media/фетальник_больничный.wav"

def normalize_audio(audio):
    """Normalize audio signal to [-1, 1]."""
    return audio / np.max(np.abs(audio))

def bandpass_filter(data, lowcut, highcut, sample_rate, order=2):
    """Apply a bandpass filter using second-order sections for stability."""
    nyquist = 0.5 * sample_rate
    low = lowcut / nyquist
    high = highcut / nyquist
    sos = butter(order, [low, high], btype='band', output='sos')
    y = sosfilt(sos, data)
    return y

def plot_audio_signals(filtered, sample_rate):
    """Plot filtered audio signal."""
    times = np.linspace(0, len(filtered) / sample_rate, num=len(filtered))

    plt.figure(figsize=(18, 9))
    plt.plot(times, filtered)
    plt.title('Filtered Audio')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')

    plt.tight_layout()
    buf = BytesIO()
    plt.savefig(buf, format='png')
    plt.close()
    buf.seek(0)
    return buf

def plot_r_r_signals(times, filtered_audio, peaks):
    """Plot R-R intervals."""
    plt.figure(figsize=(10, 4))
    plt.plot(times, filtered_audio, label='Filtered signal')
    plt.plot(times[peaks], filtered_audio[peaks], 'x', label='Detected peaks')
    plt.title('Detected R-peaks in the signal')
    plt.legend()
    plt.xlabel('Time (seconds)')
    plt.ylabel('Amplitude')

    plt.show()

def get_audio_details(r_peak_times):
    """Calculate audio details."""
    rr_intervals = np.diff(r_peak_times)
    bpm = 60 / np.mean(rr_intervals)
    ibi = np.mean(rr_intervals)
    sdnn = np.std(rr_intervals)
    sdsd = np.std(np.diff(rr_intervals))
    rmssd = np.sqrt(np.mean(np.diff(rr_intervals)**2))
    nn20 = np.sum(np.abs(np.diff(rr_intervals)) > 0.02)
    pnn20 = nn20 / len(rr_intervals) * 100
    nn50 = np.sum(np.abs(np.diff(rr_intervals)) > 0.05)
    pnn50 = nn50 / len(rr_intervals) * 100
    hr_mad = np.median(np.abs(rr_intervals - np.median(rr_intervals)))
    sd1 = np.sqrt(np.std(rr_intervals - np.mean(rr_intervals))**2 * 0.5)
    sd2 = np.sqrt(2 * np.std(rr_intervals)**2 - sd1**2)
    s = np.pi * sd1 * sd2
    sd1_sd2_ratio = sd1 / sd2
    breathing_rate = 60 / (3 * np.median(rr_intervals))
    
    data = {
        "BPM": round(bpm, 2),
        "IBI": round(ibi, 2),
        "SDNN": round(sdnn, 2),
        "SDSD": round(sdsd, 2),
        "RMSSD": round(rmssd, 2),
        "pNN20": round(pnn20, 2),
        "pNN50": round(pnn50, 2),
        "HR_MAD": round(hr_mad, 2),
        "SD1": round(sd1, 2),
        "SD2": round(sd2, 2),
        "S": round(s, 2),
        "SD1_SD2_Ratio": round(sd1_sd2_ratio, 2),
        "Breathing Rate": round(breathing_rate, 2)
    }
    return data

def handle_base64_audio(file: Union[str, BytesIO]):
    """Decode base64 string and load audio."""
    if isinstance(file, str):
        try:
            audio, sample_rate = sf.read(file)
        except (FileNotFoundError, RuntimeError):
            # If not a path, treat it as a base64-encoded string
            file = BytesIO(base64.b64decode(file))
            audio, sample_rate = sf.read(file)
    # audio_data = base64_string
    audio, sample_rate = sf.read(file)
    if audio.ndim > 1:
        audio = audio[:, 0]
    return audio, sample_rate

def process_audio(base64_string: Union[str, BytesIO]):
    """Process the audio data."""
    audio, sample_rate = handle_base64_audio(base64_string)
    # normalized_audio = normalize_audio(audio)
    filtered_audio = bandpass_filter(audio, 20, 150, sample_rate)
    return filtered_audio, sample_rate

def create_plot_img(base64_string):
    """Create a plot image of the filtered audio."""
    filtered_audio, sample_rate = process_audio(base64_string)
    plot_buffer = plot_audio_signals(filtered_audio, sample_rate)
    return plot_buffer

def create_rr_plot_img(base64_string: Union[str, BytesIO]):
    """Create a plot image of R-R intervals."""
    filtered_audio, sample_rate = process_audio(base64_string)
    times = np.arange(filtered_audio.size) / sample_rate
    peaks, _ = find_peaks(filtered_audio, height=np.mean(filtered_audio), distance=sample_rate/4)
    rr_plot_buffer = plot_r_r_signals(times, filtered_audio, peaks)
    return rr_plot_buffer

def fetal_analysis(base64_string):
    """Perform detailed analysis on the audio."""
    filtered_audio, sample_rate = process_audio(base64_string)
    times = np.arange(filtered_audio.size) / sample_rate
    peaks, _ = find_peaks(filtered_audio, height=np.mean(filtered_audio), distance=sample_rate/4)
    r_peak_times = times[peaks]
    info = get_audio_details(r_peak_times)
    return info


# Create RR Plot
create_rr_plot_img(FILEPATH)

filtered_audio, sample_rate = fetal_analysis(FILEPATH)
times = np.arange(filtered_audio.size) / sample_rate
peaks, _ = find_peaks(filtered_audio, height=np.mean(filtered_audio), distance=sample_rate/4)
r_peak_times = times[peaks]

r_peak_times_clear = r_peak_times - r_peak_times[0]

f_time = r_peak_times[0]
last_time = r_peak_times[-1]
timing = last_time - f_time

valid_r_peak_times = r_peak_times_clear[1:]

# Compute BPM values excluding the first invalid point
bpm_fixed = [60 * (i + 1) / valid_r_peak_times[i] for i in range(len(valid_r_peak_times))]


# BPM
np.mean(bpm_fixed)

# Variability BPM
rr_intervals = np.diff(valid_r_peak_times)
hr_values = 60 / rr_intervals

# Calculate SDNN in BPM
sdnn_bpm = np.std(hr_values)

# Calculate RMSSD in BPM
rmssd_bpm = np.sqrt(np.mean(np.diff(hr_values) ** 2))


# Amplitude of HR variability
amplitude_bpm = np.max(bpm_fixed) - np.min(bpm_fixed)
