import librosa, numpy as np
from scipy.signal import find_peaks

from pathlib import Path

def process_song(song_path):
    """
    Processes a song to extract spectral flux from low, mid, and high frequency bands.

    Args:
        song_path (str or Path): The path to the audio file.

    Returns:
        dict: A dictionary containing the processed data:
              'flux_low': Spectral flux for the low band.
              'flux_mid': Spectral flux for the mid band.
              'flux_high': Spectral flux for the high band.
              'times': Timestamps for each frame.
              'audio_start_offset': The detected start time of the audio in seconds.
    """
    y, sr = librosa.load(song_path, sr=None)
    S = np.abs(librosa.stft(y, n_fft=2048, hop_length=512))

    # Helper function to remove silence from the beginning
    def detect_audio_start(y, sr, threshold=0.01, frame_length=2048, hop_length=512):
        """Detect where audio actually starts by finding first frame above threshold"""
        rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]
        
        start_frame = 0
        for i, val in enumerate(rms):
            if val > threshold:
                start_frame = i
                break
        
        start_sample = librosa.frames_to_samples(start_frame, hop_length=hop_length)
        audio_start_offset = start_sample / sr
        return start_sample, audio_start_offset

    start_sample, audio_start_offset = detect_audio_start(y, sr)
    y = y[start_sample:]
    S = S[:, start_sample // 512:]

    print(f"Trimmed {audio_start_offset:.2f} seconds of silence from the beginning")

    freqs = librosa.fft_frequencies(sr=sr, n_fft=2048)

    def band_energy(S, freqs, f_low, f_high):
        mask = (freqs >= f_low) & (freqs < f_high)
        return S[mask].mean(axis=0)

    low  = band_energy(S, freqs, 30, 120)
    mid  = band_energy(S, freqs, 120, 2000)
    high = band_energy(S, freqs, 2000, 12000)

    # normalize & smooth
    def smooth(x, alpha=0.05):
        y = np.zeros_like(x)
        if len(x) > 1:
            for i in range(1, len(x)):
                y[i] = alpha*x[i] + (1-alpha)*y[i-1]
        return y

    low, mid, high = map(smooth, [low, mid, high])

    # identify “hits”
    flux_low  = np.clip(np.diff(np.r_[low[0], low]) if len(low) > 0 else [], 0, None)
    flux_mid  = np.clip(np.diff(np.r_[mid[0], mid]) if len(mid) > 0 else [], 0, None)
    flux_high = np.clip(np.diff(np.r_[high[0], high]) if len(high) > 0 else [], 0, None)

    times = librosa.frames_to_time(np.arange(len(low)), sr=sr, hop_length=512)

    return {
        'flux_low': flux_low,
        'flux_mid': flux_mid,
        'flux_high': flux_high,
        'times': times,
        'audio_start_offset': audio_start_offset
    }

def detect_kicks(processed_song, bpm, kick_percentile_threshold=95, video_switch_frequency=4):
    """
    Detects kick drum hits from processed song data.

    Args:
        processed_song (dict): The dictionary returned by process_song().
        bpm (float): The beats per minute of the song.
        kick_percentile_threshold (float, optional): The percentile of flux values to use as a
                                                     height threshold for peak detection. Defaults to 95.
        video_switch_frequency (int, optional): A divisor for the beat to set the minimum
                                                distance between peaks (e.g., 4 for 16th notes).
                                                Defaults to 4.

    Returns:
        tuple: A tuple containing:
            - np.ndarray: Timestamps of the detected kicks.
            - np.ndarray: A binary array indicating kick presence at each frame.
    """
    flux_low = processed_song['flux_low']
    times = processed_song['times']
    sr = processed_song['sr']

    # Detect kicks using peak detection on low-frequency flux
    peak_indices, _ = find_peaks(
        flux_low,
        height=np.percentile(flux_low, kick_percentile_threshold),
        distance=int(sr / 512 * (60 / bpm / video_switch_frequency))  # Minimum note spacing based on BPM
    )

    # Create binary array (1 = kick detected, 0 = no kick)
    kick_binary = np.zeros(len(flux_low))
    kick_binary[peak_indices] = 1

    # Get timestamps for kicks
    kick_times = times[peak_indices]

    print(f"Detected {len(kick_times)} kicks/808 hits")

    return kick_times, kick_binary