import os
import librosa
import librosa.display
import numpy as np
import pandas as pd
from scipy.stats import skew
import math 
from scipy.stats import kurtosis

def spectral_skew(y, sr, n_fft=2048, hop_length=512):
    S, _ = librosa.core.stft(y, n_fft=n_fft, hop_length=hop_length)
    S = librosa.util.normalize(S)
    M2 = np.sum(S**2 * np.arange(S.shape[1]), axis=1)
    M3 = np.sum(S**3 * (np.arange(S.shape[1]) - np.mean(np.arange(S.shape[1]))), axis=1)
    spectral_skew = M3 / (np.sqrt(M2) ** 3)
    return spectral_skew

def _feature_extraction(sound_file, start, end, selec, bp, wl, threshold):
    audio, sr = librosa.load(sound_file)
    if selec is not None:
        audio = audio[int(sr * start) : int(sr * end)]
    stft = np.abs(librosa.stft(audio))
    power_spec = librosa.amplitude_to_db(stft, ref=np.max)
    frequencies = librosa.core.fft_frequencies(sr=sr)
    mean_freq = np.sum(frequencies * power_spec.T) / np.sum(power_spec)
    mean_freq = mean_freq / 1000  # Convert to kHz
    frequencies = librosa.fft_frequencies(sr=sr, n_fft=wl)
    frequency_sd = np.std(frequencies)
    spectral_centroids = librosa.feature.spectral_centroid(y=audio, sr=sr)
    median_frequency = librosa.hz_to_mel(librosa.median(spectral_centroids)) / 1000
    mel_spec = librosa.feature.melspectrogram(y=audio, sr=sr)
    mel_spec_db = librosa.power_to_db(mel_spec)
    q25 = np.percentile(mel_spec_db, 25)
    q75 = np.percentile(mel_spec_db, 75)
    iqr = q75 - q25
    freqs = librosa.mel_frequencies(n_mels=mel_spec.shape[0], fmin=0, fmax=sr/2)
    freqs_khz = freqs / 1000
    q25_khz = np.interp(q25, mel_spec_db.min(), mel_spec_db.max(), freqs_khz.min(), freqs_khz.max())
    q75_khz = np.interp(q75, mel_spec_db.min(), mel_spec_db.max(), freqs_khz.min(), freqs_khz.max())
    iqr_khz = q75_khz - q25_khz
    skew = spectral_skew(y, sr)
    kurtosis_librosa = librosa.feature.spectral_contrast(audio)[0]
    kurtosis_scipy = kurtosis(kurtosis_librosa)
    stft_norm = stft / np.sum(stft)
    entropy = -np.sum(stft_norm * np.log2(stft_norm))
    gmean = librosa.geometric_mean(stft, axis=0)
    amean = np.mean(stft, axis=0)
    sfm = 10 * np.log10(gmean / amean)
    freq_mode, count_mode = stats.mode(librosa.fft_frequencies(sr=sr, n_fft=wl))
    peak_freq_bin_index = np.argmax(stft)
    fft_freqs = librosa.fft_frequencies(sr)
    peak_freq = fft_freqs[peak_freq_bin_index]
    f0 = librosa.yin(y=audio, sr=sr)
    mean_f0 = np.mean(f0)
    min_f0 = np.min(f0)
    max_f0 = np.max(f0)
    dominant_freq = librosa.feature.spectral_centroid(y=power_spec, sr=sr)
    meandom = np.mean(dominant_freq)
    mindom = np.min(dominant_freq)
    maxdom = np.max(dominant_freq)
    dfrange = np.ptp(dominant_freq)
    delta_f0 = np.abs(np.diff(f0))
    f0_range = np.max(f0) - np.min(f0)
    modindx = np.sum(delta_f0) / f0_range
    features = {
        "duration": duration,
        "meanfreq": mean_freq,
        "sd": frequency_sd,
        "median": median_frequency,
        "Q25": q25_khz,
        "Q75": q75_khz,
        "IQR": iqr_khz,
        "skew": skew,
        "kurt": kurtosis_scipy,
        "sp.ent":entropy,
        "sfm":sfm,
        "mode": freq_mode[0],
        "centroid":mean_freq,
        "peakf":peak_freq,
        "meanfun":mean_f0,
        "minfun":min_f0,
        "maxfun":max_f0,
        "meandom":meandom,
        "mindom":mindom,
        "maxdom":maxdom,
        "dfrange":dfrange,
        "modindx":modindx
    }
    print(features)
    return features

def specan3(X, bp=(0, 22), wl=2048, threshold=5, parallel=1):

    if isinstance(X, pd.DataFrame):
        if not all(col in X for col in ("sound.files", "selec", "start", "end")):
            raise ValueError(
                f"Missing required columns in data frame: {','.join(('sound.files', 'selec', 'start', 'end'))}"
            )
        start = np.array(X["start"])
        end = np.array(X["end"])
        sound_files = np.array(X["sound.files"])
        selec = np.array(X["selec"])
    else:
        raise ValueError("X must be a data frame")

    # Check for NaN values and invalid timings
    if np.any(np.isnan(np.concatenate((end, start)))):
        raise ValueError("NaN values found in start and/or end")
    if np.any(end - start < 0):
        raise ValueError("Start time greater than end time in some selections")
    if np.any(end - start > 20):
        raise ValueError("Some selections longer than 20 seconds")

    # Check bandpass filter and parallel processing
    if not isinstance(bp, tuple) or len(bp) != 2:
        raise ValueError("bp must be a numeric vector of length 2")
    if not isinstance(parallel, int) or parallel < 1:
        raise ValueError("parallel must be a positive integer")

    # Select sound files and apply parallel processing
    features = [
        _feature_extraction(f, s, e, sl, bp, wl, threshold)
        for f, s, e, sl in zip(sound_files, start, end, selec)
    ]

    # Combine features into data frame and rename columns
    df = pd.DataFrame(features).transpose()
    df.columns = ["sound.files", "selec"] + features[0].keys()
    df.set_index("selec", inplace=True)
    df.index.names = ["selection"]

    return df

def Extract(audio_path, wd):
    print('Extract')

    os.chdir(wd)  # Change working directory

    # Create data frame with audio file path and selection information
    data = pd.DataFrame({
        "sound.files": [audio_path],
        "selec": [None],  # Process the whole file
        "start": [0],
        "end": [20]  # Process the first 20 seconds
    })

    # Extract features
    acoustics = specan3(data)

    # Save features to CSV
    acoustics.to_csv("Features.csv", index=True)