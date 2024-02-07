import os
import librosa
import librosa.display
import numpy as np
import pandas as pd
from scipy.stats import skew
import math 

def freq_mask(P, sr, fmin, fmax, invert=True):
    N_FFT = P.shape[0]
    f_bins = np.linspace(0, sr / 2, int(N_FFT // 2 + 1))

    mask = (f_bins >= fmin) & (f_bins <= fmax)
    if invert:
        mask = ~mask

    return mask

def spectral_skew(data):
    mean = np.mean(data)
    variance = np.mean((data - mean)**2)
    skew_value = skew((data - mean) / np.sqrt(variance))
    return skew_value

def spectral_kurtosis(data):
    mean = np.mean(data)
    variance = np.mean((data - mean)**2)
    kurtosis_value = np.mean((data - mean)**4 / (variance**2))
    return kurtosis_value

def spectral_entropy(S):
    S_norm = S / np.sum(S, axis=0, keepdims=True)
    entropy = -np.sum(S_norm * np.log2(S_norm + np.finfo(float).eps), axis=0)
    return entropy

def _feature_extraction(sound_file, start, end, selec, bp, wl, threshold):
    # Load audio data
    y, sr = librosa.load(sound_file, sr=None)

    # Select specific portion if selec is not None
    if selec is not None:
        y = y[int(sr * start) : int(sr * end)]

    # Calculate spectrogram
    S, P, n_fft, t, *rest = librosa.stft(y, n_fft=wl)

    # Apply bandpass filter
    fmin, fmax = bp
    mask = freq_mask(P, sr, fmin, fmax, invert=True)
    num_rows_S = S.shape[0]
    S = S * mask[:, np.newaxis]
    S=-S.astype(np.float32)
    # Find negative values
    negative_indices = np.where(S < 0)

    # Option 1: Set negative values to small positive value (replace with desired approach)
    S[negative_indices] = 1e-8
    # Feature extraction
    analysis = librosa.feature.spectral_centroid(S=S, sr=sr)[0]
    centroid = analysis.mean() / 1000
    mean_freq = centroid 
    sd = analysis.std() / 1000
    median = np.median(analysis) / 1000
    q25 = np.percentile(analysis, 25) / 1000
    q75 = np.percentile(analysis, 75) / 1000
    iqr = q75 - q25
    skew = spectral_skew(analysis)
    kurt = spectral_kurtosis(analysis)
    spectral_entropy = -math.log2(1 / len(centroid))
    sfm = librosa.feature.spectral_flatness(S=S)
    mode = librosa.feature.mfcc(S=S, sr=sr)[1].mean() / 1000

    # Peak frequency
    peak_freq = 0
    # TODO: Implement peak frequency calculation using librosa functions

    # Fundamental frequency parameters
    f0 = librosa.core.estimate_tuning(y=y, sr=sr, threshold=threshold)[0]
    mean_f0 = np.mean(f0) if f0.size else np.nan
    min_f0 = np.min(f0) if f0.size else np.nan
    max_f0 = np.max(f0) if f0.size else np.nan

    # Dominant frequency parameters
    dfreq = librosa.feature.delta_mfcc(
        S=S, sr=sr, win_length=wl, window="hann", delta=2
    )
    mean_dfreq = np.mean(dfreq, axis=1).mean()
    min_dfreq = np.min(dfreq, axis=1).min()
    max_dfreq = np.max(dfreq, axis=1).max()
    df_range = max_dfreq - min_dfreq

    # Duration
    duration = end - start

    # Modulation index
    changes = []
    for i in range(len(dfreq) - 1):
        change = abs(dfreq[i] - dfreq[i + 1])
        changes.append(change)
    mod_index = np.mean(changes) / df_range if df_range else 0

    # Combine features into dictionary
    features = {
        "duration": duration,
        "meanfreq": mean_freq,
        "sd": sd,
        "median": median,
        "Q25": q25,
        "Q75": q75,
        "IQR": iqr,
        "skew": skew,
        "kurt": kurt,
        "sp.ent":spectral_entropy,
        "sfm":sfm,
        "mode": mode,
        "centroid":centroid,
        "peakf":peak_freq,
        "meanfun":mean_f0,
        "minfun":min_f0,
        "maxfun":max_f0,
        "meandom":mean_dfreq,
        "mindom":min_dfreq,
        "maxdom":max_dfreq,
        "dfrange":df_range,
        "modindx":mod_index
    }
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
    """
    Extracts acoustic features from an audio file and saves them to a CSV file.

    Args:
        audio_path: Path to the audio file (.wav)
        wd: Working directory to save the features
    """

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