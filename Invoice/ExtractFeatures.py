import os
import librosa
import librosa.display
import numpy as np
import pandas as pd


def create_bandpass_mask(freqs, sr=22050, n_fft=2048):
    # Calculate filter frequencies in bins
    fmin_bin = int(np.round(freqs[0] * n_fft / sr))
    fmax_bin = int(np.round(freqs[1] * n_fft / sr)) + 1

    # Create mask
    mask = np.zeros(n_fft // 2 + 1, dtype=bool)
    mask[fmin_bin:fmax_bin] = True
    return mask

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
    mask = create_bandpass_mask([fmin,fmax],sr=sr, n_fft=wl)
    if len(S.shape) == 2:
        S = S[:, mask]
    else:
        # Handle 1D spectrogram case (adapt as needed)
        S = S[mask]  # Reshape or implement alternative logic

    # Feature extraction
    analysis = librosa.feature.spectral_centroid(S=S, sr=sr)
    mean_freq = analysis.mean() / 1000
    sd = analysis.std() / 1000
    median = np.median(analysis) / 1000
    q25 = np.percentile(analysis, 25) / 1000
    q75 = np.percentile(analysis, 75) / 1000
    iqr = q75 - q25
    skew = librosa.feature.spectral_skewness(S=S, sr=sr)
    kurt = librosa.feature.spectral_kurtosis(S=S, sr=sr)
    spectral_entropy = librosa.feature.spectral_entropy(S=S, sr=sr)
    sfm = librosa.feature.spectral_flatness(S=S)
    mode = librosa.feature.mfcc(S=S, sr=sr)[1].mean() / 1000
    centroid = librosa.feature.spectral_centroid(S=S, sr=sr).mean() / 1000

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