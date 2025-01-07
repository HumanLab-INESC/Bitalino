import math
import sys
import pandas as pd
from biosppy.signals import eeg
from biosppy import plotting


def load_signal(fname, sampling_rate=1000, t_a=None, t_b=None):
    """
    Load the signal from a file and optionally slice it based on time.
    Parameters:
        fname (str): Path to the input file. 
                     For exmaple: "opensignals_98D391FD7A28_2021-05-30_15-02-09.txt"
        sampling_rate (int): Sampling rate of the device (default is 1000 Hz).
        t_a (float, optional): Start time in seconds (inclusive).
        t_b (float, optional): End time in seconds (exclusive).
    Returns:
        tuple: A tuple containing:
            - signal (numpy.ndarray): The loaded signal data as a NumPy array.
            - labels (list): List of column labels.
    """
    print("Loading file...")
    cols = ["nSeq", "I1", "I2", "O1", "O2", "A1", "A2", "A3", "A4", "A5", "A6"]
    df = pd.read_csv(fname, sep='\t', index_col=False, names=cols, skiprows=3)

    signal = df.to_numpy()
    labels = df.columns.tolist()

    if t_a is not None and t_b is not None:
        start_idx = int(t_a * sampling_rate)
        end_idx = int(t_b * sampling_rate)
        signal = signal[start_idx:end_idx, :]

    return signal, labels


def compute_frequency_bands(signal, sampling_rate, labels):
    """
    Compute EEG frequency bands and features using Biosppy.
    Parameters:
        signal (numpy.ndarray): The EEG signal data.
        sampling_rate (int): The sampling rate of the device.
        labels (list): List of column labels.
    Returns:
        dict: A dictionary containing frequency bands and other EEG features.
    """
    print("Computing frequency bands...")
    ts, filtered, features_ts, theta, alpha_low, alpha_high, beta, gamma, plf_pairs, plf = eeg.eeg(
        signal=signal,
        sampling_rate=sampling_rate,
        labels=labels,
        show=False
    )

    return {
        "ts": ts,
        "filtered": filtered,
        "features_ts": features_ts,
        "theta": theta,
        "alpha_low": alpha_low,
        "alpha_high": alpha_high,
        "beta": beta,
        "gamma": gamma,
        "plf_pairs": plf_pairs,
        "plf": plf,
    }


def plot_and_save_bands(features, raw_signal, labels, output_path='figs/plot.png'):
    """
    Plot and save EEG frequency bands.
    Parameters:
        features (dict): Dictionary of frequency bands and EEG features.
        raw_signal (numpy.ndarray): The raw EEG signal.
        labels (list): List of column labels.
        output_path (str): Path to save the plot (default is 'figs/plot.png').
    """
    print("Plotting and saving frequency bands...")
    plotting.plot_eeg(
        ts=features["ts"],
        raw=raw_signal,
        filtered=features["filtered"],
        labels=labels,
        features_ts=features["features_ts"],
        theta=features["theta"],
        alpha_low=features["alpha_low"],
        alpha_high=features["alpha_high"],
        beta=features["beta"],
        gamma=features["gamma"],
        plf_pairs=features["plf_pairs"],
        plf=features["plf"],
        path=output_path,
        show=False
    )


def compute_emotional_features(alpha_low, alpha_high, signal):
    """
    Compute emotional features (valence, arousal, and ratio) based on EEG bands.
    Parameters:
        alpha_low (numpy.ndarray): Low alpha frequency band.
        alpha_high (numpy.ndarray): High alpha frequency band.
        signal (numpy.ndarray): The EEG signal data.
    Returns:
        tuple: A tuple containing valence, arousal, and ratio.
    """
    print("Computing emotional features...")
    valence = 0.5 * alpha_low[:, 0].mean() + 0.5 * alpha_high[:, 0].mean()
    arousal = signal[:, 1].mean()
    ratio = math.degrees(math.atan(arousal / valence))

    return valence, arousal, ratio


def plot_frequency_bands(fname, t_a=None, t_b=None, sampling_rate=1000):
    """
    Load an EEG signal, compute frequency bands, plot them, and compute emotional features.
    Parameters:
        fname (str): Path to the input file.
        t_a (float, optional): Start time in seconds for slicing the signal.
        t_b (float, optional): End time in seconds for slicing the signal.
        sampling_rate (int): Sampling rate of the device (default is 1000 Hz).
    Returns:
        dict: A dictionary containing frequency bands and emotional features.
    """

    # Step 1. Load signals
    signal, labels = load_signal(fname, sampling_rate, t_a, t_b)

    # Step 2. Compute frequency bands
    features = compute_frequency_bands(signal, sampling_rate, labels)

    # Step 3. Plot and save stuff in figs/
    plot_and_save_bands(features, signal, labels, output_path='figs/plot.png')

    # Step 4. Compute valence and arousal using formula from Sandra's paper
    valence, arousal, ratio = compute_emotional_features(features["alpha_low"], features["alpha_high"], signal)

    # Print result and return
    print(f"Valence: {valence}, Arousal: {arousal}, Ratio: {ratio}")

    return {
        "frequency_bands": features,
        "valence": valence,
        "arousal": arousal,
        "ratio": ratio,
    }


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python compute_eeg_bands.py <filename>")
        sys.exit(1)

    fname = sys.argv[1]
    result = plot_frequency_bands(fname)
