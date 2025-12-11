import kagglehub
import os
import pandas as pd
import numpy as np
from scipy.stats import entropy
import nolds
from pyentrp import entropy as pe
import warnings
import pywt
from typing import Tuple, List
from scipy.signal import welch


# FEATURE EXTRACTION UTILITIES

def load_data():
    path = kagglehub.dataset_download("sigfest/database-for-emotion-recognition-system-gameemo")

    frames_raw = []
    frames_processed = []

    # Search every subject's directory for datasets
    for i in range(1, 29):
        base_dir = os.path.join(path, "GAMEEMO", f"(S{i:02})")
        csv_dir = ".csv format"
        raw_dir = os.path.join(base_dir, "Raw EEG Data", csv_dir)
        processed_dir = os.path.join(base_dir, "Preprocessed EEG Data", csv_dir)

        # Search every game CSV file
        for j in range (1, 5):
            csv_raw = os.path.join(raw_dir, f"S{i:02}G{j}AllRawChannels.csv")
            # For some godforsaken reason, CSV files in the same structure do not have consistent names
            if os.path.exists(csv_raw):
                df_raw = pd.read_csv(csv_raw)
            else:
                df_raw = pd.read_csv(os.path.join(raw_dir, f"S{i:02}G{j}AllChannels.csv"))
            df_processed = pd.read_csv(os.path.join(processed_dir, f"S{i:02}G{j}AllChannels.csv"))

            # Append Subject and Game information stored in filesystem to dataset
            df_raw["SUBJECT"] = i
            df_processed["SUBJECT"] = i
            df_raw["GAME"] = j
            df_processed["GAME"] = j
            frames_raw.append(df_raw)
            frames_processed.append(df_processed)

    df_raw = pd.concat(frames_raw)
    df_processed = pd.concat(frames_processed)

    # Clean bad columns
    df_raw = df_raw.dropna(axis=1, how="all")
    df_processed = df_processed.dropna(axis=1, how="all")
    return df_processed

def get_subsignal_and_coefficients(
    raw_signal,
    wavelet_name,
    level,
    band_map,
    reconstruction_bands):
    """
    Decomposes a single channel's signal.
    1. COLLECTS all coefficients (A4, D4, D3, D2, D1).
    2. RECONSTRUCTS the time-series for the four main EEG bands (A4-D2).

    Returns:
        - sub_signals (Dict[str, np.ndarray]): The 4 reconstructed time-series arrays.
        - raw_coeffs (Dict[str, np.ndarray]): The 5 collected coefficient arrays.
    """
    # 1. Perform Multi-level DWT Decomposition
    # coeffs list structure: [A_level, D_level, D_level-1, ..., D_1]
    coeffs = pywt.wavedec(raw_signal, wavelet_name, level=level)

    sub_signals = {}
    raw_coeffs = {}
    original_length = len(raw_signal)

    # --- Part 1: COLLECT RAW COEFFICIENTS (All 5 bands) ---
    for band_name, coeff_index in band_map.items():
        raw_coeffs[band_name] = coeffs[coeff_index]

    # --- Part 2: RECONSTRUCT TIME-SERIES (Only 4 bands: A4, D4, D3, D2) ---
    for band_name in reconstruction_bands:
        coeff_index = band_map[band_name]

        # Create a zero-initialized list of coefficients for reconstruction
        reconstruction_coeffs = [np.zeros_like(c) for c in coeffs]

        # ISOLATE: Copy only the target coefficient array into the structure
        reconstruction_coeffs[coeff_index] = coeffs[coeff_index]

        # RECONSTRUCT: Convert the isolated coefficient structure back into a time-series
        sub_signal = pywt.waverec(reconstruction_coeffs, wavelet_name)

        # TRUNCATE: Truncate the reconstructed signal to match the original length
        sub_signal_truncated = sub_signal[:original_length]

        sub_signals[band_name] = sub_signal_truncated

    return sub_signals, raw_coeffs

def normalize_eeg_segment(eeg_data: dict) -> dict:
    """
    Performs Z-score standardization on a segment of EEG time series data.
    Args:
        eeg_data: A NumPy array of shape (Time_Steps, Channels),
                  containing the raw or pre-processed EEG data for one segment.

    Returns:
        A NumPy array of the same shape, with each column (channel) standardized.
    """

    normalized_eeg_data = {}
    for band, data in eeg_data.items():
      # Check if the array is empty or too short
      if data.size == 0 or data.shape[0] < 2:
          print("Copied data array too short")
          return data.copy()

      # 1. Calculate the mean of each channel (column)
      mean = np.mean(data, axis=0)

      # 2. Calculate the standard deviation of each channel (column)
      std_dev = np.std(data, axis=0) + 1e-6

      # 3. Apply Z-score transformation
      normalized_data = (data - mean) / std_dev
      normalized_eeg_data[band] = normalized_data

    return normalized_eeg_data

def normalize_eeg_segment_array(eeg_data: np.ndarray) -> np.ndarray:
    """Performs Z-score standardization on each channel of the time series data."""
    if eeg_data.size == 0 or eeg_data.shape[0] < 2:
        return eeg_data.copy()
    
    mean = np.mean(eeg_data, axis=0)
    std_dev = np.std(eeg_data, axis=0) + 1e-6 
    
    return (eeg_data - mean) / std_dev

def create_windows(
    data: np.ndarray, 
    sf: int, 
    window_sec: float, 
    overlap_sec: float
) -> List[np.ndarray]:
    """
    Splits the EEG time series data into overlapping or non-overlapping windows.

    Args:
        data: NumPy array of shape (Time_Steps, Channels).
        sf: Sampling frequency (Hz) of the data.
        window_sec: Length of each window in seconds.
        overlap_sec: Overlap between windows in seconds.
        
    Returns:
        A list of NumPy arrays, where each array is one time window.
    """
    n_samples, n_channels = data.shape
    win_samples = int(sf * window_sec)
    step_samples = int(sf * (window_sec - overlap_sec))

    if win_samples <= 0 or step_samples <= 0:
        raise ValueError("Window size and step size must be positive.")
    
    windows = []
    start = 0
    while start + win_samples <= n_samples:
        end = start + win_samples
        window = data[start:end, :]
        windows.append(window)
        start += step_samples
        
    return windows



# =========================================================================
# --- FEATURE EXTRACTION FUNCTIONS ---
# =========================================================================

def feat_std_dev(data: np.ndarray) -> float:
    """Standard Deviation of the signal."""
    return np.std(data)

def feat_zero_crossings(data: np.ndarray) -> float:
    """Number of times the signal crosses zero."""
    # Ensure mean is centered near zero for proper interpretation
    return np.sum(np.diff(np.sign(data)) != 0) / 2 # Divide by 2 as diff counts up/down crossings

def feat_avg_wavelet_energy(coeffs: np.ndarray) -> float:
    """Average Energy of Wavelet Coefficients."""
    if len(coeffs) == 0: return 0.0
    return np.sum(coeffs**2) / len(coeffs)

def feat_log_energy_entropy(coeffs: np.ndarray) -> float:
    """Logarithmic Energy Entropy (requires coefficients)."""
    # Calculate energy
    energy = coeffs**2
    # Add a small epsilon to avoid log(0)
    epsilon = np.finfo(float).eps
    return np.sum(np.log(energy + epsilon))

def feat_shannon_entropy(coeffs: np.ndarray) -> float:
    """Shannon Entropy based on the normalized energy distribution of coefficients."""
    # Calculate energy
    energy = coeffs**2
    total_energy = np.sum(energy)
    if total_energy == 0: return 0.0

    # Calculate probability distribution (normalized energy)
    p_i = energy / total_energy

    # Use scipy.stats.entropy, which handles p_i = 0 safely
    return entropy(p_i, base=2)

def feat_hjorth_activity(data: np.ndarray) -> float:
    """Activity (equivalent to Variance)."""
    return np.var(data)

def feat_hjorth_mobility(data: np.ndarray) -> float:
    """Mobility (mean frequency). Ratio of the standard deviation of the first derivative to the standard deviation of the signal."""
    diff_data = np.diff(data)
    if np.std(data) == 0: return 0.0
    return np.std(diff_data) / np.std(data)

def feat_hjorth_complexity(data: np.ndarray) -> float:
    """Complexity (change in frequency). Ratio of the mobility of the first derivative to the mobility of the signal."""
    mobility = feat_hjorth_mobility(data)
    if mobility == 0: return 0.0

    # Calculate Mobility of the first derivative
    diff_data = np.diff(data)
    mobility_diff = feat_hjorth_mobility(diff_data)

    return mobility_diff / mobility

def feat_sample_entropy(data: np.ndarray, m: int = 2, r_factor: float = 0.25) -> float:
    """Calculates Sample Entropy (SampEn) using nolds.sampen()."""
    if len(data) < m + 1: return np.nan
    tolerance = r_factor * np.std(data)
    return nolds.sampen(data, emb_dim=m, tolerance=tolerance)

def feat_dfa(data: np.ndarray, min_n=100, n_windows=30) -> float:
    """
    Calculates Detrended Fluctuation Analysis (DFA) using nolds.dfa().
    """
    if len(data) < min_n: return np.nan
    return nolds.dfa(data, order=1, overlap=False, fit_exp='poly')

# We didn't end up using MSE because of computational cost
def feat_multiscale_entropy(
    data: np.ndarray,
    max_scale: int = 7,
    m: int = 2,
    r_factor: float = 0.25
) -> np.ndarray:
    """
    Calculates Multiscale Entropy (MSE) using pyentrp.entropy.multiscale_entropy().
    """
    if len(data) < max_scale: return np.array([np.nan] * max_scale)

    tolerance = r_factor * np.std(data)
    # Temporarily suppress the RuntimeWarning during the MSE calculation.
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=RuntimeWarning)
        
        mse_vector = pe.multiscale_entropy(
            data, 
            sample_length=max_scale, 
            tolerance=tolerance,
        )
    
    # Clean up the output: Convert infinite values (which result from log(0)) to NaN
    mse_vector[np.isinf(mse_vector)] = np.nan
    return mse_vector


def bandpower(data: np.ndarray, sf: int, band: Tuple[float, float], window_sec: int = 10) -> float:
    """
    Computes the absolute band power using Welch's method.
    
    Args:
        data: Single channel EEG time series.
        sf: Sampling frequency.
        band: Tuple (low_freq, high_freq).
        window_sec: Length of the window for Welch's method.
        
    Returns:
        Absolute power within the band.
    """
    low, high = band
    
    # Calculate window size and overlap
    nperseg = int(window_sec * sf)
    
    # Compute the Power Spectral Density (PSD)
    freqs, psd = welch(data, sf, nperseg=nperseg)
    
    # Find the indices of the frequency band
    idx_band = np.logical_and(freqs >= low, freqs <= high)
    
    # Calculate absolute band power (integral under the PSD curve)
    # Using the trapezoidal rule (summing PSD * delta_freq)
    band_power = np.trapezoid(psd[idx_band], freqs[idx_band])
    
    return band_power


# MODEL TRAINING/EVALUATION UTILITIES

def print_per_fold_accuracy(cv_results: dict, model_name: str):
    """
    Prints the individual test accuracy score obtained for each cross-validation fold.

    Args:
        cv_results: The dictionary returned by sklearn.model_selection.cross_validate.
        model_name: Name of the classifier (e.g., 'SVM - AF3 Channel').
    """
    test_scores = cv_results.get('test_accuracy')

    if test_scores is None or len(test_scores) == 0:
        print(f"Error: No 'test_accuracy' scores found for {model_name}.")
        return

    print(f"\n--- Detailed Per-Fold Accuracy for {model_name} ---")

    # Iterate through the array of scores (one score per fold)
    for i, score in enumerate(test_scores):
        print(f"Fold {i+1}: {score:.4f}")

    # Also print the overall mean and standard deviation
    mean_score = np.mean(test_scores)
    std_score = np.std(test_scores)
    print("--------------------------------------------------")
    print(f"Mean Accuracy: {mean_score:.4f} (±{std_score:.4f})")
    print("--------------------------------------------------")


def print_cv_results(cv_results, model_name):
    print(f"## 🏆 Results for {model_name} (Multi-class with 10-Fold CV)")
    print("---")
    for key in ['test_accuracy', 'test_f1_score', 'test_precision', 'test_recall', 'test_kappa']:
        metric_name = key.replace('test_', '').replace('_w', '').capitalize()
        scores = cv_results.get(key, [np.nan])
        mean_score = np.mean(scores)
        std_score = np.std(scores)
        print(f"**{metric_name}**: {mean_score:.4f} (±{std_score:.4f})")
    print("\n" + "="*50 + "\n")

