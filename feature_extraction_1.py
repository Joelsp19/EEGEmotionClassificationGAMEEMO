
from utils import load_data, get_subsignal_and_coefficients, normalize_eeg_segment
from utils import feat_hjorth_complexity, feat_multiscale_entropy, feat_sample_entropy, feat_dfa, feat_shannon_entropy, feat_std_dev, feat_zero_crossings, feat_avg_wavelet_energy, feat_log_energy_entropy, feat_hjorth_activity, feat_hjorth_mobility
from joblib import Parallel, delayed

import pandas as pd
import numpy as np
from tqdm import tqdm
import os

NUM_CHANNELS = 14
SAMPLING_RATE_HZ = 128 # Hz
WAVELET = 'db2' # Daubechies 2nd order filter
DECOMPOSITION_LEVEL = 4 # Necessary to generate A4, D4, D3, D2, D1
BAND_COEFFICIENT_MAP = {
    'Delta_Theta': 0, # A4 Coefficient
    'Alpha': 1,       # D4 Coefficient
    'Beta': 2,        # D3 Coefficient
    'Gamma': 3,       # D2 Coefficient
    'Noise_D1': 4     # D1 Coefficient 
}
RECONSTRUCTION_BANDS = ['Delta_Theta', 'Alpha', 'Beta', 'Gamma']
CHANNEL_NAMES = ['AF3', 'AF4', 'F3', 'F4', 'F7', 'F8', 'FC5', 'FC6', 'O1', 'O2', 'P7', 'P8', 'T7', 'T8'] # Adjust to your channel names

def get_all_subsignals_coefficients_dict(eeg_data):
    all_subsignals_dict = {}
    all_coefficients_dict = {}

    for i, channel_name in enumerate(CHANNEL_NAMES):
        channel_signal = eeg_data[:, i]

        # Get the 4 reconstructed sub-signals and the 5 raw coefficient arrays
        channel_subsignals, channel_raw_coeffs = get_subsignal_and_coefficients(
            channel_signal, WAVELET, DECOMPOSITION_LEVEL, BAND_COEFFICIENT_MAP, RECONSTRUCTION_BANDS
        )

        normalized_subsignals = normalize_eeg_segment(channel_subsignals)

        # Store reconstructed time-series
        for band, sub_signal_array in normalized_subsignals.items():
            key = f'{channel_name}_{band}'
            all_subsignals_dict[key] = sub_signal_array

        # Store raw coefficients (A4, D4, D3, D2, D1)
        for band, coeff_array in channel_raw_coeffs.items():
            key = f'{channel_name}_{band}_Coeff'
            all_coefficients_dict[key] = coeff_array

    return all_subsignals_dict, all_coefficients_dict



def feature_extraction(all_subsignals_dict, all_coefficients_dict):
    feature_rows = []
    feature_names = []

    # List of Time-Domain features to calculate on sub-signals (48 arrays)
    time_domain_features = {
        'STD': feat_std_dev,
        'ZCR': feat_zero_crossings,
        'Hjorth_Act': feat_hjorth_activity,
        'Hjorth_Mob': feat_hjorth_mobility,
        'Hjorth_Comp': feat_hjorth_complexity,
        'DFA': feat_dfa,
        'SampEn': feat_sample_entropy,
        # 'MSE' : feat_multiscale_entropy
    }

    # List of Coefficient features to calculate on raw coefficients (60 arrays)
    coeff_domain_features = {
        'AvgE': feat_avg_wavelet_energy,
        'LogEE': feat_log_energy_entropy,
        'ShanE': feat_shannon_entropy,
    }

    # Initialize a single row of features for this epoch/trial
    current_features = {}

    # 2a. Calculate Time-Domain features on 48 reconstructed sub-signals
    for key, data_array in all_subsignals_dict.items():      # Example key: Channel_1_Delta_Theta
        channel_band = key
        for feat_name, feat_func in time_domain_features.items():
            feature_key = f'{channel_band}_{feat_name}'
            current_features[feature_key] = feat_func(data_array)
            feature_names.append(feature_key)

    # 2b. Calculate Coefficient features on 60 raw coefficient arrays
    for key, coeff_array in all_coefficients_dict.items():
        # Example key: Channel_1_Delta_Theta_Coeff
        channel_band = key.replace('_Coeff', '')
        for feat_name, feat_func in coeff_domain_features.items():
            feature_key = f'{channel_band}_{feat_name}'
            current_features[feature_key] = feat_func(coeff_array)
            feature_names.append(feature_key)


    # Remove duplicates from feature_names (due to repeated appending in the loop)
    final_feature_columns = sorted(list(set(current_features.keys())))
    final_feature_values = [current_features[col] for col in final_feature_columns]
    return final_feature_values, final_feature_columns


def generate_features_task(subject, game):
    """
    A single task executed in parallel: Loads one segment, normalizes, extracts features.

    Returns: A dictionary containing all calculated features plus metadata.
    """
    global df_processed # Access the global DataFrame (efficient for reading)

    # 1. Select data segment (Must return (Time_Steps, Channels) array)
    segment_df = df_processed[(df_processed['SUBJECT'] == subject) & (df_processed['GAME'] == game)]
    eeg_data = segment_df[CHANNEL_NAMES].values

    if eeg_data.size == 0:
        return None

    # 2. Decompose Signals (DWT)
    all_subsignals_dict, all_coefficients_dict = get_all_subsignals_coefficients_dict(eeg_data)
    # 3. Feature Extraction
    final_feature_values, final_feature_columns = feature_extraction(all_coefficients_dict, all_subsignals_dict)

    # 4. Format Output
    row_dict = {col: val for col, val in zip(final_feature_columns, final_feature_values)}
    row_dict["SUBJECT"] = subject
    row_dict["GAME"] = game

    return row_dict


def run_parallel():
    subjects = [i for i in range(1,29)]
    games = [1,2,3,4]
    tasks = [(subject, game) for subject in subjects for game in games]
    total_tasks = len(tasks)
    print(f"Starting parallel feature extraction for {total_tasks} segments across {os.cpu_count()} cores.")

    all_feature_rows = []

    results_iterator = Parallel(n_jobs=15, verbose=0)(
        delayed(generate_features_task)(subject, game)
        for subject, game in tqdm(tasks)
    )

    # add results to our final csv
    for result in results_iterator:
        if result is not None:
            all_feature_rows.append(result)

    final_df = pd.DataFrame(all_feature_rows)
    output_filename = f"simple_coefficients.csv" # change filename here
    final_df.to_csv(output_filename, index=False)

    print(f"\n--- SUCCESS ---")
    print(f"Extraction complete in parallel. Saved {len(final_df)} rows to '{output_filename}'")
    print(f"Speedup: Process now runs {os.cpu_count()} tasks concurrently (using robust tqdm monitoring).")

if __name__ == "__main__":
    # Note for our final results (simple_features_paper) we decied to compute DFA + SampleEntropy per channel instead of per subsignal
    # This involved doing one pass without DFA + SampEn
    # Another pass computing only DFA + SampEn per channel (remove the decomposition part)
    # Then merging the two results on SUBJECT + GAME
    # The code above does all features in one pass for simplicity
    df_processed = load_data()
    run_parallel()