import pandas as pd
from final_files.utils import bandpower, create_windows, load_data, normalize_eeg_segment_array
import numpy as np  

RAW_SAMPLING_RATE = 128  # Assuming raw data rate
DURATION_SEC = 5 * 60 # 5 minutes per segment

# Define EEG Frequency Bands and their limits (Hz)
EEG_BANDS = {
    'Delta': [0.5, 4],
    'Theta': [4, 8],
    'Alpha': [8, 13],
    'Beta': [13, 30]
}

EEG_CHANNEL_COLUMNS = ['AF3', 'AF4', 'F3', 'F4', 'F7', 'F8', 'FC5', 'FC6', 'O1', 'O2', 'P7', 'P8', 'T7', 'T8']

def apply_bandpower_feature_extraction(
    df_eeg: pd.DataFrame,
    window_sec: float = 12.0, 
    overlap_sec: float = 8.0, 
) -> pd.DataFrame:
    """
    Converts raw EEG time series data into a feature matrix where each row 
    represents one window and includes columns for SUBJECT, GAME, and WINDOW_INDEX.
    
    Args:
        df_eeg: DataFrame containing raw EEG data segments indexed by SUBJECT and GAME.
        window_sec: Duration of the sliding window in seconds.
        overlap_sec: Duration of the overlap between windows in seconds.
        
    Returns:
        A DataFrame where each row is a single window feature vector.
    """
    all_window_rows = []
    
    # Ensure there are no duplicate subject/game rows before processing
    segment_ids = df_eeg[['SUBJECT', 'GAME']].drop_duplicates()

    # Outer loop to process each 5-minute segment (Subject x Game)
    for _, row in segment_ids.iterrows():
        subject, game = row['SUBJECT'], row['GAME']
        
        df_segment = df_eeg[(df_eeg['SUBJECT'] == subject) & (df_eeg['GAME'] == game)]
        eeg_data_full = df_segment[EEG_CHANNEL_COLUMNS].values
    
        windows = create_windows(
            eeg_data_full, 
            sf=RAW_SAMPLING_RATE, 
            window_sec=window_sec, 
            overlap_sec=overlap_sec
        )

        for w_idx, window_data in enumerate(windows):
            
            row_dict = {
                'SUBJECT': subject, 
                'GAME': game,
                'WINDOW_INDEX': w_idx + 1 # Window index starts at 1
            }
            
            # Normalize the window (Z-score standardization on this small time series)
            window_data_normalized = normalize_eeg_segment_array(window_data)
            
            # Inner loop to calculate features for each channel
            for i, ch_name in enumerate(EEG_CHANNEL_COLUMNS):
                channel_data = window_data_normalized[:, i]
                
                # Calculate power for each band
                for band_name, band_limits in EEG_BANDS.items():
                    
                    power_val = bandpower(
                        channel_data, 
                        sf=RAW_SAMPLING_RATE, 
                        band=band_limits,
                        window_sec=window_sec 
                    )
                    
                    feature_name = f"{ch_name}_{band_name}_Power"
                    row_dict[feature_name] = power_val
            
            # Append the completed window row
            all_window_rows.append(row_dict)
            
        
    return pd.DataFrame(all_window_rows)


if __name__ == "__main__":
    print("Loading data and applying bandpower feature extraction...")
    df = load_data()
    window_sec = 12.0
    overlap_sec = 8.0
    feature_df = apply_bandpower_feature_extraction(
        df,
        window_sec=window_sec,
        overlap_sec=overlap_sec
    )
    print("Feature extraction complete. Sample of resulting features:")
    feature_df.to_csv(f"eeg_bandpower_features_windowed_{window_sec}_{overlap_sec}.csv", index=False)
