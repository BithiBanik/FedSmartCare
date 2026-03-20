import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import logging

# Logging setup
logging.basicConfig(level=logging.INFO)

def natural_sort_key(text):
    import re
    return [int(s) if s.isdigit() else s.lower() for s in re.split(r'(\d+)', text)]

# Preprocessing function
def process_with_Lag_llama(filepath: str):
    try:
        logging.info(f"Preprocessing file: {filepath}")

        df_rr = pd.read_csv(filepath)

        if 'timestamp' not in df_rr.columns or 'datapoint' not in df_rr.columns:
            raise ValueError(f"Missing required columns in {filepath}. Required: 'timestamp', 'datapoint'")

        df_rr["timestamp"] = pd.to_datetime(df_rr["timestamp"])
        df_rr.set_index("timestamp", inplace=True)
        df_rr.rename(columns={"datapoint": "hr"}, inplace=True)
        df_rr["hr"] = pd.to_numeric(df_rr["hr"], errors='coerce')
        df_rr.dropna(inplace=True)
        df_rr = df_rr[~df_rr.index.duplicated(keep='first')]
        df_rr = df_rr.resample("1S").mean()
        df_rr = df_rr.interpolate(method="linear")

        logging.info(f"Finished preprocessing: {filepath}")
        return df_rr

    except Exception as e:
        logging.error(f"Failed to process file {filepath}: {e}")
        raise

# Load and preprocess each CSV file in a directory
def load_and_plot_all_smartcare_files(DATASET_DIR: str):
    all_csv_files = sorted(glob.glob(os.path.join(DATASET_DIR, "*.csv")), key=natural_sort_key)

    if not all_csv_files:
        print("No CSV files found.")
        return

    for idx, filepath in enumerate(all_csv_files):
        print(f"\nProcessing file {idx+1}/{len(all_csv_files)}: {filepath}")
        try:
            df = process_with_Lag_llama(filepath)

            # Split
            train_size = int(len(df) * 0.8)
            train_df = df.iloc[:train_size]
            test_df = df.iloc[train_size:]

            # Plot
            plt.figure(figsize=(14, 5))
            plt.plot(train_df.index, train_df['hr'], label='Train HR', color='blue')
            plt.plot(test_df.index, test_df['hr'], label='Test HR', color='orange')
            plt.xlabel("Timestamp")
            plt.ylabel("Heart Rate (bpm)")
            plt.title(f"Heart Rate from File: {os.path.basename(filepath)}")
            plt.legend()
            plt.tight_layout()
            plt.grid(True)
            plt.show()

        except Exception as e:
            print(f"Skipping file due to error: {e}")

# Main
if __name__ == "__main__":
    # Replace with the actual path to your dataset folder
    DATASET_DIR = "SmartCareData/Pikachu"
    load_and_plot_all_smartcare_files(DATASET_DIR)
