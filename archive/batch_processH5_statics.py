import pandas as pd
import numpy as np
import os
import h5py
from tqdm import tqdm
import warnings

# Suppress specific warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

def load_mot_file(file):
    with open(file) as f:
        lines = f.readlines()
    endheader = lines.index('endheader\n')
    df = pd.read_csv(file, skiprows=endheader + 1, sep='\s+', header=None)
    df.columns = lines[endheader + 1].split()
    return df

def load_sensor_csv(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
    data_lines = lines[3:]
    columns = data_lines[0].strip().split(',')
    data = [line.strip().split(',') for line in data_lines[2:]]
    df = pd.DataFrame(data, columns=columns)
    return df

def clean_sensor_data(df):
    sensor_data_np = df.to_numpy()
    columns_np = np.array(df.columns)
    indexes = np.where(sensor_data_np[0] == '0')[0]
    sensor_data_np = np.delete(sensor_data_np, indexes, axis=1)
    columns_np = np.delete(columns_np, indexes)
    cleaned_df = pd.DataFrame(sensor_data_np, columns=columns_np)
    cleaned_df.drop(columns=['Frame'], inplace=True)
    return cleaned_df

def interpolate_mot_to_sensor_time(mot_df, sensor_df, time_column):
    mot_df[time_column] = pd.to_numeric(mot_df[time_column], errors='coerce')
    mot_time = mot_df.dropna(subset=[time_column])[time_column].to_numpy().astype(float)
    sensor_time = np.linspace(mot_time[0], mot_time[-1], num=len(sensor_df))
    interpolated_mot_data = {col: np.interp(sensor_time, mot_time, mot_df[col].astype(float)) for col in mot_df.columns if col != time_column}
    interpolated_mot_data[time_column] = sensor_time
    interpolated_mot_df = pd.DataFrame(interpolated_mot_data)
    return interpolated_mot_df

def downsample_sensor_data(sensor_df, target_len):
    indices = np.linspace(0, len(sensor_df) - 1, target_len).astype(int)
    downsampled_df = sensor_df.iloc[indices].reset_index(drop=True)
    return downsampled_df

def save_to_hdf5(data_dict, hdf5_path):
    with h5py.File(hdf5_path, 'w') as h5f:
        for subject, data in data_dict.items():
            # Convert non-numeric columns to strings
            data = data.applymap(lambda x: str(x) if isinstance(x, (list, tuple)) else x)
            for col in data.select_dtypes(include=['object']).columns:
                data[col] = data[col].astype(str)

            # Convert DataFrame to numeric where possible
            data = data.apply(pd.to_numeric, errors='ignore')

            # Create dataset path in HDF5
            dataset_path = f"subject_{subject}/static"
            dataset = h5f.create_dataset(dataset_path, data=data.to_numpy(), compression="gzip", compression_opts=9)
            dataset.attrs['column_names'] = data.columns.to_list()

            print(f"Saved dataset: {dataset_path}")


def combine_data_for_static_trials(folder_path, time_column='time', method='downsample'):
    csv_files = [f for f in os.listdir(folder_path) if f.endswith('_static.csv')]
    mot_files = [f for f in os.listdir(folder_path) if f.endswith('_static.mot')]
    
    combined_data = {}
    
    for mot_file in tqdm(mot_files, desc="Processing static trials"):
        base_name = mot_file.split('_static')[0]
        matching_csv = f"{base_name}_static.csv"
        matching_trc = f"{base_name}_static.trc"

        if matching_csv in csv_files:
            mot_df = load_mot_file(os.path.join(folder_path, mot_file))
            sensor_df = load_sensor_csv(os.path.join(folder_path, matching_csv))
            cleaned_sensor_df = clean_sensor_data(sensor_df)

            # Interpolate or downsample sensor data to match mot data
            if method == 'interpolate':
                interpolated_mot_df = interpolate_mot_to_sensor_time(mot_df, cleaned_sensor_df, time_column)
                combined_df = pd.concat([interpolated_mot_df, cleaned_sensor_df], axis=1)
            elif method == 'downsample':
                downsampled_sensor_df = downsample_sensor_data(cleaned_sensor_df, len(mot_df))
                combined_df = pd.concat([mot_df, downsampled_sensor_df], axis=1)
                combined_df = combined_df.loc[:, ~combined_df.columns.duplicated()]
            
            combined_data[base_name] = combined_df
        else:
            print(f"No matching CSV file found for {mot_file}")
    
    return combined_data

def main():
    # Folder containing all static trial files
    folder_path = "G:/My Drive/SD_statictrialswithsensordata/SD_statictrialswithsensordata"
    time_column = "time"
    
    combined_data = combine_data_for_static_trials(folder_path, time_column=time_column, method='downsample')
    
    # Save all combined data to an HDF5 file
    hdf5_path = "G:/My Drive/SD_statictrialswithsensordata/all_static_trials_data.h5"
    save_to_hdf5(combined_data, hdf5_path)

    # Print the shape of each DataFrame for verification
    for subject, df in combined_data.items():
        print(f"Subject {subject}: DataFrame shape = {df.shape}")

if __name__ == "__main__":
    main()