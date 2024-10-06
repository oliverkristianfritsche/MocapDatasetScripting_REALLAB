import pandas as pd
import numpy as np
import os
import warnings
from tqdm import tqdm

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
    units = data_lines[1].strip().split(',')
    data = data_lines[2:]

    data_rows = []
    for line in data:
        row = line.strip().split(',')
        if len(row) < len(columns):
            row.extend([''] * (len(columns) - len(row)))
        elif len(row) > len(columns):
            row = row[:len(columns)]
        data_rows.append(row)
    
    df = pd.DataFrame(data_rows, columns=columns)

    return df

def clean_sensor_data(df):
    sensor_data_np = df.to_numpy()
    columns_np = np.array(df.columns)

    for data in sensor_data_np[0]:
        if data == '0':
            indexes = np.where(sensor_data_np[0] == data)
            sensor_data_np = np.delete(sensor_data_np, indexes, axis=1)
            columns_np = np.delete(columns_np, indexes)
    
    cleaned_df = pd.DataFrame(sensor_data_np, columns=columns_np)
    cleaned_df.drop(columns=['Frame'], inplace=True)
    return cleaned_df

def interpolate_mot_to_sensor_time(mot_df, sensor_df, time_column):
    mot_df[time_column] = pd.to_numeric(mot_df[time_column], errors='coerce')
    mot_df = mot_df.dropna(subset=[time_column])
    mot_time = mot_df[time_column].to_numpy().astype(float)

    sensor_time = np.linspace(mot_time[0], mot_time[-1], num=len(sensor_df))

    interpolated_mot_data = {}
    for column in mot_df.columns:
        if column != time_column:
            interpolated_mot_data[column] = np.interp(sensor_time, mot_time, mot_df[column].astype(float))
    
    interpolated_mot_data[time_column] = sensor_time
    interpolated_mot_df = pd.DataFrame(interpolated_mot_data)
    return interpolated_mot_df

def downsample_sensor_data(sensor_df, target_len):
    current_len = len(sensor_df)
    indices = np.linspace(0, current_len - 1, target_len).astype(int)
    downsampled_df = sensor_df.iloc[indices].reset_index(drop=True)
    return downsampled_df

def combine_and_save_data(mot_folder, csv_folder, output_folder, time_column='time', method='interpolate'):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    mot_files = [f for f in os.listdir(mot_folder) if f.endswith('.mot')]
    csv_files = [f for f in os.listdir(csv_folder) if f.endswith('.csv')]

    combined_files = []

    for mot_file in tqdm(mot_files, desc="Processing .mot files"):
        mot_path = os.path.join(mot_folder, mot_file)
        base_name = os.path.splitext(mot_file)[0].replace('_IK', '')
        output_path = os.path.join(output_folder, f"{base_name}_combined.csv")
        matching_csv_files = [f for f in csv_files if f.startswith(base_name)]
        
        if os.path.exists(output_path):
            print(f"Combined file already exists: {output_path}")
            combined_files.append(output_path)
            continue
        
        if matching_csv_files:
            mot_df = load_mot_file(mot_path)
            csv_path = os.path.join(csv_folder, matching_csv_files[0])
            sensor_df = load_sensor_csv(csv_path)
            cleaned_sensor_df = clean_sensor_data(sensor_df)
            
            if method == 'interpolate':
                processed_mot_df = interpolate_mot_to_sensor_time(mot_df, cleaned_sensor_df, time_column)
                combined_df = pd.concat([processed_mot_df, cleaned_sensor_df], axis=1)
            elif method == 'downsample':
                downsampled_sensor_df = downsample_sensor_data(cleaned_sensor_df, len(mot_df))
                combined_df = pd.concat([mot_df, downsampled_sensor_df], axis=1)
                combined_df = combined_df.iloc[1:].reset_index(drop=True)
            
            combined_df = combined_df.loc[:, ~combined_df.columns.duplicated()]

            combined_df.to_csv(output_path, index=False)
            combined_files.append(output_path)
            print(f"Saved combined data to {output_path}")
        else:
            print(f"No matching CSV file found for {mot_file}")
    
    return combined_files

def trim_and_save(file_path, output_path):
    df = pd.read_csv(file_path)
    if len(df) > 4000:
        df = df.iloc[2000:].reset_index(drop=True)

    #drop any rows with empty cells
    df.replace("", pd.NA, inplace=True)
    df.dropna(inplace=True)
    

    df.to_csv(output_path, index=False)
    print(f"Trimmed file saved to {output_path}")

def main():

    subjects = [1,2,3,4,5,6,7,8,9,10,11,12,13]
    for i in subjects:
        mot_folder = f"g:/My Drive/sd_datacollection/subject_{i}/processed/"
        csv_folder = f"g:/My Drive/sd_datacollection/subject_{i}/sensors/"
        output_folder = f"g:/My Drive/sd_datacollection/subject_{i}/combined/"
        method = 'downsample'

        # Combine data
        combined_files = combine_and_save_data(mot_folder, csv_folder, output_folder, method=method)

        # Trim the completed combined CSV files
        for file_path in combined_files:      
            trim_and_save(file_path, file_path)

if __name__ == "__main__":
    main()
