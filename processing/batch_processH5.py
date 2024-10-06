import pandas as pd
import numpy as np
import os
import warnings
import h5py
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

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

def save_to_hdf5(data_dict, hdf5_path):
    with h5py.File(hdf5_path, 'w') as h5f:
        for subject, trial_data in data_dict.items():
            for trial, speed_data in trial_data.items():
                for speed, data in speed_data.items():
                    if isinstance(data, pd.DataFrame):
                        # Create dataset path in HDF5
                        dataset_path = f"subject_{subject}/{trial}/{speed}"
                        dataset = h5f.create_dataset(dataset_path, data=data.to_numpy(), compression="gzip", compression_opts=9)
                        
                        # Store column names as an attribute of the dataset
                        dataset.attrs['column_names'] = data.columns.to_list()

                        print(f"Saved dataset: {dataset_path}")
                    else:
                        print(f"Data for subject {subject}, trial {trial}, speed {speed} is not a DataFrame")

def combine_data(mot_folder, csv_folder, trial_types, speeds,subject, time_column='time', method='downsample'):
    mot_files = [f for f in os.listdir(mot_folder) if f.endswith('.mot')]
    csv_files = [f for f in os.listdir(csv_folder) if f.endswith('.csv')]

    # print("Lenth of ",subject, len(mot_files), len(csv_files),mot_files)
    assert len(mot_files) == 16
    combined_data = {trial: {speed: None for speed in speeds[trial]} for trial in trial_types}

    for mot_file in tqdm(mot_files, desc="Processing files for subject {}".format(subject)):
        mot_path = os.path.join(mot_folder, mot_file)
        base_name = os.path.splitext(mot_file)[0].replace('_IK', '')
        matching_csv_files = [f for f in csv_files if f.startswith(base_name)]
        
        if matching_csv_files:
            mot_df = load_mot_file(mot_path)
            csv_path = os.path.join(csv_folder, matching_csv_files[0])
            sensor_df = load_sensor_csv(csv_path)
            cleaned_sensor_df = clean_sensor_data(sensor_df)

            # Identify trial and speed from the filename structure PXX_TX_trialtype_trialspeed
            trial, speed = identify_trial_and_speed(base_name, trial_types, speeds)
            
            if trial and speed:
                if method == 'interpolate':
                    processed_mot_df = interpolate_mot_to_sensor_time(mot_df, cleaned_sensor_df, time_column)
                    combined_df = pd.concat([processed_mot_df, cleaned_sensor_df], axis=1)
                elif method == 'downsample':
                    downsampled_sensor_df = downsample_sensor_data(cleaned_sensor_df, len(mot_df))
                    combined_df = pd.concat([mot_df, downsampled_sensor_df], axis=1)
                    combined_df = combined_df.iloc[1:].reset_index(drop=True)

                combined_df = combined_df.loc[:, ~combined_df.columns.duplicated()]
                combined_data[trial][speed] = combined_df
            else:
                print(f"Unable to map file {base_name} to trial and speed.")
        else:
            print(f"No matching CSV file found for {mot_file}")
    
    return combined_data


def identify_trial_and_speed(base_name, trial_types, speeds):
    """
    Parse trial type and speed from filename. 
    Assumes filename format: PXX_TX_trialtype_trialspeed
    """
    # Split the base name by underscores
    parts = base_name.split('_')
    
    # Extract trial type and speed
    if len(parts) >= 4:  # Ensure the filename contains enough parts
        trial_type = parts[2]  # Assuming trial type is the third part
        trial_speed = parts[3]  # Assuming speed is the fourth part

        # Validate that the parsed trial_type and trial_speed are valid
        if trial_type in trial_types and trial_speed in speeds[trial_type]:
            return trial_type, trial_speed
    
    return None, None


def process_subject(subject, trial_types, speeds):
    """
    Process the data for a single subject.
    """
    subject_folder = f"P{str(subject).zfill(2)}"
    mot_folder = f"g:/My Drive/sd_datacollection_v3/{subject_folder}/processed_joint_kinematics/"
    csv_folder = f"g:/My Drive/sd_datacollection_v3/{subject_folder}/raw_sensor/"
    
    # Combine data for this subject
    combined_data = combine_data(mot_folder, csv_folder, trial_types, speeds,subject, method='downsample')
    return subject, combined_data

def verify_trial_types_and_speeds(subject, mot_files, csv_files, trial_types, speeds):
    """
    Verifies that there is exactly one .mot file and one .csv file for each trial type and speed combination.
    """
    # Dictionary to count files for each trial type and speed
    trial_speed_count = {trial: {speed: {'mot': 0, 'csv': 0} for speed in speeds[trial]} for trial in trial_types}

    # Check each .mot file to see which trial type and speed it belongs to
    for mot_file in mot_files:
        for trial in trial_types:
            for speed in speeds[trial]:
                if f"_{trial}_{speed}" in mot_file:
                    trial_speed_count[trial][speed]['mot'] += 1
                    break  # Stop checking once the trial type and speed are found

    # Check each .csv file to see which trial type and speed it belongs to
    for csv_file in csv_files:
        for trial in trial_types:
            for speed in speeds[trial]:
                if f"_{trial}_{speed}" in csv_file:
                    trial_speed_count[trial][speed]['csv'] += 1
                    break  # Stop checking once the trial type and speed are found

    # Check for missing or extra files
    missing_combinations = [(trial, speed) for trial, speeds_dict in trial_speed_count.items()
                            for speed, count in speeds_dict.items() if count['mot'] == 0 or count['csv'] == 0]
    extra_combinations = [(trial, speed) for trial, speeds_dict in trial_speed_count.items()
                          for speed, count in speeds_dict.items() if count['mot'] > 1 or count['csv'] > 1]

    # Assert that there are no missing or extra combinations
    assert not missing_combinations, f"Subject {subject} is missing the following trial type and speed combinations: {missing_combinations}"
    assert not extra_combinations, f"Subject {subject} has multiple files for the following trial type and speed combinations: {extra_combinations}"

    



def main():
    subjects = [1,2,3,4,5,6,7,8,9,10,11,12,13]
    trial_types = ['AS', 'EF', 'ER', 'OR', 'CB']
    speeds = {
        'AS': ['S', 'N', 'F', 'VF'],
        'OR': ['90', '180', 'M'],
        'EF': ['S', 'N', 'F'],
        'ER': ['S', 'N', 'F'],
        'CB': ['S', 'N', 'F']
    }

    data_dict = {}
    total_subjects = len(subjects)

    for subject in subjects:
        subject_folder = f"P{str(subject).zfill(2)}"  # Ensure subject folder is PXX with leading zero for single digits
        mot_folder = f"g:/My Drive/sd_datacollection_v3/{subject_folder}/processed_joint_kinematics/"
        csv_folder = f"g:/My Drive/sd_datacollection_v3/{subject_folder}/raw_sensor/"

        mot_files = [f for f in os.listdir(mot_folder) if f.endswith('.mot')]
        csv_files = [f for f in os.listdir(csv_folder) if f.endswith('.csv')]
         # Verify that each trial type has exactly one .mot file
        verify_trial_types_and_speeds(subject, mot_files, csv_files, trial_types, speeds)

    # Use a lock to make tqdm thread-safe
    lock = threading.Lock()

    with ThreadPoolExecutor(max_workers=5, thread_name_prefix='subject_thread') as executor:
        futures = {executor.submit(process_subject, subject, trial_types, speeds): subject for subject in subjects}

        with tqdm(total=total_subjects) as pbar:
            for future in as_completed(futures):
                subject, combined_data = future.result()

                # Iterate through the trials and speeds within combined_data
                for trial in combined_data:
                    for speed in combined_data[trial]:
                        df = combined_data[trial][speed]
                        if df is not None and len(df) > 4000:
                            combined_data[trial][speed] = df.iloc[2000:].reset_index(drop=True)
                
                data_dict[subject] = combined_data

                with lock:
                    pbar.update(1)  # Update tqdm within a thread-safe lock
                
                print(f"Finished processing subject {subject}")

    # Save all data to one HDF5 file
    hdf5_path = "g:/My Drive/sd_datacollection_v3/all_subjects_data.h5"
    save_to_hdf5(data_dict, hdf5_path)

    # Print the shape of each DataFrame
    for subject, trial_data in data_dict.items():
        for trial, speed_data in trial_data.items():
            for speed, df in speed_data.items():
                if df is not None:
                    print(f"Subject {subject}, Trial {trial}, Speed {speed}: DataFrame shape = {df.shape}")

if __name__ == "__main__":
    main()
