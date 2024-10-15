import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns

def load_mot_file(file):
    with open(file) as f:
        lines = f.readlines()
    endheader = lines.index('endheader\n')
    df = pd.read_csv(file, skiprows=endheader + 1, sep='\s+', header=None)
    df.columns = lines[endheader + 1].split()
    df = df.apply(pd.to_numeric, errors='coerce')
    df.dropna(inplace=True)
    
    # Use only the last 3000 rows if there are more than 4000 rows
    if len(df) > 4000:
        df = df.tail(3000)
    
    return df

def calculate_joint_speed(df, time_column='time', smoothing_window=100):
    speed_df = pd.DataFrame()
    for column in df.columns:
        if column != time_column and column in channels_joints:
            try:
                df[column] = df[column].astype(float)
            except ValueError:
                print(f"Non-numeric data in column {column}; skipping this column.")
                continue
            # Calculate instantaneous angular velocity
            instantaneous_speed = np.gradient(df[column], df[time_column].astype(float))
            # Apply a moving average to smooth the instantaneous speeds
            speed_df[column] = pd.Series(instantaneous_speed).rolling(window=smoothing_window, min_periods=1).mean()
    return speed_df

def identify_trial_and_speed(base_name, trial_types, speeds):
    parts = base_name.split('_')
    if len(parts) >= 4:
        trial_type = parts[2]
        trial_speed = parts[3]
        if trial_type in trial_types and trial_speed in speeds[trial_type]:
            return trial_type, trial_speed
    return None, None

channels_joints = {
    'elbow_flex_r': 'Elbow Flexion',
    'arm_flex_r': 'Arm Flexion',
    'arm_add_r': 'Arm Adduction'
}

def plot_combined_joint_speed_distributions(mot_folder, subjects, trial_types, speeds):
    combined_speeds = {joint: [] for joint in channels_joints} 

    for subject in subjects:
        subject_folder = f"{mot_folder}/P{str(subject).zfill(2)}/processed_joint_kinematics"
        mot_files = [f for f in os.listdir(subject_folder) if f.endswith('.mot')]

        for mot_file in mot_files:
            mot_path = os.path.join(subject_folder, mot_file)
            base_name = os.path.splitext(mot_file)[0].replace('_IK', '')
            trial, speed = identify_trial_and_speed(base_name, trial_types, speeds)
            
            if trial and speed:
                mot_df = load_mot_file(mot_path)
                speed_df = calculate_joint_speed(mot_df)

                for joint in channels_joints:
                    if joint in speed_df:
                        combined_speeds[joint].extend(speed_df[joint].dropna().tolist())

    # Create a single row of KDE plots for each joint
    fig, axes = plt.subplots(nrows=1, ncols=len(channels_joints), figsize=(6 * len(channels_joints), 6))
    fig.suptitle("Combined Joint Speed KDE Distributions Across All Subjects and Speeds", fontsize=16, fontweight='bold')

    colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]

    for ax, (joint, title), color in zip(axes, channels_joints.items(), colors):
        speeds = combined_speeds[joint]
        
        mean_speed = np.mean(speeds)
        std_speed = np.std(speeds)
        median_speed = np.median(speeds)

        x_min, x_max = mean_speed - 3 * std_speed, mean_speed + 3 * std_speed
        
        sns.kdeplot(speeds, ax=ax, color=color, fill=True, alpha=0.6)
        ax.set_xlim(x_min, x_max)
        
        ax.set_title(f"Right {title}", fontsize=20)
        ax.set_xlabel("Speed (angular velocity)", fontsize=18)
        ax.set_ylabel("Density", fontsize=18)
        ax.tick_params(axis='both', labelsize=14, width=0.8, color='gray')

        ax.text(0.98, 0.95, f"Mean: {mean_speed:.2f}\nMedian: {median_speed:.2f}\nStd Dev: {std_speed:.2f}",
                transform=ax.transAxes, ha="right",fontsize=12, va="top",
                bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.3'))

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()
def main():
    mot_folder = "g:/My Drive/sd_datacollection_v4"
    subjects = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
    trial_types = ['AS', 'EF', 'ER', 'OR', 'CB']
    speeds = {
        'AS': ['S', 'N', 'F', 'VF'],
        'OR': ['90', '180', 'M'],
        'EF': ['S', 'N', 'F'],
        'ER': ['S', 'N', 'F'],
        'CB': ['S', 'N', 'F']
    }

    plot_combined_joint_speed_distributions(mot_folder, subjects, trial_types, speeds)

if __name__ == "__main__":
    main()