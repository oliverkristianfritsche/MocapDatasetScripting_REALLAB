import os
import shutil

base_folder = 'G:\\My Drive\\sd_datacollection_v4'
new_base_folder = 'G:\\My Drive\\sd_datacollection_v5'
subject = [f"P{str(i).zfill(2)}" for i in range(1, 14)]
raw_ik_folder = "processed_joint_kinematics"

def get_all_files(base_folder):
    all_files = []
    for root, dirs, files in os.walk(base_folder):
        for file in files:
            if file.endswith(".mot"):
                all_files.append(os.path.join(root, file))
    return all_files

def convert_filename(file):
    # Get filename from path
    file_name = os.path.basename(file)
    # Remove extension
    file_name = file_name.split(".")[0]

    parts = file_name.split("_")

    mapping = {
        "OR": "OverheadReach",
        "ER": "ShoulderRotation",
        "AS": "ArmSwing",
        "EF": "ElbowFlexion",
        "CB": "CrossbodyReach",
        "S": "Slow",
        "N": "Normal",
        "F": "Fast",
        "VF": "VeryFast",
        "M": "Max",
    }

    # Remove second index
    parts.pop(1)

    # Convert based on mapping
    for i, part in enumerate(parts):
        if part in mapping:
            parts[i] = mapping[part]

    # Extract activity name for folder structure
    activity = parts[1] if len(parts) > 1 else "Unknown"

    # Add ".mot"
    final_filename = "_".join(parts) + ".mot"

    return final_filename, activity

for s in subject:
    raw_ik_path = os.path.join(base_folder, s, raw_ik_folder)
    all_files = get_all_files(raw_ik_path)
    print(f"Subject: {s} -> Total files: {len(all_files)}")

    for file in all_files:
        new_filename, activity = convert_filename(file)

        # Construct new path
        new_folder_path = os.path.join(new_base_folder, s, activity, "RawIK")
        new_file_path = os.path.join(new_folder_path, new_filename)

        # Create the new folder if it doesn't exist
        os.makedirs(new_folder_path, exist_ok=True)

        # Copy the file to the new location
        shutil.copy(file, new_file_path)

        # Print the new path
        print(f"Copied: {file} -> {new_file_path}")

    print("\n")