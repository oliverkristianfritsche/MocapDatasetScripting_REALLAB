import h5py

def inspect_h5_lengths(h5_path):
    total_trials = 0
    with h5py.File(h5_path, 'r') as h5f:
        for subject in h5f.keys():
            subject_group = h5f[subject]
            for trial in subject_group.keys():
                
                trial_group = subject_group[trial]
                for speed in trial_group.keys():
                    total_trials += 1
                    dataset = trial_group[speed]
                    length = dataset.shape[0]
                    print(f"{subject}/{trial}/{speed} -> Length: {length}")
    print(f"Total trials: {total_trials}")

inspect_h5_lengths("g:/My Drive/sd_datacollection_v5/all_subjects_data_final.h5")
