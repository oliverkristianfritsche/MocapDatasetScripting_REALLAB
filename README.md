# Processing Directory

This directory contains scripts and notebooks for batch processing of sensor data.

## Files

### [batch_IK.ipynb](processing/batch_IK.ipynb)
This Jupyter Notebook is used for scripting batch processes for OpenSim inverse kinematics. It includes code cells for loading, processing, and analyzing kinematic data, as well as visualizing the results.

### [batch_processH5.py](processing/batch_processH5.py)
This Python script is designed for combining all subject data into a single HDF5 file. It includes functions to:
- Read data from multiple HDF5 files.
- Process and clean the data.
- Combine the data into a single HDF5 file.

### [movement_type_classifier.ipynb](movement_type_classifier.ipynb)
This Jupyter Notebook trains a simple classifier to predict movement types from the dataset.

### [visualize_subject_distributions.py](visualize_subject_distributions.py)
This script analyzes the distribution of subject attributes such as height, age, and weight. It generates visualizations to help understand the demographic characteristics of the subjects in the dataset.

### [visualize_speed_distributions.py](visualize_speed_distributions.py)
This script visualizes the joint speed distributions from motion capture (MOT) files. It creates plots to analyze the speed profiles of different joints during various trial types and speeds, aiding in the understanding of movement dynamics.