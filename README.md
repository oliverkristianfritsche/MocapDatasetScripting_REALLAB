# Processing Directory

This directory contains scripts and notebooks for batch processing of sensor data.

## Files

### [batch_IK.ipynb](processing/batch_IK.ipynb)
This Jupyter Notebook is used for scripting batch processes for OpenSim inverse kinematics. It includes code cells for loading, processing, and analyzing kinematic data, as well as visualizing the results.

### [batch_process.py](processing/batch_process.py)
This Python script contains functions for reading and processing sensor data from CSV files. It includes functionality to:
- Read data from a file and convert it into a pandas DataFrame.
- Clean the sensor data by handling missing or zero values.

### [batch_processH5.py](processing/batch_processH5.py)
This Python script is designed for combining all subject data into a single HDF5 file. It includes functions to:
- Read data from multiple HDF5 files.
- Process and clean the data.
- Combine the data into a single HDF5 file.