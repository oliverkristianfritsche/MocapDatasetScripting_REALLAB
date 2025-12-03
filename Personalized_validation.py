###################################################################################################
# ULTRA-MoCap Upper Extremity Classification
# Conv1D+BiGRU Encoder
#
# NEW EXPERIMENT DESIGN (per subject):
#   Case 1 — Personalized:
#       - Use ONLY that subject's data (test split for this subject).
#       - 80% → train, 20% → test; from the 80%, 20% → val.
#
#   Case 2 — Population + Personalized Calibration:
#       - Train on:
#           (a) All windows from the other 12 subjects (standard LOSO train pool)
#           (b) PLUS the same 80% personal train windows from Case 1
#       - Validation: 20% of this combined training pool.
#       - Test on the SAME 20% personal test windows from Case 1.
#
# Modalities:
#   - EMG only
#   - IMU only
#   - IMU + EMG early fusion
#
# Results:
#   - Saved as CSVs in /content/Results_ConvBiGRU_Personalized/
###################################################################################################

from google.colab import drive
drive.mount('/content/drive')

# ============================================================
# 1. Imports
# ============================================================
import os
import re
import csv
import math
import random
import shutil

import h5py
import numpy as np
import pandas as pd

from scipy.signal import butter, filtfilt
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.utils.data import Dataset, DataLoader, random_split, Subset, ConcatDataset
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)

# ============================================================
# 2. Reproducibility
# ============================================================
SEED = 42

os.environ["PYTHONHASHSEED"] = str(SEED)
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"  # deterministic cuBLAS (if supported)

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Torch generator for deterministic splits
g = torch.Generator().manual_seed(SEED)

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


# ============================================================
# 3. Configuration
# ============================================================
class Config:
    """
    Simple config container.
    """

    def __init__(self, **kwargs):
        self.channels_imu_acc = kwargs.get("channels_imu_acc", [])
        self.channels_imu_gyr = kwargs.get("channels_imu_gyr", [])
        self.channels_joints = kwargs.get("channels_joints", [])
        self.channels_emg = kwargs.get("channels_emg", [])
        self.seed = kwargs.get("seed", 42)
        self.data_folder_name = kwargs.get("data_folder_name", "data.h5")
        self.dataset_root = kwargs.get("dataset_root", "./datasets")
        self.imu_transforms = kwargs.get("imu_transforms", [])
        self.joint_transforms = kwargs.get("joint_transforms", [])
        self.emg_transforms = kwargs.get("emg_transforms", [])
        self.input_format = kwargs.get("input_format", "csv")


# NOTE: Update these paths for your environment
config = Config(
    data_folder_name="/content/drive/MyDrive/Upper Extremity/All_subjects_data.h5",
    dataset_root="/content/datasets",
    input_format="csv",
    channels_imu_acc=[
        "ACCX1", "ACCY1", "ACCZ1",
        "ACCX2", "ACCY2", "ACCZ2",
        "ACCX3", "ACCY3", "ACCZ3",
        "ACCX4", "ACCY4", "ACCZ4",
        "ACCX5", "ACCY5", "ACCZ5",
        "ACCX6", "ACCY6", "ACCZ6",
    ],
    channels_imu_gyr=[
        "GYROX1", "GYROY1", "GYROZ1",
        "GYROX2", "GYROY2", "GYROZ2",
        "GYROX3", "GYROY3", "GYROZ3",
        "GYROX4", "GYROY4", "GYROZ4",
        "GYROX5", "GYROY5", "GYROZ5",
        "GYROX6", "GYROY6", "GYROZ6",
    ],
    channels_joints=["elbow_flex_r", "arm_flex_r", "arm_add_r"],
    channels_emg=["IM EMG4", "IM EMG5", "IM EMG6"],
)


# ============================================================
# 4. Sharding HDF5 to Windowed CSVs
# ============================================================
class DataSharder:
    """
    Reads subject data from a single HDF5 file and shards into
    windowed CSV segments for downstream training.
    """

    def __init__(self, config: Config, split: str):
        self.config = config
        self.h5_file_path = config.data_folder_name
        self.split = split
        self.window_length = None
        self.window_overlap = None

    def load_data(self, subjects, window_length, window_overlap, dataset_name):
        print(
            f"Processing subjects: {subjects} | "
            f"window_length={window_length}, overlap={window_overlap}"
        )

        self.window_length = window_length
        self.window_overlap = window_overlap
        self._process_and_save_patients_h5(subjects, dataset_name)

    def _process_and_save_patients_h5(self, subjects, dataset_name):
        with h5py.File(self.h5_file_path, "r") as h5_file:
            dataset_folder = os.path.join(
                self.config.dataset_root, dataset_name, self.split
            ).replace("subject", "").replace("__", "_")
            print("Dataset folder:", dataset_folder)

            if os.path.exists(dataset_folder):
                print("Dataset exists, skipping sharding...")
                return

            os.makedirs(dataset_folder, exist_ok=True)
            print("Created dataset folder:", dataset_folder)

            for subject_id in tqdm(subjects, desc="Processing subjects"):
                subject_key = subject_id
                if subject_key not in h5_file:
                    print(f"Subject {subject_key} not found in HDF5. Skipping.")
                    continue

                subject_data = h5_file[subject_key]
                session_keys = list(subject_data.keys())

                for session_id in session_keys:
                    session_data_group = subject_data[session_id]

                    for session_speed in session_data_group.keys():
                        session_data = session_data_group[session_speed]

                        imu_data, imu_columns = self._extract_channel_data(
                            session_data,
                            self.config.channels_imu_acc + self.config.channels_imu_gyr,
                        )
                        emg_data, emg_columns = self._extract_channel_data(
                            session_data, self.config.channels_emg
                        )
                        joint_data, joint_columns = self._extract_channel_data(
                            session_data, self.config.channels_joints
                        )

                        self._save_windowed_data(
                            imu_data=imu_data,
                            emg_data=emg_data,
                            joint_data=joint_data,
                            subject_key=subject_key,
                            session_id=session_id,
                            session_speed=session_speed,
                            dataset_folder=dataset_folder,
                            imu_columns=imu_columns,
                            emg_columns=emg_columns,
                            joint_columns=joint_columns,
                        )

    def _save_windowed_data(
        self,
        imu_data,
        emg_data,
        joint_data,
        subject_key,
        session_id,
        session_speed,
        dataset_folder,
        imu_columns,
        emg_columns,
        joint_columns,
    ):
        window_size = self.window_length
        overlap = self.window_overlap
        step_size = window_size - overlap

        csv_file_path = os.path.join(dataset_folder, "..", f"{self.split}_info.csv")
        os.makedirs(dataset_folder, exist_ok=True)

        csv_headers = ["file_name", "file_path"]
        file_exists = os.path.isfile(csv_file_path)

        with open(csv_file_path, mode="a", newline="") as csv_file:
            writer = csv.writer(csv_file)

            if not file_exists:
                writer.writerow(csv_headers)

            total_data_length = min(
                imu_data.shape[1], emg_data.shape[1], joint_data.shape[1]
            )

            # For longer recordings, skip first 2000 samples (warm-up)
            start = 2000 if total_data_length > 4000 else 0

            for i in range(start, total_data_length - window_size + 1, step_size):
                imu_window = imu_data[:, i : i + window_size]
                emg_window = emg_data[:, i : i + window_size]
                joint_window = joint_data[:, i : i + window_size]

                if (
                    imu_window.shape[1] != window_size
                    or emg_window.shape[1] != window_size
                    or joint_window.shape[1] != window_size
                ):
                    print(f"Skipping window {i} due to mismatched shapes.")
                    continue

                imu_df = pd.DataFrame(imu_window.T, columns=imu_columns)
                emg_df = pd.DataFrame(emg_window.T, columns=emg_columns)
                joint_df = pd.DataFrame(joint_window.T, columns=joint_columns)

                combined_df = pd.concat([imu_df, emg_df, joint_df], axis=1)

                file_name = (
                    f"{subject_key}_{session_id}_{session_speed}"
                    f"_win_{i}_ws{window_size}_ol{overlap}.csv"
                )
                file_path = os.path.join(dataset_folder, file_name)
                combined_df.to_csv(file_path, index=False)

                writer.writerow([file_name, file_path])

    def _extract_channel_data(self, session_data, channels):
        """
        Extracts per-channel data from a (possibly compound) HDF5 dataset
        and linearly interpolates NaNs.

        Returns:
          data: np.array [C, T]
          column_names: list[str] (channels that were actually found)
        """
        extracted_data = []
        new_column_names = []

        if isinstance(session_data, h5py.Dataset):
            # Case 1: Compound dataset (named fields)
            if session_data.dtype.names:
                column_names = list(session_data.dtype.names)
                for channel in channels:
                    if channel in column_names:
                        channel_data = session_data[channel][:]
                        channel_data = pd.to_numeric(channel_data, errors="coerce")
                        df = pd.DataFrame(channel_data)
                        df_interp = df.interpolate(
                            method="linear", axis=0, limit_direction="both"
                        )
                        extracted_data.append(df_interp.to_numpy().flatten())
                        new_column_names.append(channel)
                    else:
                        print(f"Channel {channel} not found in compound dataset.")
            else:
                # Case 2: Simple dataset with 'column_names' attribute
                column_names = session_data.attrs.get("column_names", [])
                column_names = list(column_names)
                assert len(column_names) > 0, "column_names not found in dataset attrs."

                for channel in channels:
                    if channel in column_names:
                        col_idx = column_names.index(channel)
                        new_column_names.append(channel)
                        channel_data = session_data[:, col_idx]
                        channel_data = pd.to_numeric(channel_data, errors="coerce")
                        df = pd.DataFrame(channel_data)
                        df_interp = df.interpolate(
                            method="linear", axis=0, limit_direction="both"
                        )
                        extracted_data.append(df_interp.to_numpy().flatten())
                    else:
                        print(f"Channel {channel} not found in session data.")

        return np.array(extracted_data), new_column_names


# ============================================================
# 5. Dataset: ImuJointPairDataset
# ============================================================
MOVEMENT_TYPES = ["OR", "EF", "ER", "CB", "AS"]
MOVEMENT_TYPE_MAP = {m: i for i, m in enumerate(MOVEMENT_TYPES)}


class ImuJointPairDataset(Dataset):
    """
    CSV-based dataset for IMU + joints + EMG windows.

    Expects sharded CSVs laid out as:
      <dataset_root>/<dataset_name>/<split>/*.csv
    plus a <split>_info.csv that logs file_name and file_path.
    """

    def __init__(
        self,
        config: Config,
        subjects,
        window_length,
        window_overlap,
        split="train",
        dataset_name="dataset",
        transforms=None,
    ):
        self.config = config
        self.split = split
        self.subjects = subjects
        self.window_length = window_length
        self.window_overlap = window_overlap if split == "train" else 0
        self.input_format = config.input_format
        self.channels_imu_acc = config.channels_imu_acc
        self.channels_imu_gyr = config.channels_imu_gyr
        self.channels_joints = config.channels_joints
        self.channels_emg = config.channels_emg
        self.transforms = (
            transforms
            if transforms is not None
            else {"imu": [], "joint": [], "emg": []}
        )
        self.dataset_name = dataset_name

        subjects_str = "_".join(map(str, subjects)).replace("subject", "").replace(
            "__", "_"
        )
        self.dataset_folder_name = (
            f"{dataset_name}_wl{self.window_length}_ol{self.window_overlap}_"
            f"{self.split}{subjects_str}"
        )
        self.root_dir = os.path.join(self.config.dataset_root, self.dataset_folder_name)

        self._ensure_resharded(subjects)

        info_path = os.path.join(self.root_dir, f"{self.split}_info.csv")
        self.data = pd.read_csv(info_path)

    def _ensure_resharded(self, subjects):
        if not os.path.exists(self.root_dir):
            print(f"Sharded data not found at {self.root_dir}. Resharding...")
            sharder = DataSharder(self.config, self.split)
            sharder.load_data(
                subjects,
                window_length=self.window_length,
                window_overlap=self.window_overlap,
                dataset_name=self.dataset_folder_name,
            )
        else:
            print(f"Sharded data found at {self.root_dir}. Skipping resharding.")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        file_name = self.data.iloc[idx, 0]
        file_path = os.path.join(self.root_dir, self.split, file_name)

        if self.input_format == "csv":
            combined_data = pd.read_csv(file_path)
        else:
            raise ValueError(f"Unsupported input format: {self.input_format}")

        imu_acc, imu_gyr, joint_data, emg_data = self._extract_and_transform(
            combined_data
        )
        movement_label = self._extract_movement_label(file_name)

        return imu_acc, imu_gyr, joint_data, emg_data, movement_label

    def _extract_and_transform(self, combined_data):
        imu_acc = self._extract_channels(combined_data, self.channels_imu_acc)
        imu_gyr = self._extract_channels(combined_data, self.channels_imu_gyr)
        joint_data = self._extract_channels(combined_data, self.channels_joints)
        emg_data = self._extract_channels(combined_data, self.channels_emg)

        imu_acc = self._apply_transforms(imu_acc, self.transforms.get("imu", []))
        imu_gyr = self._apply_transforms(imu_gyr, self.transforms.get("imu", []))
        joint_data = self._apply_transforms(joint_data, self.transforms.get("joint", []))
        emg_data = self._apply_transforms(emg_data, self.transforms.get("emg", []))

        return imu_acc, imu_gyr, joint_data, emg_data

    def _extract_movement_label(self, file_name: str) -> torch.Tensor:
        m = re.search(r"_(OR|EF|ER|CB|AS)_", file_name)
        if m:
            movement_type = m.group(1)
            idx = MOVEMENT_TYPE_MAP[movement_type]
            one_hot = torch.zeros(len(MOVEMENT_TYPES))
            one_hot[idx] = 1.0
            return one_hot
        raise ValueError(f"Unknown movement type in filename: {file_name}")

    @staticmethod
    def _extract_channels(combined_data, channels):
        return combined_data[channels].values

    @staticmethod
    def _apply_transforms(data, transforms):
        for t in transforms:
            data = t(data)
        return torch.tensor(data, dtype=torch.float32)


# ============================================================
# 6. Preprocessing: EMG & IMU
# ============================================================
def preprocess_emg_tensor(emg, fs=100, low=5, high=45, kernel_size=5):
    """
    sEMG preprocessing at 100 Hz:
      - 5–45 Hz bandpass
      - rectification
      - moving-average smoothing
      - per-window Z-score
    emg: [B, T, C]
    """
    B, T, C = emg.shape
    emg_np = emg.detach().cpu().numpy().astype(np.float32)

    def butter_bandpass(lowcut, highcut, fs, order=4):
        nyq = 0.5 * fs
        low_n = lowcut / nyq
        high_n = highcut / nyq
        b, a = butter(order, [low_n, high_n], btype="band")
        return b, a

    b, a = butter_bandpass(low, high, fs)
    emg_filtered = np.zeros_like(emg_np, dtype=np.float32)

    for bi in range(B):
        for ci in range(C):
            x = emg_np[bi, :, ci]
            x = filtfilt(b, a, x)
            x = np.abs(x)
            x = np.convolve(
                x,
                np.ones(kernel_size, dtype=np.float32) / kernel_size,
                mode="same",
            )
            emg_filtered[bi, :, ci] = x

    mean = emg_filtered.mean(axis=1, keepdims=True)
    std = emg_filtered.std(axis=1, keepdims=True) + 1e-6
    emg_z = (emg_filtered - mean) / std

    return torch.tensor(emg_z, dtype=emg.dtype, device=emg.device)


def preprocess_imu_tensor(imu):
    """
    Per-window Z-score normalization for IMU data.
    imu: [B, T, C]
    """
    imu_np = imu.detach().cpu().numpy().astype(np.float32)
    mean = imu_np.mean(axis=1, keepdims=True)
    std = imu_np.std(axis=1, keepdims=True)
    imu_norm = (imu_np - mean) / (std + 1e-6)
    return torch.tensor(imu_norm, dtype=imu.dtype, device=imu.device)


# ============================================================
# 7. Unified Conv1D + BiGRU Encoder & Models
# ============================================================
class ConvBiGRUEncoder(nn.Module):
    """
    Unified encoder for IMU, EMG, and IMU+EMG fused.
    Input  : [B, T, C]
    Output : [B, D]
    """
    def __init__(self, input_dim, d_model=128, gru_hidden=128,
                 num_layers=1, dropout=0.1):
        super().__init__()

        # ----- CNN Frontend -----
        self.conv = nn.Sequential(
            nn.Conv1d(input_dim, d_model, kernel_size=5, padding=2),
            nn.GELU(),
            nn.Conv1d(d_model, d_model, kernel_size=3, padding=1),
            nn.GELU(),
            nn.BatchNorm1d(d_model),
            nn.Dropout(dropout),
        )

        # ----- BiGRU -----
        self.bigru = nn.GRU(
            input_size=d_model,
            hidden_size=gru_hidden,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=True,
        )

        self.output_dim = gru_hidden * 2   # bidirectional

    def forward(self, x):
        # x: [B,T,C] → Conv: [B,d_model,T]
        x = x.permute(0, 2, 1)
        x = self.conv(x)            # [B, d_model, T]
        x = x.permute(0, 2, 1)      # [B, T, d_model]

        # BiGRU → [B,T,2H]
        outputs, _ = self.bigru(x)

        # Temporal average → [B,2H]
        feat = outputs.mean(dim=1)
        return feat


class EMGConvBiGRUModel(nn.Module):
    """
    EMG-only Conv1D+BiGRU model.
    Input : emg_x [B,T,C_emg]
    """
    def __init__(self, num_emg_channels, num_classes,
                 d_model=128, dropout=0.1):
        super().__init__()

        self.encoder = ConvBiGRUEncoder(
            input_dim=num_emg_channels,
            d_model=d_model,
            dropout=dropout,
        )
        self.fc = nn.Linear(self.encoder.output_dim, num_classes)

    def forward(self, emg_x):
        feat = self.encoder(emg_x)
        logits = self.fc(feat)
        return logits


class IMUConvBiGRUModel(nn.Module):
    """
    IMU-only Conv1D+BiGRU model.
    Input : imu_x [B,T,C_imu]  (acc + gyro concatenated)
    """
    def __init__(self, num_imu_channels, num_classes,
                 d_model=128, dropout=0.1):
        super().__init__()

        self.encoder = ConvBiGRUEncoder(
            input_dim=num_imu_channels,
            d_model=d_model,
            dropout=dropout,
        )
        self.fc = nn.Linear(self.encoder.output_dim, num_classes)

    def forward(self, imu_x):
        feat = self.encoder(imu_x)
        logits = self.fc(feat)
        return logits


class IMUEMGConvBiGRUModel(nn.Module):
    """
    Early-fusion IMU+EMG Conv1D+BiGRU model.
      imu_x: [B,T,C_imu]
      emg_x: [B,T,C_emg]
      fused: [B,T,C_imu+C_emg]
    """
    def __init__(self, input_dim, num_classes,
                 d_model=128, dropout=0.1):
        super().__init__()

        self.encoder = ConvBiGRUEncoder(
            input_dim=input_dim,
            d_model=d_model,
            dropout=dropout,
        )
        self.fc = nn.Linear(self.encoder.output_dim, num_classes)

    def forward(self, emg_x, imu_x):
        fused = torch.cat([imu_x, emg_x], dim=-1)   # [B,T,C_imu+C_emg]
        feat = self.encoder(fused)
        logits = self.fc(feat)
        return logits


# ============================================================
# 8. Evaluation Helper
# ============================================================
def evaluate_model(
    model,
    test_loader,
    criterion,
    device,
    modality,
    num_emg_channels,
    num_imu_sensors,
):
    """
    Evaluation wrapper used for all three modalities.
    num_emg_channels / num_imu_sensors kept for API compatibility (unused).
    """
    model.eval()
    test_preds, test_true = [], []
    test_loss = 0.0

    with torch.no_grad():
        for batch in tqdm(test_loader, desc=f"Testing [{modality}]"):
            if modality == "emg":
                _, _, _, emg, labels = batch
                emg_proc = preprocess_emg_tensor(emg).to(device)    # [B,T,C_emg]
                labels = labels.to(device).argmax(dim=1)
                outputs = model(emg_proc)

            elif modality == "imu":
                imu_acc, imu_gyr, _, _, labels = batch
                imu_acc = preprocess_imu_tensor(imu_acc)
                imu_gyr = preprocess_imu_tensor(imu_gyr)
                imu_full = torch.cat([imu_acc, imu_gyr], dim=-1).to(device)  # [B,T,C_imu]
                labels = labels.to(device).argmax(dim=1)
                outputs = model(imu_full)

            else:  # "imu_emg"
                imu_acc, imu_gyr, _, emg, labels = batch
                imu_acc = preprocess_imu_tensor(imu_acc)
                imu_gyr = preprocess_imu_tensor(imu_gyr)
                imu_full = torch.cat([imu_acc, imu_gyr], dim=-1).to(device)  # [B,T,C_imu]

                emg_proc = preprocess_emg_tensor(emg).to(device)             # [B,T,C_emg]
                labels = labels.to(device).argmax(dim=1)
                outputs = model(emg_proc, imu_full)

            loss = criterion(outputs, labels)
            test_loss += loss.item()

            _, predicted = outputs.max(1)
            test_preds.extend(predicted.cpu().numpy())
            test_true.extend(labels.cpu().numpy())

    test_loss /= max(len(test_loader), 1)
    acc = accuracy_score(test_true, test_preds)
    conf = confusion_matrix(test_true, test_preds, labels=list(range(len(MOVEMENT_TYPES))))
    precision, recall, _, _ = precision_recall_fscore_support(
        test_true,
        test_preds,
        labels=list(range(len(MOVEMENT_TYPES))),
        average=None,
    )

    return acc, conf, precision, recall, test_loss


# ============================================================
# 9. Generic training function for a given dataset & modality
# ============================================================
def run_training(
    modality,
    train_dataset,
    val_dataset,
    test_dataset,
    model_tag,
    results_folder,
    num_epochs,
    patience,
    num_emg_channels,
    num_imu_channels,
    num_classes,
):
    """
    Train + validate + test for one modality and one subject / case.
    Returns: accuracy, conf, precision, recall, test_loss
    """
    train_loader = DataLoader(
        train_dataset,
        batch_size=128,
        shuffle=True,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=128,
        shuffle=False,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=128,
        shuffle=False,
        pin_memory=True,
    )

    # ----- Model selection -----
    if modality == "emg":
        model = EMGConvBiGRUModel(
            num_emg_channels=num_emg_channels,
            num_classes=num_classes,
            d_model=128,
            dropout=0.1,
        ).to(device)
    elif modality == "imu":
        model = IMUConvBiGRUModel(
            num_imu_channels=num_imu_channels,
            num_classes=num_classes,
            d_model=128,
            dropout=0.1,
        ).to(device)
    else:  # "imu_emg"
        total_input_dim = num_imu_channels + num_emg_channels
        model = IMUEMGConvBiGRUModel(
            input_dim=total_input_dim,
            num_classes=num_classes,
            d_model=128,
            dropout=0.1,
        ).to(device)

    model_path = os.path.join(results_folder, f"{model_tag}.pt")

    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    criterion = nn.CrossEntropyLoss()
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=2
    )

    best_val_loss = float("inf")
    patience_counter = 0

    # ----- Training loop -----
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for batch in tqdm(
            train_loader, desc=f"[{model_tag} | {modality}] Epoch {epoch+1}/{num_epochs}"
        ):
            if modality == "emg":
                _, _, _, emg, labels = batch
                emg_proc = preprocess_emg_tensor(emg).to(device)
                labels = labels.to(device).argmax(dim=1)
                outputs = model(emg_proc)

            elif modality == "imu":
                imu_acc, imu_gyr, _, _, labels = batch
                imu_acc = preprocess_imu_tensor(imu_acc)
                imu_gyr = preprocess_imu_tensor(imu_gyr)
                imu_full = torch.cat([imu_acc, imu_gyr], dim=-1).to(device)
                labels = labels.to(device).argmax(dim=1)
                outputs = model(imu_full)

            else:  # "imu_emg"
                imu_acc, imu_gyr, _, emg, labels = batch
                imu_acc = preprocess_imu_tensor(imu_acc)
                imu_gyr = preprocess_imu_tensor(imu_gyr)
                imu_full = torch.cat([imu_acc, imu_gyr], dim=-1).to(device)

                emg_proc = preprocess_emg_tensor(emg).to(device)
                labels = labels.to(device).argmax(dim=1)
                outputs = model(emg_proc, imu_full)

            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            running_loss += loss.item()

        avg_loss = running_loss / max(len(train_loader), 1)

        # ----- Validation -----
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                if modality == "emg":
                    _, _, _, emg, labels = batch
                    emg_proc = preprocess_emg_tensor(emg).to(device)
                    labels = labels.to(device).argmax(dim=1)
                    outputs = model(emg_proc)

                elif modality == "imu":
                    imu_acc, imu_gyr, _, _, labels = batch
                    imu_acc = preprocess_imu_tensor(imu_acc)
                    imu_gyr = preprocess_imu_tensor(imu_gyr)
                    imu_full = torch.cat([imu_acc, imu_gyr], dim=-1).to(device)
                    labels = labels.to(device).argmax(dim=1)
                    outputs = model(imu_full)

                else:
                    imu_acc, imu_gyr, _, emg, labels = batch
                    imu_acc = preprocess_imu_tensor(imu_acc)
                    imu_gyr = preprocess_imu_tensor(imu_gyr)
                    imu_full = torch.cat([imu_acc, imu_gyr], dim=-1).to(device)
                    emg_proc = preprocess_emg_tensor(emg).to(device)
                    labels = labels.to(device).argmax(dim=1)
                    outputs = model(emg_proc, imu_full)

                val_loss += criterion(outputs, labels).item()

        val_loss /= max(len(val_loader), 1)
        scheduler.step(val_loss)
        print(
            f"[{model_tag} | {modality}] Epoch {epoch+1} | "
            f"Train: {avg_loss:.4f} | Val: {val_loss:.4f}"
        )

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), model_path)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"[{model_tag} | {modality}] Early stopping.")
                break

    print(f"[{model_tag} | {modality}] ✅ Best model saved: {model_path}")

    # ----- Test -----
    model.load_state_dict(torch.load(model_path, map_location=device))

    acc, conf, precision, recall, test_loss = evaluate_model(
        model,
        test_loader,
        criterion,
        device,
        modality,
        num_emg_channels=num_emg_channels,
        num_imu_sensors=0,
    )

    return acc, conf, precision, recall, test_loss


# ============================================================
# 10. Experiment Setup: Cases 1 & 2
# ============================================================
num_epochs = 100
patience = 10
batch_size = 128  # used inside run_training
window_length = 200
window_overlap = 0

num_classes = len(MOVEMENT_TYPES)
num_emg_channels = len(config.channels_emg)
num_imu_sensors = len(config.channels_imu_acc) // 3    # kept for compatibility
num_imu_channels = len(config.channels_imu_acc) + len(config.channels_imu_gyr)

# Default: 13 subjects labeled "subject_1" ... "subject_13"
if "all_subjects" not in globals():
    all_subjects = [f"subject_{i}" for i in range(1, 14)]

results_folder = "/content/Results_ConvBiGRU_Personalized/"
os.makedirs(results_folder, exist_ok=True)

# Case 1 CSVs
csv_case1_emg = os.path.join(results_folder, "Case1_Personal_EMGOnly_ConvBiGRU.csv")
csv_case1_imu = os.path.join(results_folder, "Case1_Personal_IMUOnly_ConvBiGRU.csv")
csv_case1_imu_emg = os.path.join(results_folder, "Case1_Personal_IMU_EMG_ConvBiGRU.csv")

rows_case1_emg, rows_case1_imu, rows_case1_imu_emg = [], [], []

# Case 2 CSVs
csv_case2_emg = os.path.join(results_folder, "Case2_PopPlusPersonal_EMGOnly_ConvBiGRU.csv")
csv_case2_imu = os.path.join(results_folder, "Case2_PopPlusPersonal_IMUOnly_ConvBiGRU.csv")
csv_case2_imu_emg = os.path.join(results_folder, "Case2_PopPlusPersonal_IMU_EMG_ConvBiGRU.csv")

rows_case2_emg, rows_case2_imu, rows_case2_imu_emg = [], [], []


for i, test_subject in enumerate(all_subjects):
    print(f"\n====== Subject {i+1}/{len(all_subjects)} | Test Subject: {test_subject} ======")

    # Standard LOSO train pool: all other subjects
    train_subjects = [s for s in all_subjects if s != test_subject]

    # LOSO-style training dataset (all other subjects)
    full_train_dataset = ImuJointPairDataset(
        config,
        train_subjects,
        window_length,
        window_overlap,
        split="train",
    )

    # Full data for this subject (we treat this as the subject's pool)
    subject_dataset = ImuJointPairDataset(
        config,
        [test_subject],
        window_length,
        0,
        split="test",
    )

    N_subj = len(subject_dataset)
    if N_subj < 10:
        print(f"Subject {test_subject} has too few windows ({N_subj}). Skipping.")
        continue

    # ============================================================
    # CASE 1: Personalized 80/20 (within-subject only)
    # ============================================================
    print(f"\n--- CASE 1 (Personalized 80/20) for {test_subject} ---")

    # Deterministic permutation of indices for this subject
    all_idx = torch.randperm(N_subj, generator=g).tolist()

    N_test = max(1, int(0.2 * N_subj))
    test_idx_personal = all_idx[:N_test]
    trainval_idx_personal = all_idx[N_test:]

    N_trainval = len(trainval_idx_personal)
    if N_trainval < 2:
        print(f"Not enough data for train/val after reserving test for {test_subject}. Skipping.")
        continue

    N_val = max(1, int(0.2 * N_trainval))
    val_idx_personal = trainval_idx_personal[:N_val]
    train_idx_personal = trainval_idx_personal[N_val:]

    ds_case1_train = Subset(subject_dataset, train_idx_personal)
    ds_case1_val = Subset(subject_dataset, val_idx_personal)
    ds_case1_test = Subset(subject_dataset, test_idx_personal)

    # ----- Case 1: EMG-only -----
    acc, conf, precision, recall, tloss = run_training(
        modality="emg",
        train_dataset=ds_case1_train,
        val_dataset=ds_case1_val,
        test_dataset=ds_case1_test,
        model_tag=f"{test_subject}_Case1_EMG",
        results_folder=results_folder,
        num_epochs=num_epochs,
        patience=patience,
        num_emg_channels=num_emg_channels,
        num_imu_channels=num_imu_channels,
        num_classes=num_classes,
    )
    rows_case1_emg.append({
        "case": 1,
        "subject": test_subject,
        "accuracy": acc,
        "precision_per_class": np.array_str(precision),
        "recall_per_class": np.array_str(recall),
        "confusion_matrix": conf.tolist(),
    })
    pd.DataFrame(rows_case1_emg).to_csv(csv_case1_emg, index=False)
    print(f"[Case 1 | EMG] {test_subject} | Test ACC = {acc:.4f}")

    # ----- Case 1: IMU-only -----
    acc, conf, precision, recall, tloss = run_training(
        modality="imu",
        train_dataset=ds_case1_train,
        val_dataset=ds_case1_val,
        test_dataset=ds_case1_test,
        model_tag=f"{test_subject}_Case1_IMU",
        results_folder=results_folder,
        num_epochs=num_epochs,
        patience=patience,
        num_emg_channels=num_emg_channels,
        num_imu_channels=num_imu_channels,
        num_classes=num_classes,
    )
    rows_case1_imu.append({
        "case": 1,
        "subject": test_subject,
        "accuracy": acc,
        "precision_per_class": np.array_str(precision),
        "recall_per_class": np.array_str(recall),
        "confusion_matrix": conf.tolist(),
    })
    pd.DataFrame(rows_case1_imu).to_csv(csv_case1_imu, index=False)
    print(f"[Case 1 | IMU] {test_subject} | Test ACC = {acc:.4f}")

    # ----- Case 1: IMU + EMG early fusion -----
    acc, conf, precision, recall, tloss = run_training(
        modality="imu_emg",
        train_dataset=ds_case1_train,
        val_dataset=ds_case1_val,
        test_dataset=ds_case1_test,
        model_tag=f"{test_subject}_Case1_IMU_EMG",
        results_folder=results_folder,
        num_epochs=num_epochs,
        patience=patience,
        num_emg_channels=num_emg_channels,
        num_imu_channels=num_imu_channels,
        num_classes=num_classes,
    )
    rows_case1_imu_emg.append({
        "case": 1,
        "subject": test_subject,
        "accuracy": acc,
        "precision_per_class": np.array_str(precision),
        "recall_per_class": np.array_str(recall),
        "confusion_matrix": conf.tolist(),
    })
    pd.DataFrame(rows_case1_imu_emg).to_csv(csv_case1_imu_emg, index=False)
    print(f"[Case 1 | IMU+EMG] {test_subject} | Test ACC = {acc:.4f}")

    # ============================================================
    # CASE 2: Population + Personalized Calibration
    #   Train = (12 other subjects) + personal 80% from above
    #   Val   = 20% of this combined pool
    #   Test  = same personal 20% as Case 1
    # ============================================================
    print(f"\n--- CASE 2 (Population + Personalized 80%) for {test_subject} ---")

    combined_train_dataset = ConcatDataset([full_train_dataset, ds_case1_train])
    N_combined = len(combined_train_dataset)

    if N_combined < 10:
        print(f"Combined train data too small for {test_subject} (N={N_combined}). Skipping Case 2.")
        continue

    # Deterministic split for combined train/val
    all_idx_combined = torch.randperm(N_combined, generator=g).tolist()
    N_val2 = max(1, int(0.2 * N_combined))
    val_idx_combined = all_idx_combined[:N_val2]
    train_idx_combined = all_idx_combined[N_val2:]

    ds_case2_train = Subset(combined_train_dataset, train_idx_combined)
    ds_case2_val = Subset(combined_train_dataset, val_idx_combined)
    ds_case2_test = ds_case1_test  # SAME personal 20% as Case 1

    # ----- Case 2: EMG-only -----
    acc, conf, precision, recall, tloss = run_training(
        modality="emg",
        train_dataset=ds_case2_train,
        val_dataset=ds_case2_val,
        test_dataset=ds_case2_test,
        model_tag=f"{test_subject}_Case2_EMG",
        results_folder=results_folder,
        num_epochs=num_epochs,
        patience=patience,
        num_emg_channels=num_emg_channels,
        num_imu_channels=num_imu_channels,
        num_classes=num_classes,
    )
    rows_case2_emg.append({
        "case": 2,
        "subject": test_subject,
        "accuracy": acc,
        "precision_per_class": np.array_str(precision),
        "recall_per_class": np.array_str(recall),
        "confusion_matrix": conf.tolist(),
    })
    pd.DataFrame(rows_case2_emg).to_csv(csv_case2_emg, index=False)
    print(f"[Case 2 | EMG] {test_subject} | Test ACC = {acc:.4f}")

    # ----- Case 2: IMU-only -----
    acc, conf, precision, recall, tloss = run_training(
        modality="imu",
        train_dataset=ds_case2_train,
        val_dataset=ds_case2_val,
        test_dataset=ds_case2_test,
        model_tag=f"{test_subject}_Case2_IMU",
        results_folder=results_folder,
        num_epochs=num_epochs,
        patience=patience,
        num_emg_channels=num_emg_channels,
        num_imu_channels=num_imu_channels,
        num_classes=num_classes,
    )
    rows_case2_imu.append({
        "case": 2,
        "subject": test_subject,
        "accuracy": acc,
        "precision_per_class": np.array_str(precision),
        "recall_per_class": np.array_str(recall),
        "confusion_matrix": conf.tolist(),
    })
    pd.DataFrame(rows_case2_imu).to_csv(csv_case2_imu, index=False)
    print(f"[Case 2 | IMU] {test_subject} | Test ACC = {acc:.4f}")

    # ----- Case 2: IMU + EMG early fusion -----
    acc, conf, precision, recall, tloss = run_training(
        modality="imu_emg",
        train_dataset=ds_case2_train,
        val_dataset=ds_case2_val,
        test_dataset=ds_case2_test,
        model_tag=f"{test_subject}_Case2_IMU_EMG",
        results_folder=results_folder,
        num_epochs=num_epochs,
        patience=patience,
        num_emg_channels=num_emg_channels,
        num_imu_channels=num_imu_channels,
        num_classes=num_classes,
    )
    rows_case2_imu_emg.append({
        "case": 2,
        "subject": test_subject,
        "accuracy": acc,
        "precision_per_class": np.array_str(precision),
        "recall_per_class": np.array_str(recall),
        "confusion_matrix": conf.tolist(),
    })
    pd.DataFrame(rows_case2_imu_emg).to_csv(csv_case2_imu_emg, index=False)
    print(f"[Case 2 | IMU+EMG] {test_subject} | Test ACC = {acc:.4f}")

    ##########################################################################
    # SAVE ZIP TO GOOGLE DRIVE AFTER EACH SUBJECT
    ##########################################################################
    drive_results_dir = "/content/drive/MyDrive/Upper_extremity_results/"
    os.makedirs(drive_results_dir, exist_ok=True)

    # name per subject to avoid overwriting
    zip_path = os.path.join(drive_results_dir, f"{test_subject}_Personalized_Results")

    shutil.make_archive(
        base_name=zip_path,     # no .zip needed, Colab adds it
        format="zip",
        root_dir=results_folder
    )

    print(f"📦 Saved ZIP for {test_subject} → {zip_path}.zip")



print("\n🎉 All subjects completed for Case 1 & Case 2.")
print(f"📄 Case 1 EMG       : {csv_case1_emg}")
print(f"📄 Case 1 IMU       : {csv_case1_imu}")
print(f"📄 Case 1 IMU+EMG   : {csv_case1_imu_emg}")
print(f"📄 Case 2 EMG       : {csv_case2_emg}")
print(f"📄 Case 2 IMU       : {csv_case2_imu}")
print(f"📄 Case 2 IMU+EMG   : {csv_case2_imu_emg}")
