import logging
from pathlib import Path
import pickle
import torch
import numpy as np
import os
from scipy.signal import resample
from typing import List, Tuple, Optional, Dict, Any, Union


class MEGDataset(torch.utils.data.Dataset):
    """PyTorch Dataset for loading preprocessed MEG data.

    This dataset loads preprocessed MEG data from disk, stored as PyTorch tensors.
    Each data sample contains the MEG signal and its associated label.

    Attributes:
        root (str): Root directory containing the processed data.
        files (List[str]): List of filenames to use.
        logger (logging.Logger): Logger for this class.
    """

    def __init__(self, root: str, files: List[str], log_dir: Optional[str] = None):
        """Initialize the MEG dataset.

        Args:
            root: Root directory containing the processed data.
            files: List of filenames to use.
            log_dir: Optional directory for log files. If None, logs to console only.
        """
        self.root = root
        self.files = files
        self.logger = logging.getLogger(__name__ + ".MEGDataset")
        
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
            file_handler = logging.FileHandler(os.path.join(log_dir, "meg_dataset.log"))
            file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
            self.logger.addHandler(file_handler)
        
        self.logger.info(f"Initialized MEGDataset with {len(files)} files from {root}")

    def __len__(self) -> int:
        """Return the number of samples in the dataset.

        Returns:
            Number of samples in the dataset.
        """
        return len(self.files)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get a sample from the dataset.

        Args:
            idx: Index of the sample to retrieve.

        Returns:
            Tuple containing the MEG data and its label.
        """
        # Load MEG data of shape (n_channels, sample_length)
        file_path = Path(self.root, self.files[idx])
        try:
            sample = torch.load(file_path)
            return sample['data'], sample['label']
        except Exception as e:
            self.logger.error(f"Error loading file {file_path}: {str(e)}")
            raise


class TUABLoader(torch.utils.data.Dataset):
    """PyTorch Dataset for loading TUH Abnormal EEG (TUAB) data.
    
    Attributes:
        root (str): Root directory containing the processed data.
        files (List[str]): List of filenames to use.
        default_rate (int): Default sampling rate of the data.
        sampling_rate (int): Target sampling rate for resampling.
        logger (logging.Logger): Logger for this class.
    """
    
    def __init__(self, root: str, files: List[str], sampling_rate: int = 200, log_dir: Optional[str] = None):
        """Initialize the TUAB dataset.
        
        Args:
            root: Root directory containing the processed data.
            files: List of filenames to use.
            sampling_rate: Target sampling rate for the data.
            log_dir: Optional directory for log files. If None, logs to console only.
        """
        self.root = root
        self.files = files
        self.default_rate = 200
        self.sampling_rate = sampling_rate
        self.logger = logging.getLogger(__name__ + ".TUABLoader")
        
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
            file_handler = logging.FileHandler(os.path.join(log_dir, "tuab_dataset.log"))
            file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
            self.logger.addHandler(file_handler)
        
        self.logger.info(f"Initialized TUABLoader with {len(files)} files from {root}")
        self.logger.info(f"Using sampling rate: {sampling_rate} Hz (default: {self.default_rate} Hz)")

    def __len__(self) -> int:
        """Return the number of samples in the dataset.
        
        Returns:
            Number of samples in the dataset.
        """
        return len(self.files)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, Union[int, float]]:
        """Get a sample from the dataset.
        
        Args:
            index: Index of the sample to retrieve.
            
        Returns:
            Tuple containing the EEG data and its label.
        """
        try:
            file_path = os.path.join(self.root, self.files[index])
            sample = pickle.load(open(file_path, "rb"))
            X = sample["X"]
            
            # Resample from default rate to target rate if needed
            if self.sampling_rate != self.default_rate:
                X = resample(X, 10 * self.sampling_rate, axis=-1)
                
            # Normalize the signal using 95th percentile
            X = X / (
                np.quantile(np.abs(X), q=0.95, method="linear", axis=-1, keepdims=True)
                + 1e-8
            )
            
            Y = sample["y"]
            X = torch.FloatTensor(X)
            return X, Y
            
        except Exception as e:
            self.logger.error(f"Error loading file {self.files[index]}: {str(e)}")
            raise


class CHBMITLoader(torch.utils.data.Dataset):
    """PyTorch Dataset for loading CHB-MIT EEG data.
    
    Attributes:
        root (str): Root directory containing the processed data.
        files (List[str]): List of filenames to use.
        default_rate (int): Default sampling rate of the data.
        sampling_rate (int): Target sampling rate for resampling.
        logger (logging.Logger): Logger for this class.
    """
    
    def __init__(self, root: str, files: List[str], sampling_rate: int = 200, log_dir: Optional[str] = None):
        """Initialize the CHB-MIT dataset.
        
        Args:
            root: Root directory containing the processed data.
            files: List of filenames to use.
            sampling_rate: Target sampling rate for the data.
            log_dir: Optional directory for log files. If None, logs to console only.
        """
        self.root = root
        self.files = files
        self.default_rate = 256
        self.sampling_rate = sampling_rate
        self.logger = logging.getLogger(__name__ + ".CHBMITLoader")
        
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
            file_handler = logging.FileHandler(os.path.join(log_dir, "chbmit_dataset.log"))
            file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
            self.logger.addHandler(file_handler)
        
        self.logger.info(f"Initialized CHBMITLoader with {len(files)} files from {root}")
        self.logger.info(f"Using sampling rate: {sampling_rate} Hz (default: {self.default_rate} Hz)")

    def __len__(self) -> int:
        """Return the number of samples in the dataset.
        
        Returns:
            Number of samples in the dataset.
        """
        return len(self.files)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, Union[int, float]]:
        """Get a sample from the dataset.
        
        Args:
            index: Index of the sample to retrieve.
            
        Returns:
            Tuple containing the EEG data and its label.
        """
        try:
            file_path = os.path.join(self.root, self.files[index])
            sample = pickle.load(open(file_path, "rb"))
            X = sample["X"]
            
            # Resample from 256Hz to target rate if needed
            if self.sampling_rate != self.default_rate:
                X = resample(X, 10 * self.sampling_rate, axis=-1)
                
            # Normalize the signal using 95th percentile
            X = X / (
                np.quantile(np.abs(X), q=0.95, method="linear", axis=-1, keepdims=True)
                + 1e-8
            )
            
            Y = sample["y"]
            X = torch.FloatTensor(X)
            return X, Y
            
        except Exception as e:
            self.logger.error(f"Error loading file {self.files[index]}: {str(e)}")
            raise


class PTBLoader(torch.utils.data.Dataset):
    """PyTorch Dataset for loading PTB ECG data.
    
    Attributes:
        root (str): Root directory containing the processed data.
        files (List[str]): List of filenames to use.
        default_rate (int): Default sampling rate of the data.
        sampling_rate (int): Target sampling rate for resampling.
        logger (logging.Logger): Logger for this class.
    """
    
    def __init__(self, root: str, files: List[str], sampling_rate: int = 500, log_dir: Optional[str] = None):
        """Initialize the PTB dataset.
        
        Args:
            root: Root directory containing the processed data.
            files: List of filenames to use.
            sampling_rate: Target sampling rate for the data.
            log_dir: Optional directory for log files. If None, logs to console only.
        """
        self.root = root
        self.files = files
        self.default_rate = 500
        self.sampling_rate = sampling_rate
        self.logger = logging.getLogger(__name__ + ".PTBLoader")
        
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
            file_handler = logging.FileHandler(os.path.join(log_dir, "ptb_dataset.log"))
            file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
            self.logger.addHandler(file_handler)
        
        self.logger.info(f"Initialized PTBLoader with {len(files)} files from {root}")
        self.logger.info(f"Using sampling rate: {sampling_rate} Hz (default: {self.default_rate} Hz)")

    def __len__(self) -> int:
        """Return the number of samples in the dataset.
        
        Returns:
            Number of samples in the dataset.
        """
        return len(self.files)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, Union[int, float]]:
        """Get a sample from the dataset.
        
        Args:
            index: Index of the sample to retrieve.
            
        Returns:
            Tuple containing the ECG data and its label.
        """
        try:
            file_path = os.path.join(self.root, self.files[index])
            sample = pickle.load(open(file_path, "rb"))
            X = sample["X"]
            
            # Resample if needed
            if self.sampling_rate != self.default_rate:
                X = resample(X, self.sampling_rate * 5, axis=-1)
                
            # Normalize the signal using 95th percentile
            X = X / (
                np.quantile(np.abs(X), q=0.95, method="linear", axis=-1, keepdims=True)
                + 1e-8
            )
            
            Y = sample["y"]
            X = torch.FloatTensor(X)
            return X, Y
            
        except Exception as e:
            self.logger.error(f"Error loading file {self.files[index]}: {str(e)}")
            raise


class TUEVLoader(torch.utils.data.Dataset):
    """PyTorch Dataset for loading TUH EEG (TUEV) data.
    
    Attributes:
        root (str): Root directory containing the processed data.
        files (List[str]): List of filenames to use.
        default_rate (int): Default sampling rate of the data.
        sampling_rate (int): Target sampling rate for resampling.
        logger (logging.Logger): Logger for this class.
    """
    
    def __init__(self, root: str, files: List[str], sampling_rate: int = 200, log_dir: Optional[str] = None):
        """Initialize the TUEV dataset.
        
        Args:
            root: Root directory containing the processed data.
            files: List of filenames to use.
            sampling_rate: Target sampling rate for the data.
            log_dir: Optional directory for log files. If None, logs to console only.
        """
        self.root = root
        self.files = files
        self.default_rate = 256
        self.sampling_rate = sampling_rate
        self.logger = logging.getLogger(__name__ + ".TUEVLoader")
        
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
            file_handler = logging.FileHandler(os.path.join(log_dir, "tuev_dataset.log"))
            file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
            self.logger.addHandler(file_handler)
        
        self.logger.info(f"Initialized TUEVLoader with {len(files)} files from {root}")
        self.logger.info(f"Using sampling rate: {sampling_rate} Hz (default: {self.default_rate} Hz)")

    def __len__(self) -> int:
        """Return the number of samples in the dataset.
        
        Returns:
            Number of samples in the dataset.
        """
        return len(self.files)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        """Get a sample from the dataset.
        
        Args:
            index: Index of the sample to retrieve.
            
        Returns:
            Tuple containing the EEG data and its label.
        """
        try:
            file_path = os.path.join(self.root, self.files[index])
            sample = pickle.load(open(file_path, "rb"))
            X = sample["signal"]
            
            # Resample from 256Hz to target rate if needed
            if self.sampling_rate != self.default_rate:
                X = resample(X, 5 * self.sampling_rate, axis=-1)
                
            # Normalize the signal using 95th percentile
            X = X / (
                np.quantile(np.abs(X), q=0.95, method="linear", axis=-1, keepdims=True)
                + 1e-8
            )
            
            Y = int(sample["label"][0] - 1)
            X = torch.FloatTensor(X)
            return X, Y
            
        except Exception as e:
            self.logger.error(f"Error loading file {self.files[index]}: {str(e)}")
            raise


class HARLoader(torch.utils.data.Dataset):
    """PyTorch Dataset for loading Human Activity Recognition (HAR) data.
    
    Attributes:
        list_IDs: List of file IDs to use.
        dir: Directory containing the data.
        label_map: Mapping of activity labels.
        default_rate (int): Default sampling rate of the data.
        sampling_rate (int): Target sampling rate for resampling.
        logger (logging.Logger): Logger for this class.
    """
    
    def __init__(self, dir: str, list_IDs: List[str], sampling_rate: int = 50, log_dir: Optional[str] = None):
        """Initialize the HAR dataset.
        
        Args:
            dir: Directory containing the data.
            list_IDs: List of file IDs to use.
            sampling_rate: Target sampling rate for the data.
            log_dir: Optional directory for log files. If None, logs to console only.
        """
        self.list_IDs = list_IDs
        self.dir = dir
        self.label_map = ["1", "2", "3", "4", "5", "6"]
        self.default_rate = 50
        self.sampling_rate = sampling_rate
        self.logger = logging.getLogger(__name__ + ".HARLoader")
        
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
            file_handler = logging.FileHandler(os.path.join(log_dir, "har_dataset.log"))
            file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
            self.logger.addHandler(file_handler)
        
        self.logger.info(f"Initialized HARLoader with {len(list_IDs)} files from {dir}")
        self.logger.info(f"Using sampling rate: {sampling_rate} Hz (default: {self.default_rate} Hz)")

    def __len__(self) -> int:
        """Return the number of samples in the dataset.
        
        Returns:
            Number of samples in the dataset.
        """
        return len(self.list_IDs)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        """Get a sample from the dataset.
        
        Args:
            index: Index of the sample to retrieve.
            
        Returns:
            Tuple containing the HAR data and its label.
        """
        try:
            path = os.path.join(self.dir, self.list_IDs[index])
            sample = pickle.load(open(path, "rb"))
            X, y = sample["X"], self.label_map.index(sample["y"])
            
            # Resample if needed
            if self.sampling_rate != self.default_rate:
                X = resample(X, int(2.56 * self.sampling_rate), axis=-1)
                
            # Normalize the signal using 95th percentile
            X = X / (
                np.quantile(
                    np.abs(X), q=0.95, interpolation="linear", axis=-1, keepdims=True
                )
                + 1e-8
            )
            
            return torch.FloatTensor(X), y
            
        except Exception as e:
            self.logger.error(f"Error loading file {self.list_IDs[index]}: {str(e)}")
            raise


class UnsupervisedPretrainLoader(torch.utils.data.Dataset):
    """PyTorch Dataset for loading unsupervised pretraining data from PREST and SHHS datasets.
    
    Attributes:
        root_prest (str): Root directory containing PREST data.
        root_shhs (str): Root directory containing SHHS data.
        prest_list (List[str]): List of PREST files to use.
        shhs_list (List[str]): List of SHHS files to use.
        prest_idx_all: Array of all indices for PREST data segmentation.
        prest_mask_idx_N: Number of mask indices for PREST data.
        shhs_idx_all: Array of all indices for SHHS data segmentation.
        shhs_mask_idx_N: Number of mask indices for SHHS data.
        logger (logging.Logger): Logger for this class.
    """
    
    def __init__(self, root_prest: str, root_shhs: str, log_dir: Optional[str] = None):
        """Initialize the unsupervised pretraining dataset.
        
        Args:
            root_prest: Root directory containing PREST data.
            root_shhs: Root directory containing SHHS data.
            log_dir: Optional directory for log files. If None, logs to console only.
        """
        self.logger = logging.getLogger(__name__ + ".UnsupervisedPretrainLoader")
        
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
            file_handler = logging.FileHandler(os.path.join(log_dir, "unsupervised_pretrain.log"))
            file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
            self.logger.addHandler(file_handler)

        # PREST dataset
        self.root_prest = root_prest
        exception_files = ["319431_data.npy"]
        self.prest_list = list(
            filter(
                lambda x: ("data" in x) and (x not in exception_files),
                os.listdir(self.root_prest),
            )
        )

        PREST_LENGTH = 2000
        WINDOW_SIZE = 200

        self.logger.info(f"(PREST) Unlabeled data size: {len(self.prest_list) * 16}")
        self.prest_idx_all = np.arange(PREST_LENGTH // WINDOW_SIZE)
        self.prest_mask_idx_N = PREST_LENGTH // WINDOW_SIZE // 3

        SHHS_LENGTH = 6000
        # SHHS dataset
        self.root_shhs = root_shhs
        self.shhs_list = os.listdir(self.root_shhs)
        self.logger.info(f"(SHHS) Unlabeled data size: {len(self.shhs_list)}")
        self.shhs_idx_all = np.arange(SHHS_LENGTH // WINDOW_SIZE)
        self.shhs_mask_idx_N = SHHS_LENGTH // WINDOW_SIZE // 5

    def __len__(self) -> int:
        """Return the number of samples in the dataset.
        
        Returns:
            Number of samples in the dataset.
        """
        return len(self.prest_list) + len(self.shhs_list)

    def prest_load(self, index: int) -> Tuple[torch.Tensor, int]:
        """Load data from the PREST dataset.
        
        Args:
            index: Index of the sample to retrieve.
            
        Returns:
            Tuple containing the data and a flag (0 for PREST)
        """
        try:
            sample_path = self.prest_list[index]
            # (16, 16, 2000), 10s
            samples = np.load(os.path.join(self.root_prest, sample_path)).astype("float32")

            # Find all zeros or all 500 signals and then remove them
            samples_max = np.max(samples, axis=(1, 2))
            samples_min = np.min(samples, axis=(1, 2))
            valid = np.where((samples_max > 0) & (samples_min < 0))[0]
            valid = np.random.choice(valid, min(8, len(valid)), replace=False)
            samples = samples[valid]

            # Normalize samples (remove the amplitude)
            samples = samples / (
                np.quantile(
                    np.abs(samples), q=0.95, method="linear", axis=-1, keepdims=True
                )
                + 1e-8
            )
            samples = torch.FloatTensor(samples)
            return samples, 0
            
        except Exception as e:
            self.logger.error(f"Error loading PREST file {self.prest_list[index]}: {str(e)}")
            raise

    def shhs_load(self, index: int) -> Tuple[torch.Tensor, int]:
        """Load data from the SHHS dataset.
        
        Args:
            index: Index of the sample to retrieve.
            
        Returns:
            Tuple containing the data and a flag (1 for SHHS)
        """
        try:
            sample_path = self.shhs_list[index]
            # (2, 3750) sampled at 125
            sample = pickle.load(open(os.path.join(self.root_shhs, sample_path), "rb"))
            # (2, 6000) resample to 200
            samples = resample(sample, 6000, axis=-1)

            # Normalize samples (remove the amplitude)
            samples = samples / (
                np.quantile(
                    np.abs(samples), q=0.95, method="linear", axis=-1, keepdims=True
                )
                + 1e-8
            )
            # Generate samples and targets and mask_indices
            samples = torch.FloatTensor(samples)

            return samples, 1
            
        except Exception as e:
            self.logger.error(f"Error loading SHHS file {self.shhs_list[index]}: {str(e)}")
            raise

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        """Get a sample from the dataset.
        
        Args:
            index: Index of the sample to retrieve.
            
        Returns:
            Tuple containing the data and a flag (0 for PREST, 1 for SHHS)
        """
        if index < len(self.prest_list):
            return self.prest_load(index)
        else:
            index = index - len(self.prest_list)
            return self.shhs_load(index)


def collate_fn_unsupervised_pretrain(batch):
    """Collate function for unsupervised pretraining data.
    
    Args:
        batch: Batch of data samples.
        
    Returns:
        Tuple of PREST and SHHS samples.
    """
    prest_samples, shhs_samples = [], []
    for sample, flag in batch:
        if flag == 0:
            prest_samples.append(sample)
        else:
            shhs_samples.append(sample)

    shhs_samples = torch.stack(shhs_samples, 0)
    if len(prest_samples) > 0:
        prest_samples = torch.cat(prest_samples, 0)
        return prest_samples, shhs_samples
    return 0, shhs_samples


class EEGSupervisedPretrainLoader(torch.utils.data.Dataset):
    """PyTorch Dataset for loading supervised pretraining data from multiple EEG datasets.
    
    Attributes:
        tuev_root (str): Root directory for TUEV data.
        tuev_files (List[str]): List of TUEV files.
        tuev_size (int): Number of TUEV samples.
        chb_mit_root (str): Root directory for CHB-MIT data.
        chb_mit_files (List[str]): List of CHB-MIT files.
        chb_mit_size (int): Number of CHB-MIT samples.
        iiic_x: IIIC data features.
        iiic_y: IIIC data labels.
        iiic_size (int): Number of IIIC samples.
        tuab_root (str): Root directory for TUAB data.
        tuab_files (List[str]): List of TUAB files.
        tuab_size (int): Number of TUAB samples.
        logger (logging.Logger): Logger for this class.
    """
    
    def __init__(self, tuev_data, chb_mit_data, iiic_data, tuab_data, log_dir: Optional[str] = None):
        """Initialize the supervised pretraining dataset.
        
        Args:
            tuev_data: Tuple of (root_dir, files) for TUEV dataset.
            chb_mit_data: Tuple of (root_dir, files) for CHB-MIT dataset.
            iiic_data: Tuple of (features, labels) for IIIC dataset.
            tuab_data: Tuple of (root_dir, files) for TUAB dataset.
            log_dir: Optional directory for log files. If None, logs to console only.
        """
        self.logger = logging.getLogger(__name__ + ".EEGSupervisedPretrainLoader")
        
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
            file_handler = logging.FileHandler(os.path.join(log_dir, "supervised_pretrain.log"))
            file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
            self.logger.addHandler(file_handler)
        
        # For TUEV
        tuev_root, tuev_files = tuev_data
        self.tuev_root = tuev_root
        self.tuev_files = tuev_files
        self.tuev_size = len(self.tuev_files)

        # For CHB-MIT
        chb_mit_root, chb_mit_files = chb_mit_data
        self.chb_mit_root = chb_mit_root
        self.chb_mit_files = chb_mit_files
        self.chb_mit_size = len(self.chb_mit_files)

        # For IIIC seizure
        iiic_x, iiic_y = iiic_data
        self.iiic_x = iiic_x
        self.iiic_y = iiic_y
        self.iiic_size = len(self.iiic_x)

        # For TUAB
        tuab_root, tuab_files = tuab_data
        self.tuab_root = tuab_root
        self.tuab_files = tuab_files
        self.tuab_size = len(self.tuab_files)
        
        self.logger.info(f"Initialized with TUEV: {self.tuev_size}, CHB-MIT: {self.chb_mit_size}, "
                         f"IIIC: {self.iiic_size}, TUAB: {self.tuab_size} samples")

    def __len__(self) -> int:
        """Return the number of samples in the dataset.
        
        Returns:
            Number of samples in the dataset.
        """
        return self.tuev_size + self.chb_mit_size + self.iiic_size + self.tuab_size

    def tuev_load(self, index: int) -> Tuple[torch.Tensor, int, int]:
        """Load data from the TUEV dataset.
        
        Args:
            index: Index of the sample to retrieve.
            
        Returns:
            Tuple containing the data, label, and a flag (0 for TUEV)
        """
        try:
            sample = pickle.load(
                open(os.path.join(self.tuev_root, self.tuev_files[index]), "rb")
            )
            X = sample["signal"]
            # 256 * 5 -> 1000
            X = resample(X, 1000, axis=-1)
            X = X / (
                np.quantile(np.abs(X), q=0.95, method="linear", axis=-1, keepdims=True)
                + 1e-8
            )
            Y = int(sample["label"][0] - 1)
            X = torch.FloatTensor(X)
            return X, Y, 0
            
        except Exception as e:
            self.logger.error(f"Error loading TUEV file {self.tuev_files[index]}: {str(e)}")
            raise

    def chb_mit_load(self, index: int) -> Tuple[torch.Tensor, Union[int, float], int]:
        """Load data from the CHB-MIT dataset.
        
        Args:
            index: Index of the sample to retrieve.
            
        Returns:
            Tuple containing the data, label, and a flag (1 for CHB-MIT)
        """
        try:
            sample = pickle.load(
                open(os.path.join(self.chb_mit_root, self.chb_mit_files[index]), "rb")
            )
            X = sample["X"]
            # 2560 -> 2000
            X = resample(X, 2000, axis=-1)
            X = X / (
                np.quantile(np.abs(X), q=0.95, method="linear", axis=-1, keepdims=True)
                + 1e-8
            )
            Y = sample["y"]
            X = torch.FloatTensor(X)
            return X, Y, 1
            
        except Exception as e:
            self.logger.error(f"Error loading CHB-MIT file {self.chb_mit_files[index]}: {str(e)}")
            raise

    def iiic_load(self, index: int) -> Tuple[torch.Tensor, int, int]:
        """Load data from the IIIC dataset.
        
        Args:
            index: Index of the sample to retrieve.
            
        Returns:
            Tuple containing the data, label, and a flag (2 for IIIC)
        """
        try:
            data = self.iiic_x[index]
            samples = torch.FloatTensor(data)
            samples = samples / (
                torch.quantile(torch.abs(samples), q=0.95, dim=-1, keepdim=True) + 1e-8
            )
            y = np.argmax(self.iiic_y[index])
            return samples, y, 2
            
        except Exception as e:
            self.logger.error(f"Error loading IIIC sample at index {index}: {str(e)}")
            raise

    def tuab_load(self, index: int) -> Tuple[torch.Tensor, Union[int, float], int]:
        """Load data from the TUAB dataset.
        
        Args:
            index: Index of the sample to retrieve.
            
        Returns:
            Tuple containing the data, label, and a flag (3 for TUAB)
        """
        try:
            sample = pickle.load(
                open(os.path.join(self.tuab_root, self.tuab_files[index]), "rb")
            )
            X = sample["X"]
            X = X / (
                np.quantile(np.abs(X), q=0.95, method="linear", axis=-1, keepdims=True)
                + 1e-8
            )
            Y = sample["y"]
            X = torch.FloatTensor(X)
            return X, Y, 3
            
        except Exception as e:
            self.logger.error(f"Error loading TUAB file {self.tuab_files[index]}: {str(e)}")
            raise

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, Union[int, float], int]:
        """Get a sample from the dataset.
        
        Args:
            index: Index of the sample to retrieve.
            
        Returns:
            Tuple containing the data, label, and a flag indicating the source dataset
        """
        if index < self.tuev_size:
            return self.tuev_load(index)
        elif index < self.tuev_size + self.chb_mit_size:
            index = index - self.tuev_size
            return self.chb_mit_load(index)
        elif index < self.tuev_size + self.chb_mit_size + self.iiic_size:
            index = index - self.tuev_size - self.chb_mit_size
            return self.iiic_load(index)
        elif (
            index < self.tuev_size + self.chb_mit_size + self.iiic_size + self.tuab_size
        ):
            index = index - self.tuev_size - self.chb_mit_size - self.iiic_size
            return self.tuab_load(index)
        else:
            error_msg = f"Index {index} out of range (total size: {self.__len__()})"
            self.logger.error(error_msg)
            raise ValueError(error_msg)


def collate_fn_supervised_pretrain(batch):
    """Collate function for supervised pretraining data.
    
    Args:
        batch: Batch of data samples.
        
    Returns:
        Tuple of (TUEV, IIIC, CHB-MIT, TUAB) data and labels.
    """
    tuev_samples, tuev_labels = [], []
    iiic_samples, iiic_labels = [], []
    chb_mit_samples, chb_mit_labels = [], []
    tuab_samples, tuab_labels = [], []

    for sample, labels, idx in batch:
        if idx == 0:
            tuev_samples.append(sample)
            tuev_labels.append(labels)
        elif idx == 1:
            iiic_samples.append(sample)
            iiic_labels.append(labels)
        elif idx == 2:
            chb_mit_samples.append(sample)
            chb_mit_labels.append(labels)
        elif idx == 3:
            tuab_samples.append(sample)
            tuab_labels.append(labels)
        else:
            raise ValueError(f"Invalid idx {idx} in batch")

    if len(tuev_samples) > 0:
        tuev_samples = torch.stack(tuev_samples)
        tuev_labels = torch.LongTensor(tuev_labels)
    if len(iiic_samples) > 0:
        iiic_samples = torch.stack(iiic_samples)
        iiic_labels = torch.LongTensor(iiic_labels)
    if len(chb_mit_samples) > 0:
        chb_mit_samples = torch.stack(chb_mit_samples)
        chb_mit_labels = torch.LongTensor(chb_mit_labels)
    if len(tuab_samples) > 0:
        tuab_samples = torch.stack(tuab_samples)
        tuab_labels = torch.LongTensor(tuab_labels)

    return (
        (tuev_samples, tuev_labels),
        (iiic_samples, iiic_labels),
        (chb_mit_samples, chb_mit_labels),
        (tuab_samples, tuab_labels),
    )


def focal_loss(y_hat: torch.Tensor, y: torch.Tensor, alpha: float = 0.8, gamma: float = 0.7) -> torch.Tensor:
    """Calculate focal loss for binary classification.
    
    Focal loss puts more weight on hard, misclassified examples.
    
    Args:
        y_hat: Predicted values (N, 1)
        y: True values (N, 1)
        alpha: Balance parameter (0-1)
        gamma: Focusing parameter (≥ 0)
        
    Returns:
        Calculated focal loss
    """
    y_hat = y_hat.view(-1, 1)
    y = y.view(-1, 1)
    p = torch.sigmoid(y_hat)
    loss = -alpha * (1 - p) ** gamma * y * torch.log(p) - (1 - alpha) * p**gamma * (
        1 - y
    ) * torch.log(1 - p)
    return loss.mean()


def BCE(y_hat: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Calculate binary cross-entropy loss.
    
    This implementation is numerically stable.
    
    Args:
        y_hat: Predicted values (N, 1)
        y: True values (N, 1)
        
    Returns:
        Calculated BCE loss
    """
    y_hat = y_hat.view(-1, 1)
    y = y.view(-1, 1)
    loss = (
        -y * y_hat
        + torch.log(1 + torch.exp(-torch.abs(y_hat)))
        + torch.max(y_hat, torch.zeros_like(y_hat))
    )
    return loss.mean()