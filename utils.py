import logging
from pathlib import Path
import pickle
import torch
import torch.nn.functional as F
import numpy as np
import os
from scipy.signal import resample
from collections import defaultdict, OrderedDict
from tqdm import tqdm
from typing import List, Tuple, Optional, Dict, Union

class MEGDataset(torch.utils.data.Dataset):
    """PyTorch Dataset for loading preprocessed MEG data.

    This dataset loads preprocessed MEG data from disk, stored as PyTorch tensors.
    Each data sample contains the MEG signal and its associated label.

    Attributes:
        root: Root directory containing the processed data.
        files: List of filenames to use.
        class_weights: Optional weights for each class to handle imbalance.
        cache_size: Number of samples to cache in memory (0 for no caching).
        metadata_only: If True, only load metadata without loading full tensors.
    """

    def __init__(self,
                 root: str,
                 files: List[str],
                 class_weights: Optional[Dict[int, float]] = None,
                 cache_size: int = 0,
                 metadata_only: bool = False):
        """Initialize the MEG dataset.

        Args:
            root: Root directory containing the processed data.
            files: List of filenames to use.
            class_weights: Optional dictionary mapping class indices to weights.
            cache_size: Number of samples to cache in memory (0 for no caching).
            metadata_only: If True, only load metadata without loading full tensors.
        """
        self.root = root
        self.files = files
        self.class_weights = class_weights
        self.cache_size = cache_size
        self.metadata_only = metadata_only
        self.custom_logger = logging.getLogger(__name__)

        # Initialize cache
        self.cache = OrderedDict()

        # Preload metadata if requested
        self.metadata = {}
        if self.metadata_only:
            self.preload_metadata()

    def preload_metadata(self) -> None:
        """Preload metadata for all files to enable efficient querying."""
        self.custom_logger.info(f"Preloading metadata for {len(self.files)} files...")
        for idx, filename in enumerate(tqdm(self.files, desc="Loading metadata")):
            file_path = Path(self.root, filename)
            try:
                # Use torch.load with map_location='cpu' for better compatibility
                sample = torch.load(file_path, map_location='cpu', weights_only=False)
                # Extract only metadata (exclude large tensors)
                meta = {k: v for k, v in sample.items() if k != 'data'}
                self.metadata[idx] = meta
            except Exception as e:
                self.custom_logger.warning(f"Error loading metadata for {file_path}: {str(e)}")
                self.metadata[idx] = None

        self.custom_logger.info(f"Metadata preloaded for {len(self.metadata)} files")

    def __len__(self) -> int:
        """Return the number of samples in the dataset.

        Returns:
            Number of samples in the dataset.
        """
        return len(self.files)

    def __getitem__(self, idx: int) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """Get a sample from the dataset."""
        # Check cache first
        if idx in self.cache:
            sample = self.cache[idx]
        else:
            # Load MEG data
            file_path = Path(self.root, self.files[idx])
            try:
                sample = torch.load(file_path, map_location='cpu', weights_only=False)

                # Check if required keys exist
                if 'data' not in sample or 'label' not in sample:
                    self.custom_logger.warning(f"File {file_path} missing required keys. Found: {sample.keys()}")
                    raise ValueError("Missing required keys in sample")

                # Update cache if enabled
                if self.cache_size > 0:
                    self._update_cache(idx, sample)

            except Exception as e:
                self.custom_logger.error(f"Error loading file {file_path}: {str(e)}")
                raise RuntimeError(f"Error loading file {file_path}: {str(e)}")

        data, label = sample['data'], sample['label']

        # convert to tensor if not already
        if not torch.is_tensor(data):
            data = torch.tensor(data, dtype=torch.float32)
        if not torch.is_tensor(label):
            label = torch.tensor(label, dtype=torch.long)

        # always return (data, labels) tuple as (n_segments, n_channels, n_samples_per_segment), (n_segments)
        return data, label

    def _update_cache(self, idx: int, sample: Dict[str, torch.Tensor]) -> None:
        """Update the sample cache using LRU strategy.

        Args:
            idx: Index of the sample.
            sample: Sample data.
        """
        # Add to cache
        self.cache[idx] = sample

        # Remove the oldest if exceeding cache size
        while len(self.cache) > self.cache_size:
            self.cache.popitem(last=False)  # Remove oldest item

    def get_patient_ids(self) -> List[str]:
        """Get a list of unique patient IDs in the dataset.

        Returns:
            List of unique patient IDs.
        """
        if self.metadata_only and self.metadata:
            patient_ids = set()
            for meta in self.metadata.values():
                if meta and 'patient_id' in meta:
                    patient_ids.add(meta['patient_id'])
            return sorted(list(patient_ids))

        # Fall back to loading from files
        patient_ids = set()
        for idx in range(len(self)):
            file_path = Path(self.root, self.files[idx])
            try:
                sample = torch.load(file_path, map_location='cpu', weights_only=False)
                if 'patient_id' in sample:
                    patient_ids.add(sample['patient_id'])
            except (FileNotFoundError, RuntimeError) as e:
                self.custom_logger.warning(f"Error loading {file_path}: {e}")

        return sorted(list(patient_ids))

    def get_class_distribution(self) -> Dict[int, int]:
        """Get distribution of classes in the dataset.

        Returns:
            Dictionary mapping class labels to counts.
        """
        if self.metadata_only and self.metadata:
            distribution = defaultdict(int)
            for meta in self.metadata.values():
                if meta and 'label' in meta:
                    label = meta['label'].item()
                    distribution[label] += 1
            return dict(distribution)

        # Process in batches for better efficiency
        distribution = defaultdict(int)
        batch_size = 100

        for batch_start in range(0, len(self), batch_size):
            batch_end = min(batch_start + batch_size, len(self))

            for idx in range(batch_start, batch_end):
                file_path = Path(self.root, self.files[idx])
                try:
                    # Just load the label instead of the full sample
                    sample = torch.load(file_path, map_location='cpu', weights_only=False)
                    label = sample['label'].item()
                    distribution[label] += 1
                except Exception as e:
                    self.custom_logger.warning(f"Error loading {file_path}: {e}")

        return dict(distribution)

    def get_samples_by_patient(self, patient_id: str) -> List[int]:
        """Get indices of samples belonging to a specific patient.

        Args:
            patient_id: ID of the patient.

        Returns:
            List of sample indices.
        """
        if self.metadata_only and self.metadata:
            return [idx for idx, meta in self.metadata.items()
                    if meta and 'patient_id' in meta and meta['patient_id'] == patient_id]

        # Fall back to loading from files
        indices = []
        for idx in range(len(self)):
            file_path = Path(self.root, self.files[idx])
            try:
                sample = torch.load(file_path, map_location='cpu', weights_only=False)
                if sample.get('patient_id') == patient_id:
                    indices.append(idx)
            except (FileNotFoundError, RuntimeError) as e:
                self.custom_logger.warning(f"Error loading {file_path}: {e}")
        return indices

    @classmethod
    def calculate_class_weights(cls, files_list: List[str], root_dir: str) -> Dict[int, float]:
        """Calculate class weights based on class distribution.

        Args:
            files_list: List of data file paths
            root_dir: Root directory containing the files

        Returns:
            Dictionary mapping class indices to weights
        """
        # Count occurrences of each class
        class_counts = defaultdict(int)
        custom_logger = logging.getLogger(__name__)

        for file_path in tqdm(files_list, desc="Calculating class weights"):
            full_path = os.path.join(root_dir, file_path)
            try:
                sample = torch.load(full_path, map_location='cpu', weights_only=False)
                label = sample['label'].item()
                class_counts[label] += 1
            except Exception as e:
                custom_logger.warning(f"Error loading {file_path} for class weight calculation: {e}")

        # Calculate weights as inverse of frequency
        total_samples = sum(class_counts.values())
        class_weights = {}

        for label, count in class_counts.items():
            # Inverse frequency weighting
            class_weights[label] = total_samples / (len(class_counts) * count)

        custom_logger.info(f"Calculated class weights: {class_weights}")
        return class_weights

    @classmethod
    def from_processed_dir(cls, processed_dir: str, split: str = 'train',
                           use_weights: bool = True, cache_size: int = 0,
                           metadata_only: bool = False) -> 'MEGDataset':
        """Create a dataset from a processed directory.

        Args:
            processed_dir: Path to the processed data directory.
            split: Data split to use ('train', 'val', or 'test').
            use_weights: Whether to use class weights.
            cache_size: Number of samples to cache in memory.
            metadata_only: Whether to only load metadata.

        Returns:
            Configured MEGDataset instance.

        Raises:
            FileNotFoundError: If the file list for the specified split is not found.
        """
        custom_logger = logging.getLogger(__name__)

        # Load file list
        file_list_path = os.path.join(processed_dir, f"{split}_files.pkl")
        if not os.path.exists(file_list_path):
            raise FileNotFoundError(f"File list not found at {file_list_path}")

        with open(file_list_path, 'rb') as f:
            files_list = pickle.load(f)

        custom_logger.info(f"Loaded {len(files_list)} files for {split} split")

        # Load class weights if requested
        class_weights = None
        if use_weights:
            weights_path = os.path.join(processed_dir, "class_weights.pkl")
            if os.path.exists(weights_path):
                with open(weights_path, 'rb') as f:
                    all_weights = pickle.load(f)
                class_weights = all_weights.get(split)
                custom_logger.info(f"Loaded class weights for {split}: {class_weights}")
            else:
                custom_logger.warning(f"Class weights file not found at {weights_path}")

        # Create dataset
        return cls(
            root=os.path.join(processed_dir, split),
            files=files_list,
            class_weights=class_weights,
            cache_size=cache_size,
            metadata_only=metadata_only
        )


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


def focal_loss(y_hat: torch.Tensor, y: torch.Tensor, alpha: float = 0.25, gamma: float = 2.0) -> torch.Tensor:
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


def focal_loss_with_class_weights(y_hat: torch.Tensor, y: torch.Tensor, class_weights: Dict[int, float],
                                  gamma: float = 2.0) -> torch.Tensor:
    """Enhanced focal loss that uses class-specific weights.
    
    Args:
        y_hat: Predicted values (N, 1)
        y: True values (N, 1)
        class_weights: Dictionary mapping class indices to weights
        gamma: Focusing parameter (≥ 0)
        
    Returns:
        Calculated focal loss with class weights
    """
    y_hat = y_hat.view(-1, 1)
    y = y.view(-1, 1)
    p = torch.sigmoid(y_hat)
    
    # Apply class-specific weights
    weights = torch.ones_like(y, dtype=torch.float32)
    weights[y == 0] = class_weights[0]  # Weight for non-spike class
    weights[y == 1] = class_weights[1]  # Weight for spike class
    
    # Standard focal loss formula with per-sample weights
    pt = p * y + (1 - p) * (1 - y)  # Get p_t for correct class
    focal_weight = (1 - pt) ** gamma
    
    # Apply both class weights and focal weights
    loss = -weights * focal_weight * torch.log(pt + 1e-12)  # Add epsilon for stability
    
    return loss.mean()


def weighted_BCE(y_hat: torch.Tensor, y: torch.Tensor, class_weights: Dict[int, float]) -> torch.Tensor:
    """Calculate weighted binary cross-entropy loss with logits."""
    # Reshape from 
    # - y_hat: (batch_size, n_segments, 1), y: (batch_size, n_segments) to 
    # - y_hat: (batch_size * n_segments), y: (batch_size * n_segments)
    y_hat = y_hat.view(-1)
    y = y.view(-1).float()  # Ensure targets are float for BCE
    
    # Create weight tensor for sample weighting
    weights = torch.ones_like(y, dtype=torch.float32)
    weights[y == 0] = class_weights[0]
    weights[y == 1] = class_weights[1]
    
    # Use PyTorch's built-in function that handles logits properly
    loss = torch.nn.functional.binary_cross_entropy_with_logits(
        y_hat, y, weight=weights, reduction='none'
    )
    
    return loss.mean()


def BCE(y_hat: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Binary cross-entropy loss with logits."""
    # Reshape from 
    # - y_hat: (batch_size, n_segments, 1), y: (batch_size, n_segments) to 
    # - y_hat: (batch_size * n_segments), y: (batch_size * n_segments)
    y_hat = y_hat.view(-1)
    y = y.view(-1).float()  # Ensure targets are float for BCE
    
    # Use PyTorch's built-in function that handles logits properly
    loss = torch.nn.functional.binary_cross_entropy_with_logits(
        y_hat, y, reduction='none'
    )
    
    return loss.mean()


def temporal_consistency_loss(
    y_hat: torch.Tensor,
    y: torch.Tensor,
    consistency_weight: float = 0.3,
    smoothness_window: int = 3
) -> torch.Tensor:
    """Calculate temporal consistency loss for spike detection.
    
    This loss encourages:
    1. Similar predictions for adjacent segments with the same label
    2. Smooth transitions even when labels change (avoiding sudden jumps)
    3. Gradual onset/offset for detected spikes
    
    Args:
        y_hat: Predicted values of shape (batch_size, n_segments, 1)
        y: True values of shape (batch_size, n_segments)
        consistency_weight: Weight for temporal consistency term
        smoothness_window: Window size for calculating smoothness
        
    Returns:
        Temporal consistency loss
    """
    batch_size, n_segments, _ = y_hat.shape
    device = y_hat.device
    
    # Create empty loss
    consistency_loss = torch.tensor(0.0, device=device)
    
    # Skip if we have only one segment or consistency weight is zero
    if n_segments <= 1 or consistency_weight <= 0:
        return consistency_loss
    
    # Get probabilities from logits
    probs = torch.sigmoid(y_hat).view(batch_size, n_segments)
    y = y.float()
    
    # 1. Label consistency loss: adjacent segments with same label should have similar predictions
    for i in range(n_segments - 1):
        # Check if adjacent segments have the same label
        same_label = (y[:, i] == y[:, i + 1]).float()
        
        # Difference between predictions for adjacent segments
        pred_diff = torch.abs(probs[:, i] - probs[:, i + 1])
        
        # Penalize differences when labels are the same
        label_consistency = (pred_diff * same_label).mean()
        consistency_loss = consistency_loss + label_consistency
    
    # 2. Smoothness loss: avoid sudden changes in predictions
    if n_segments >= smoothness_window:
        for i in range(n_segments - smoothness_window + 1):
            # Get window of predictions
            window = probs[:, i:i+smoothness_window]
            
            # Calculate variance within the window
            window_mean = window.mean(dim=1, keepdim=True)
            window_var = ((window - window_mean) ** 2).mean(dim=1)
            
            # Add to total loss
            consistency_loss = consistency_loss + window_var.mean()
    
    # 3. Transitional smoothness: for spike onset/offset
    if n_segments >= 3:
        for i in range(1, n_segments - 1):
            # Check for transition points (label changes)
            is_transition = ((y[:, i-1] != y[:, i]) | (y[:, i] != y[:, i+1])).float()
            
            # Calculate mean squared error of second derivative (measure of "jerkiness")
            second_deriv = torch.abs(probs[:, i+1] - 2*probs[:, i] + probs[:, i-1])
            
            # Penalize non-smooth transitions at label change points
            transition_loss = (second_deriv * is_transition).mean()
            consistency_loss = consistency_loss + transition_loss
    
    # Normalize by number of segments
    consistency_loss = consistency_loss / n_segments
    
    return consistency_weight * consistency_loss


def enhanced_focal_loss(
    y_hat: torch.Tensor,
    y: torch.Tensor,
    class_weights: Dict[int, float],
    gamma: float = 2.0,
    alpha: float = 0.25,
    temporal_consistency: bool = True,
    consistency_weight: float = 0.3
) -> torch.Tensor:
    """Enhanced focal loss with temporal consistency for spike detection.
    
    Combines focal loss (which focuses on hard examples) with a temporal consistency
    term that encourages smooth and consistent predictions across segments.
    
    Args:
        y_hat: Predicted values (batch_size, n_segments, 1)
        y: True values (batch_size, n_segments)
        class_weights: Dictionary mapping class indices to weights
        gamma: Focusing parameter for focal loss
        alpha: Balance parameter for focal loss
        temporal_consistency: Whether to add temporal consistency term
        consistency_weight: Weight for temporal consistency term
        
    Returns:
        Combined loss value
    """
    batch_size, n_segments, _ = y_hat.shape
    
    # Flatten predictions and targets for focal loss calculation
    y_flat = y.reshape(-1, 1).float()  # (batch_size * n_segments, 1)
    y_hat_flat = y_hat.reshape(-1, 1)  # (batch_size * n_segments, 1)
    
    # Create weight tensor based on class weights
    weights = torch.ones_like(y_flat, device=y_hat.device)
    weights[y_flat == 0] = class_weights[0]  # Weight for non-spike
    weights[y_flat == 1] = class_weights[1]  # Weight for spike
    
    # Get probabilities
    p = torch.sigmoid(y_hat_flat)
    
    # Calculate pt (probability of the correct class)
    pt = p * y_flat + (1 - p) * (1 - y_flat)
    
    # Focal weight based on how hard the example is
    focal_weight = (1 - pt) ** gamma
    
    # Alpha weighting for class balance
    alpha_weight = alpha * y_flat + (1 - alpha) * (1 - y_flat)
    
    # Combine all weights
    combined_weight = weights * alpha_weight * focal_weight
    
    # Binary cross-entropy
    bce = -torch.log(pt + 1e-12)  # Add epsilon for numerical stability
    
    # Calculate weighted focal loss
    focal_loss_val = (combined_weight * bce).mean()
    
    # Add temporal consistency loss if requested
    if temporal_consistency and n_segments > 1:
        consistency_loss = temporal_consistency_loss(
            y_hat, y, 
            consistency_weight=consistency_weight
        )
        return focal_loss_val + consistency_loss
    else:
        return focal_loss_val


def spike_aware_loss(
    y_hat: torch.Tensor, 
    y: torch.Tensor,
    class_weights: Dict[int, float],
    onset_weight: float = 2.0,
    offset_weight: float = 1.5,
    base_weight: float = 1.0,
    temporal_weight: float = 0.3
) -> torch.Tensor:
    """Special loss function for spike detection that emphasizes onset and offset detection.
    
    This loss function:
    1. Applies higher weights to spike onset/offset points to improve boundary detection
    2. Applies class weights to handle imbalance
    3. Adds temporal consistency to encourage smooth predictions
    
    Args:
        y_hat: Predicted values (batch_size, n_segments, 1)
        y: True values (batch_size, n_segments)
        class_weights: Dictionary mapping class indices to weights
        onset_weight: Weight multiplier for spike onset points
        offset_weight: Weight multiplier for spike offset points
        base_weight: Base weight for non-transition points
        temporal_weight: Weight for temporal consistency term
        
    Returns:
        Combined loss value
    """
    batch_size, n_segments, _ = y_hat.shape
    device = y_hat.device
    
    # Apply sigmoid to get probabilities
    probs = torch.sigmoid(y_hat).view(batch_size, n_segments)
    y = y.float()
    
    # Create weight tensor based on class weights
    weights = torch.ones(batch_size, n_segments, device=device)
    weights[y == 0] = class_weights[0] * base_weight
    weights[y == 1] = class_weights[1] * base_weight
    
    # Find spike onset and offset points (transitions from 0->1 and 1->0)
    if n_segments > 1:
        for i in range(1, n_segments):
            # Onset: previous is 0, current is 1
            onset_mask = (y[:, i-1] == 0) & (y[:, i] == 1)
            # Offset: previous is 1, current is 0
            offset_mask = (y[:, i-1] == 1) & (y[:, i] == 0)
            
            # Apply higher weights to these points
            weights[onset_mask, i] *= onset_weight
            weights[offset_mask, i] *= offset_weight
    
    # Calculate BCE loss
    bce = F.binary_cross_entropy(probs, y, reduction='none')
    
    # Apply weights
    weighted_bce = (weights * bce).mean()
    
    # Add temporal consistency
    consistency_loss = temporal_consistency_loss(
        y_hat, y, 
        consistency_weight=temporal_weight
    )
    
    return weighted_bce + consistency_loss


def hierarchical_spike_loss(
    y_hat: torch.Tensor,
    y: torch.Tensor,
    class_weights: Optional[Dict[int, float]] = None,
    sequence_weight: float = 0.4
) -> torch.Tensor:
    """Hierarchical loss that considers both segment-level and sequence-level spike detection.
    
    This loss operates on two levels:
    1. Segment-level: Standard binary classification of each segment
    2. Sequence-level: Detection of spike sequences in the chunk
    
    Args:
        y_hat: Predicted values (batch_size, n_segments, 1)
        y: True values (batch_size, n_segments)
        class_weights: Optional dictionary mapping class indices to weights
        sequence_weight: Weight for sequence-level component
        
    Returns:
        Combined loss value
    """
    batch_size, n_segments, _ = y_hat.shape
    device = y_hat.device
    
    # Default class weights if not provided
    if class_weights is None:
        class_weights = {0: 1.0, 1: 1.0}
    
    # 1. Segment-level component: enhanced focal loss
    segment_loss = enhanced_focal_loss(
        y_hat, y, 
        class_weights=class_weights,
        temporal_consistency=True
    )
    
    # 2. Sequence-level component: detect if the chunk contains any spikes
    # Aggregate predictions across segments
    seg_probs = torch.sigmoid(y_hat).view(batch_size, n_segments)
    
    # Create chunk-level target: 1 if any segment has a spike, 0 otherwise
    chunk_target = (y.sum(dim=1) > 0).float().view(batch_size, 1)
    
    # Create chunk-level prediction: max probability across segments
    chunk_pred = seg_probs.max(dim=1)[0].view(batch_size, 1)
    
    # Calculate chunk-level loss
    chunk_loss = F.binary_cross_entropy(chunk_pred, chunk_target)
    
    # Combine losses
    total_loss = (1 - sequence_weight) * segment_loss + sequence_weight * chunk_loss
    
    return total_loss

def relaxed_scores(true, pred):
    """
    Calculate relaxed metrics allowing detections to be 1 window away from the ground truth.

    Args:
        true: Array of ground truth labels
        pred: Array of predictions

    Returns:
        Dictionary containing relaxed metrics
    """
    relaxed_pred = np.zeros_like(pred)

    for ind, l in enumerate(true):
        # Left boundary condition
        if ind == 0:
            if l == 1:
                if (pred[ind] == 1 or pred[ind + 1] == 1):
                    relaxed_pred[ind] = 1
            elif pred[ind] == 1 and true[ind + 1] == 0:
                relaxed_pred[ind] = 1
        # Right boundary condition
        elif ind == true.shape[0] - 1:
            if l == 1:
                if (pred[ind] == 1 or pred[ind - 1] == 1):
                    relaxed_pred[ind] = 1
            elif pred[ind] == 1 and true[ind - 1] == 0:
                relaxed_pred[ind] = 1
                # General case
        elif l == 1:
            if (pred[ind - 1] == 1 or pred[ind] == 1 or pred[ind + 1] == 1):
                relaxed_pred[ind] = 1
        elif pred[ind] == 1 and true[ind + 1] == 0 and true[ind - 1] == 0:
            relaxed_pred[ind] = 1

    # Calculate relaxed metrics
    relaxed_tp = len(np.intersect1d(np.where(true == 1), np.where(relaxed_pred == 1)))
    relaxed_tn = len(np.intersect1d(np.where(true == 0), np.where(relaxed_pred == 0)))
    relaxed_fp = len(np.intersect1d(np.where(true == 0), np.where(relaxed_pred == 1)))
    relaxed_fn = len(np.intersect1d(np.where(true == 1), np.where(relaxed_pred == 0)))

    # Calculate metrics, handling edge cases
    relaxed_f1 = (2 * relaxed_tp) / (2 * relaxed_tp + relaxed_fp + relaxed_fn) if (
                2 * relaxed_tp + relaxed_fp + relaxed_fn) else 1
    relaxed_precision = (relaxed_tp / (relaxed_tp + relaxed_fp)) if (relaxed_tp + relaxed_fp) else 1
    relaxed_recall = relaxed_sens = (relaxed_tp / (relaxed_tp + relaxed_fn)) if (relaxed_tp + relaxed_fn) else 1
    relaxed_spec = (relaxed_tn / (relaxed_tn + relaxed_fp)) if (relaxed_tn + relaxed_fp) else 1
    relaxed_acc = (relaxed_tp + relaxed_tn) / (relaxed_tp + relaxed_tn + relaxed_fp + relaxed_fn)

    # Create a dictionary of relaxed metrics
    relaxed_metrics = {
        "relaxed_f1": relaxed_f1,
        "relaxed_precision": relaxed_precision,
        "relaxed_recall": relaxed_recall,
        "relaxed_sensitivity": relaxed_sens,
        "relaxed_specificity": relaxed_spec,
        "relaxed_accuracy": relaxed_acc,
        "relaxed_tp": relaxed_tp,
        "relaxed_tn": relaxed_tn,
        "relaxed_fp": relaxed_fp,
        "relaxed_fn": relaxed_fn
    }

    return relaxed_metrics




def relaxed_scores(true, pred):
    """
    Calculate relaxed metrics allowing detections to be 1 window away from the ground truth.

    Args:
        true: Array of ground truth labels
        pred: Array of predictions

    Returns:
        Dictionary containing relaxed metrics
    """
    relaxed_pred = np.zeros_like(pred)

    for ind, l in enumerate(true):
        # Left boundary condition
        if ind == 0:
            if l == 1:
                if (pred[ind] == 1 or pred[ind + 1] == 1):
                    relaxed_pred[ind] = 1
            elif pred[ind] == 1 and true[ind + 1] == 0:
                relaxed_pred[ind] = 1
        # Right boundary condition
        elif ind == true.shape[0] - 1:
            if l == 1:
                if (pred[ind] == 1 or pred[ind - 1] == 1):
                    relaxed_pred[ind] = 1
            elif pred[ind] == 1 and true[ind - 1] == 0:
                relaxed_pred[ind] = 1
                # General case
        elif l == 1:
            if (pred[ind - 1] == 1 or pred[ind] == 1 or pred[ind + 1] == 1):
                relaxed_pred[ind] = 1
        elif pred[ind] == 1 and true[ind + 1] == 0 and true[ind - 1] == 0:
            relaxed_pred[ind] = 1

    # Calculate relaxed metrics
    relaxed_tp = len(np.intersect1d(np.where(true == 1), np.where(relaxed_pred == 1)))
    relaxed_tn = len(np.intersect1d(np.where(true == 0), np.where(relaxed_pred == 0)))
    relaxed_fp = len(np.intersect1d(np.where(true == 0), np.where(relaxed_pred == 1)))
    relaxed_fn = len(np.intersect1d(np.where(true == 1), np.where(relaxed_pred == 0)))

    # Calculate metrics, handling edge cases
    relaxed_f1 = (2 * relaxed_tp) / (2 * relaxed_tp + relaxed_fp + relaxed_fn) if (
                2 * relaxed_tp + relaxed_fp + relaxed_fn) else 1
    relaxed_precision = (relaxed_tp / (relaxed_tp + relaxed_fp)) if (relaxed_tp + relaxed_fp) else 1
    relaxed_recall = relaxed_sens = (relaxed_tp / (relaxed_tp + relaxed_fn)) if (relaxed_tp + relaxed_fn) else 1
    relaxed_spec = (relaxed_tn / (relaxed_tn + relaxed_fp)) if (relaxed_tn + relaxed_fp) else 1
    relaxed_acc = (relaxed_tp + relaxed_tn) / (relaxed_tp + relaxed_tn + relaxed_fp + relaxed_fn)

    # Create a dictionary of relaxed metrics
    relaxed_metrics = {
        "relaxed_f1": relaxed_f1,
        "relaxed_precision": relaxed_precision,
        "relaxed_recall": relaxed_recall,
        "relaxed_sensitivity": relaxed_sens,
        "relaxed_specificity": relaxed_spec,
        "relaxed_accuracy": relaxed_acc,
        "relaxed_tp": relaxed_tp,
        "relaxed_tn": relaxed_tn,
        "relaxed_fp": relaxed_fp,
        "relaxed_fn": relaxed_fn
    }

    return relaxed_metrics

