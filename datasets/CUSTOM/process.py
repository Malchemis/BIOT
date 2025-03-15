import mne
import numpy as np
import os
import pickle
import torch
import logging
import yaml
import time
import json
import re
from pathlib import Path
from typing import List, Tuple, Dict, Optional, Any, Union
from tqdm import tqdm
from collections import defaultdict
import matplotlib.pyplot as plt
import multiprocessing as mp
import warnings

from mne_interpolate import interpolate_missing_channels


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
        self.cache = {}
        self.cache_keys = []
        
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
                sample = torch.load(file_path, map_location='cpu')
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
        """Get a sample from the dataset.

        Args:
            idx: Index of the sample to retrieve.

        Returns:
            Tuple containing the MEG data and its label (and optionally weight).
        """
        # Check cache first
        if idx in self.cache:
            sample = self.cache[idx]
        else:
            # Load MEG data
            file_path = Path(self.root, self.files[idx])
            try:
                # Use torch.load with map_location='cpu' for better compatibility
                sample = torch.load(file_path, map_location='cpu')
                
                # Check if required keys exist in the sample
                if 'data' not in sample or 'label' not in sample:
                    self.custom_logger.warning(f"File {file_path} missing required keys. Found keys: {sample.keys()}")
                    # Return zero tensors as fallback
                    return torch.zeros((1, 100)), torch.tensor(0)
                
                # Update cache if enabled
                if self.cache_size > 0:
                    self._update_cache(idx, sample)
                    
            except Exception as e:
                self.custom_logger.error(f"Error loading file {file_path}: {str(e)}")
                # Return zero tensors as fallback
                return torch.zeros((1, 100)), torch.tensor(0)
        
        data, label = sample['data'], sample['label']
        
        # If class weights are provided, return weight as well
        if self.class_weights is not None:
            label_value = label.item()
            weight = torch.tensor(self.class_weights.get(label_value, 1.0), dtype=torch.float32)
            return data, label, weight
        
        return data, label

    def _update_cache(self, idx: int, sample: Dict[str, torch.Tensor]) -> None:
        """Update the sample cache using LRU strategy.

        Args:
            idx: Index of the sample.
            sample: Sample data.
        """
        # Add to cache
        self.cache[idx] = sample
        self.cache_keys.append(idx)
        
        # Remove oldest if exceeding cache size
        if len(self.cache) > self.cache_size:
            oldest_idx = self.cache_keys.pop(0)
            del self.cache[oldest_idx]

    def get_patient_ids(self) -> List[str]:
        """Get list of unique patient IDs in the dataset.

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
                sample = torch.load(file_path, map_location='cpu')
                if 'patient_id' in sample:
                    patient_ids.add(sample['patient_id'])
            except Exception:
                pass
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
        
        # Fall back to loading from files
        distribution = defaultdict(int)
        for idx in tqdm(range(len(self)), desc="Calculating class distribution"):
            _, label = self[idx]
            distribution[label.item()] += 1
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
                sample = torch.load(file_path, map_location='cpu')
                if sample.get('patient_id') == patient_id:
                    indices.append(idx)
            except Exception:
                pass
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
                sample = torch.load(full_path, map_location='cpu')
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


class MEGPreprocessor:
    """Preprocessor for MEG data.

    This preprocessor:
    - Loads .ds files from CTF MEG recordings
    - Removes bad segments
    - Preprocesses the data (resampling, filtering, normalization)
    - Cuts the data into clips of specified length with overlap
    - Labels clips using annotations (spikes vs non-spikes)
    - Saves the processed data for use with PyTorch
    - Splits data by patient to avoid data leakage
    - Provides balanced sampling and data augmentation options

    Attributes:
        root_dir: Directory containing the .ds files.
        output_dir: Directory to save the processed data.
        good_channels_file_path: Path to file with good channels.
        loc_meg_channels_file_path: Path to file with channel locations.
        clip_length_s: Length of each clip in seconds.
        overlap: Fraction of overlap between consecutive clips.
        sampling_rate: Target sampling rate in Hz.
        l_freq: Lower frequency for bandpass filter.
        h_freq: Higher frequency for bandpass filter.
        notch_freq: Frequency for notch filter.
        train_ratio: Ratio of patients to use for training.
        val_ratio: Ratio of patients to use for validation.
        random_state: Random seed for reproducibility.
        min_spikes_per_file: Minimum number of spikes required to process a file.
        process_no_spike_files: Whether to process files with no valid spikes (for control data).
        skip_files: List of filenames to skip during processing.
        target_class_ratio: Target ratio of spike to non-spike samples.
        balance_method: Method to use for balancing classes.
        augmentation: Configuration for augmentation.
        normalization: Configuration for signal normalization.
        n_jobs: Number of parallel jobs to use.
        max_segments_per_file: Maximum number of segments to extract per file.
        online_saving: Whether to save segments immediately after processing.
        annotation_rules: Rules for extracting spike annotations.
        benchmark: Whether to collect timing statistics.
        channel_stats: Whether to collect channel statistics.
    """

    def __init__(
            self,
            root_dir: str,
            output_dir: str,
            good_channels_file_path: str,
            loc_meg_channels_file_path: str,
            clip_length_s: float = 1.0,
            overlap: float = 0.5,
            sampling_rate: int = 200,
            l_freq: float = 1.0,
            h_freq: float = 70.0,
            notch_freq: float = 50.0,
            train_ratio: float = 0.8,
            val_ratio: float = 0.2,
            random_state: int = 42,
            min_spikes_per_file: int = 10,
            process_no_spike_files: bool = False,
            skip_files: List[str] = None,
            target_class_ratio: float = 0.5,            # 0.5 means balanced 1:1 ratio
            balance_method: str = 'undersample',        # 'undersample', 'oversample', or 'none'
            augmentation: Dict[str, Any] = None,
            normalization: Dict[str, Any] = None,
            n_jobs: int = 1,
            max_segments_per_file: int = None,          # Maximum number of segments to extract per file
            online_saving: bool = True,                 # Save segments immediately after processing
            annotation_rules: Optional[Dict[str, Any]] = None,  # Rules for spike annotation extraction
            benchmark: bool = True,                     # Collect timing statistics
            channel_stats: bool = True,                 # Collect channel statistics
            logger: Optional[logging.Logger] = None
    ):
        """Initialize the MEG preprocessor.

        Args:
            root_dir: Directory containing the .ds files.
            output_dir: Directory to save the processed data.
            good_channels_file_path: Path to the file containing the list of good channels.
            loc_meg_channels_file_path: Path to the file containing the channel locations.
            clip_length_s: Length of each clip in seconds.
            overlap: Fraction of overlap between consecutive clips (0.0 to < 1.0).
            sampling_rate: Target sampling rate in Hz.
            l_freq: Lower frequency for bandpass filter.
            h_freq: Higher frequency for bandpass filter.
            notch_freq: Frequency for notch filter.
            train_ratio: Ratio of patients to use for training.
            val_ratio: Ratio of patients to use for validation.
            random_state: Random seed for reproducibility.
            min_spikes_per_file: Minimum number of spikes required to process a file.
            process_no_spike_files: Whether to process files with no valid spikes.
            skip_files: List of filenames or patterns to skip during processing.
            target_class_ratio: Target ratio of spike to non-spike segments (0.5 means balanced).
            balance_method: Method to use for balancing classes ('undersample', 'oversample', 'none').
            augmentation: Configuration for augmentation.
            normalization: Configuration for signal normalization.
            n_jobs: Number of parallel jobs to use.
            max_segments_per_file: Maximum number of segments to extract per file.
            online_saving: Whether to save segments immediately after processing.
            annotation_rules: Rules for extracting spike annotations.
            benchmark: Whether to collect timing statistics.
            channel_stats: Whether to collect channel statistics.
            logger: Logger instance for this class.

        Raises:
            ValueError: If the overlap is invalid or if the split ratios don't sum to 1.
            FileNotFoundError: If the good channels or channel locations files are not found.
        """
        self.root_dir = root_dir
        self.output_dir = output_dir
        self.clip_length_s = clip_length_s
        self.overlap = overlap
        self.sampling_rate = sampling_rate
        self.l_freq = l_freq
        self.h_freq = h_freq
        self.notch_freq = notch_freq
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.random_state = random_state
        self.min_spikes_per_file = min_spikes_per_file
        self.process_no_spike_files = process_no_spike_files
        self.skip_files = skip_files or []
        self.target_class_ratio = target_class_ratio
        self.balance_method = balance_method
        self.n_jobs = n_jobs
        self.max_segments_per_file = max_segments_per_file
        self.online_saving = online_saving
        self.benchmark = benchmark
        self.channel_stats = channel_stats

        # Set up logging
        self.logger = logger or logging.getLogger(__name__)

        # Check if ratios sum to 1
        if abs(self.train_ratio + self.val_ratio - 1.0) > 1e-7:
            error_msg = f"Train and val ratios must sum to 1. Got {self.train_ratio + self.val_ratio}"
            self.logger.error(error_msg)
            raise ValueError(error_msg)

        # Calculate clip length in samples
        self.clip_length_samples = int(self.clip_length_s * self.sampling_rate)

        # Calculate step size in samples (accounting for overlap)
        self.step_size = int(self.clip_length_samples * (1 - self.overlap))
        if self.step_size <= 0:
            error_msg = "Invalid overlap value. Must be less than 1.0"
            self.logger.error(error_msg)
            raise ValueError(error_msg)

        # Validate augmentation parameters if provided
        if augmentation and augmentation.get('enabled', False):
            self._validate_augmentation_params(augmentation)

        # Default augmentation settings
        self.augmentation = {
            'enabled': False,
            'time_shift_ms': 50,        # Maximum time shift in milliseconds
            'max_shifts': 2,            # Number of shifts to generate per spike
            'preserve_spikes': True,    # Ensure spikes remain visible after shifting
            'amplitude_scale': False,   # Whether to scale amplitude
            'amplitude_range': (0.8, 1.2),  # Range for amplitude scaling
            'add_noise': False,         # Whether to add noise
            'noise_level': 0.05,        # Level of noise to add (relative to signal std)
            'channel_dropout': False,   # Whether to randomly zero out channels
            'channel_dropout_rate': 0.1 # Fraction of channels to zero out
        }
        if augmentation:
            self.augmentation.update(augmentation)

        # Default normalization settings
        self.normalization = {
            'method': 'percentile',  # 'percentile', 'zscore', 'minmax'
            'percentile': 95,        # If using percentile method
            'per_channel': True,     # Apply normalization per channel
            'per_segment': False,    # Apply normalization per segment instead of whole recording
        }
        if normalization:
            self.normalization.update(normalization)
            
        # Default annotation rules (can be overridden)
        self.default_annotation_rules = {
            'Holdout': {
                'include': ['jj_add', 'JJ_add', 'jj_valid', 'JJ_valid'],
                'exclude': []
            },
            'IterativeLearningFeedback1': {
                'include': [],
                'exclude': ['detected_spike']
            },
            'IterativeLearningFeedback2': {
                'include': [],
                'exclude': ['detected_spike']
            },
            'Default': {  # For ILF3-9 and others
                'include': [],
                'exclude': ['true_spike', 'detected_spike']
            }
        }
        
        # Override default rules if provided
        self.annotation_rules = self.default_annotation_rules
        if annotation_rules:
            for group, rules in annotation_rules.items():
                if group in self.annotation_rules:
                    self.annotation_rules[group].update(rules)
                else:
                    self.annotation_rules[group] = rules

        # Load the list of good channels
        if not os.path.exists(good_channels_file_path):
            raise FileNotFoundError(f"Good channels file not found: {good_channels_file_path}")
            
        self.logger.info(f"Loading good channels from {good_channels_file_path}")
        try:
            with open(good_channels_file_path, 'rb') as f:
                self.good_channels = pickle.load(f)
        except Exception as e:
            raise ValueError(f"Failed to load good channels: {str(e)}")

        # Load the channel locations
        if not os.path.exists(loc_meg_channels_file_path):
            raise FileNotFoundError(f"Channel locations file not found: {loc_meg_channels_file_path}")
            
        self.logger.info(f"Loading channel locations from {loc_meg_channels_file_path}")
        try:
            with open(loc_meg_channels_file_path, 'rb') as f:
                self.loc_meg_channels = pickle.load(f)
        except Exception as e:
            raise ValueError(f"Failed to load channel locations: {str(e)}")

        # Create output directories
        for split in ['train', 'val', 'test']:
            split_dir = os.path.join(output_dir, split)
            os.makedirs(split_dir, exist_ok=True)
            self.logger.debug(f"Created output directory: {split_dir}")

        # Compile skip file patterns for faster matching
        self.skip_patterns = [re.compile(pattern) for pattern in self.skip_files]

        # Keep track of processed files
        self.processed_files = {'train': [], 'val': [], 'test': []}
        
        # Keep track of which patient goes into which split
        self.patient_splits = {}
        
        # Store information about patient groups (Holdout, IterativeLearningFeedback1-9)
        self.patient_groups = {}
        
        # For tracking metadata about the dataset
        self.dataset_metadata = {
            'config': self._get_config_dict(),
            'class_distribution': {'train': {0: 0, 1: 0}, 'val': {0: 0, 1: 0}, 'test': {0: 0, 1: 0}},
            'patients_summary': {},
            'files_summary': {},
            'processing_stats': {
                'total_files': 0,
                'processed_files': 0,
                'skipped_files': 0,
                'skipped_low_spikes': 0,
                'skipped_pattern': 0,
                'no_spike_files_processed': 0,
                'augmented_segments': 0,
            }
        }
        
        # For benchmarking
        if self.benchmark:
            self.timing_stats = {
                'per_file': {},
                'total_time': 0,
                'file_count': 0,
                'segment_count': 0
            }
            
        # For channel statistics
        if self.channel_stats:
            self.channels_info = {
                'correlation_with_spikes': {},  # Channel correlation with spike presence
                'variance_during_spikes': {},   # Channel variance during spikes
                'most_informative': [],         # Ranked list of most informative channels
                'per_patient': {}               # Per-patient statistics
            }

    def _validate_augmentation_params(self, augmentation: Dict[str, Any]) -> None:
        """Validate augmentation parameters against clip length.

        Args:
            augmentation: Augmentation configuration dictionary.

        Raises:
            ValueError: If augmentation parameters are invalid.
        """
        if not augmentation.get('enabled', False):
            return
            
        time_shift_ms = augmentation.get('time_shift_ms', 50)
        max_shifts = augmentation.get('max_shifts', 2)
        
        # Check if time shift is too large for clip length
        max_shift_samples = int(time_shift_ms * self.sampling_rate / 1000)
        if max_shift_samples >= self.clip_length_samples // 2:
            warning_msg = (
                f"Warning: time_shift_ms ({time_shift_ms}ms) is too large for the clip length "
                f"({self.clip_length_s * 1000}ms). This may shift spikes out of the segment. "
                f"Consider reducing to at most {self.clip_length_s * 500}ms."
            )
            self.logger.warning(warning_msg)
            warnings.warn(warning_msg)
            
        # Check if max_shifts is too large
        if max_shifts > 3:
            warning_msg = (
                f"Warning: max_shifts ({max_shifts}) is quite high and might create redundant data. "
                f"Consider reducing to 2-3 shifts per spike segment."
            )
            self.logger.warning(warning_msg)
            warnings.warn(warning_msg)
            
        # Estimate resulting dataset size after augmentation
        approx_increase = max_shifts
        warning_msg = (
            f"Augmentation will increase dataset size by approximately {approx_increase}x "
            f"for spike segments."
        )
        self.logger.info(warning_msg)

    def _get_config_dict(self) -> Dict[str, Any]:
        """Create a dictionary of configuration parameters for metadata.

        Returns:
            Dictionary of configuration parameters
        """
        return {
            'clip_length_s': self.clip_length_s,
            'overlap': self.overlap,
            'sampling_rate': self.sampling_rate,
            'l_freq': self.l_freq,
            'h_freq': self.h_freq,
            'notch_freq': self.notch_freq,
            'min_spikes_per_file': self.min_spikes_per_file,
            'process_no_spike_files': self.process_no_spike_files,
            'target_class_ratio': self.target_class_ratio,
            'balance_method': self.balance_method,
            'augmentation': self.augmentation,
            'normalization': self.normalization,
            'annotation_rules': self.annotation_rules,
            'online_saving': self.online_saving
        }

    def _should_skip_file(self, file_path: str) -> bool:
        """Check if a file should be skipped based on skip patterns.

        Args:
            file_path: Path to the file

        Returns:
            True if the file should be skipped, False otherwise
        """
        filename = os.path.basename(file_path)
        
        # Check against each skip pattern
        for pattern in self.skip_patterns:
            if pattern.search(filename) or pattern.search(file_path):
                self.logger.info(f"Skipping file {filename} - matches skip pattern {pattern.pattern}")
                self.dataset_metadata['processing_stats']['skipped_pattern'] += 1
                return True
                
        return False

    def find_ds_files(self) -> List[Dict[str, str]]:
        """Find all .ds directories in the hierarchical directory structure.

        Returns:
            List of dictionaries containing paths to .ds directories, patient IDs, and groups.
        """
        ds_files = []
        self.logger.debug(f"Searching for .ds files in {self.root_dir}")
        
        # Look for Holdout and IterativeLearningFeedback* directories
        for group_dir in os.listdir(self.root_dir):
            group_path = os.path.join(self.root_dir, group_dir)
            
            # Skip if not a directory
            if not os.path.isdir(group_path):
                continue
                
            # Process each patient directory within the group
            for patient_dir in os.listdir(group_path):
                patient_path = os.path.join(group_path, patient_dir)
                
                # Skip if not a directory
                if not os.path.isdir(patient_path):
                    continue
                    
                # Look for .ds files within the patient directory
                found_ds = False
                for item in os.listdir(patient_path):
                    if item.endswith('.ds') and os.path.isdir(os.path.join(patient_path, item)):
                        ds_path = os.path.join(patient_path, item)
                        
                        # Check if file should be skipped
                        if self._should_skip_file(ds_path):
                            continue
                            
                        ds_files.append({
                            'path': ds_path,
                            'patient_id': patient_dir,
                            'group': group_dir,
                            'filename': item
                        })
                        found_ds = True
                        
                        # Store patient's group information
                        self.patient_groups[patient_dir] = group_dir
                
                if not found_ds:
                    self.logger.warning(f"No .ds files found for patient {patient_dir} in group {group_dir}")

        self.dataset_metadata['processing_stats']['total_files'] = len(ds_files)
        self.logger.info(f"Found {len(ds_files)} .ds files across {len(self.patient_groups)} patients")
        return ds_files

    def split_patients(self) -> Dict[str, List[str]]:
        """Split patients into train, validation, and test sets.

        This ensures that all data from one patient stays in the same split,
        with the exception of validation which can overlap with train.
        All Holdout patients are assigned to the test set.

        Returns:
            Dictionary mapping split names to lists of patient IDs.
        """
        ds_files = self.find_ds_files()
        
        # Group patients by their group (Holdout or IterativeLearningFeedback)
        patients_by_group = defaultdict(set)
        for file_info in ds_files:
            patients_by_group[file_info['group']].add(file_info['patient_id'])
            
        # All Holdout patients go to the test set
        test_patients = list(patients_by_group.get('Holdout', set()))
        self.logger.info(f"Assigned {len(test_patients)} Holdout patients to test set")
        
        # Pool all IterativeLearningFeedback patients for train/val split
        feedback_patients = []
        for group, patients in patients_by_group.items():
            if group != 'Holdout':
                feedback_patients.extend(list(patients))
                
        # Remove duplicates
        feedback_patients = list(set(feedback_patients))
        
        # Shuffle remaining patients for train/val split
        np.random.seed(self.random_state)
        np.random.shuffle(feedback_patients)
        
        # Calculate number for train
        n_patients = len(feedback_patients)
        n_train = int(n_patients * self.train_ratio / (self.train_ratio + self.val_ratio))
        
        # Split patients
        train_patients = feedback_patients[:n_train]
        # For validation, we can include all non-test patients that are left
        val_patients = feedback_patients[n_train:]

        # Store patient splits
        for patient in train_patients:
            self.patient_splits[patient] = 'train'
        for patient in val_patients:
            self.patient_splits[patient] = 'val'
        for patient in test_patients:
            self.patient_splits[patient] = 'test'

        # Store patient counts in metadata
        self.dataset_metadata['patients_summary'] = {
            'train': len(train_patients),
            'val': len(val_patients),
            'test': len(test_patients),
            'total': len(train_patients) + len(val_patients) + len(test_patients)
        }

        self.logger.info(f"Patient split: {len(train_patients)} train, {len(val_patients)} val, {len(test_patients)} test")
        return {'train': train_patients, 'val': val_patients, 'test': test_patients}

    def _get_spike_annotations(self, annotations, group: str) -> List[float]:
        """Extract spike annotations based on the group's annotation rules.

        Args:
            annotations: The MNE annotations object
            group: The dataset group (Holdout or IterativeLearningFeedback*)

        Returns:
            List of spike onset times in seconds
        """
        spike_onsets = []
        seen_onsets = set()  # Track onsets to avoid duplicates
        
        # Extract descriptions and onsets
        descriptions = annotations.description
        onsets = annotations.onset
        
        # Determine which rules to use
        if group in self.annotation_rules:
            rules = self.annotation_rules[group]
        else:
            # Use default rules
            rules = self.annotation_rules['Default']
            
        include_patterns = rules.get('include', [])
        exclude_patterns = rules.get('exclude', [])
        
        # Compile patterns for faster matching
        include_regex = [re.compile(pattern.lower()) for pattern in include_patterns]
        exclude_regex = [re.compile(pattern.lower()) for pattern in exclude_patterns]
        
        for i in range(len(annotations)):
            description = descriptions[i].lower()
            onset = onsets[i]
            
            # Skip if we've already seen this onset (within a small time window)
            duplicate = False
            for seen_onset in seen_onsets:
                if abs(onset - seen_onset) < 0.01:  # 10ms window for duplicates
                    duplicate = True
                    break
                    
            if duplicate:
                continue
                
            # Check if description matches include patterns
            included = False
            # if there is no include pattern, include all
            if len(include_regex) == 0:
                included = True
            else:
                for pattern in include_regex:
                    if pattern.search(description):
                        included = True
                        break
                    
            # Check if description matches exclude patterns
            excluded = False
            for pattern in exclude_regex:
                if pattern.search(description):
                    excluded = True
                    break
                    
            # Add onset if included and not excluded
            if included and not excluded:
                spike_onsets.append(onset)
                seen_onsets.add(onset)
        
        return spike_onsets

    def _normalize_data(self, data: np.ndarray) -> np.ndarray:
        """Normalize the data according to the specified method.

        Args:
            data: Data array of shape (n_channels, n_samples)

        Returns:
            Normalized data
        """
        norm_method = self.normalization['method']
        per_channel = self.normalization['per_channel']
        per_segment = self.normalization['per_segment']
        
        # Determine axes for normalization
        if per_channel:
            # Normalize each channel separately (axis=1 means across time dimension)
            axis = 1
        else:
            # Normalize across all channels and time points
            axis = None
            
        # Apply normalization method
        if norm_method == 'percentile':
            percentile = self.normalization['percentile']
            # Calculate normalization factor
            q_val = np.percentile(np.abs(data), percentile, axis=axis, keepdims=True)
            # Avoid division by zero
            normalized = data / (q_val + 1e-8)
            
        elif norm_method == 'zscore':
            # Z-score normalization: (x - mean) / std
            mean = np.mean(data, axis=axis, keepdims=True)
            std = np.std(data, axis=axis, keepdims=True)
            normalized = (data - mean) / (std + 1e-8)
            
        elif norm_method == 'minmax':
            # Min-max normalization: (x - min) / (max - min)
            min_val = np.min(data, axis=axis, keepdims=True)
            max_val = np.max(data, axis=axis, keepdims=True)
            normalized = (data - min_val) / (max_val - min_val + 1e-8)
            
        else:
            # Default to no normalization
            self.logger.warning(f"Unknown normalization method: {norm_method}, using raw data")
            normalized = data
            
        return normalized
        
    def _calculate_channel_statistics(self, data: np.ndarray, labels: List[int], 
                                      patient_id: str) -> Dict[str, Any]:
        """Calculate channel statistics for spike detection.

        Args:
            data: MEG data array of shape (n_channels, n_samples)
            labels: Labels for each segment (1 for spike, 0 for non-spike)
            patient_id: ID of the patient

        Returns:
            Dictionary with channel statistics
        """
        if not self.channel_stats:
            return {}
            
        try:
            n_channels = data.shape[0]
            
            # Convert labels to numpy array and get spike/non-spike indices
            label_array = np.array(labels)
            spike_indices = np.where(label_array == 1)[0]
            non_spike_indices = np.where(label_array == 0)[0]
            
            if len(spike_indices) == 0 or len(non_spike_indices) == 0:
                return {}
                
            # Calculate correlation between each channel and spike presence
            channel_correlations = {}
            channel_variances = {}
            
            for ch_idx in range(n_channels):
                # Calculate variance during spikes vs. non-spikes
                if len(spike_indices) > 0:
                    spike_variance = np.var(data[ch_idx, spike_indices])
                    non_spike_variance = np.var(data[ch_idx, non_spike_indices])
                    variance_ratio = spike_variance / (non_spike_variance + 1e-10)
                    channel_variances[ch_idx] = variance_ratio
                
                # Calculate correlation with spike presence
                # Use point-biserial correlation (correlation between continuous and binary variables)
                # This is equivalent to Pearson correlation when one variable is binary
                channel_data = data[ch_idx, :]
                corr = np.corrcoef(channel_data, label_array)[0, 1]
                channel_correlations[ch_idx] = corr
            
            # Store per-patient statistics
            patient_stats = {
                'correlations': channel_correlations,
                'variances': channel_variances,
                'n_segments': len(labels),
                'n_spikes': len(spike_indices)
            }
            
            # Update global statistics
            if patient_id not in self.channels_info['per_patient']:
                self.channels_info['per_patient'][patient_id] = patient_stats
            else:
                # Combine with existing statistics
                existing = self.channels_info['per_patient'][patient_id]
                n_total = existing['n_segments'] + patient_stats['n_segments']
                
                # Weighted average of correlations
                for ch_idx in channel_correlations:
                    if ch_idx in existing['correlations']:
                        weight1 = existing['n_segments'] / n_total
                        weight2 = patient_stats['n_segments'] / n_total
                        existing['correlations'][ch_idx] = (
                            weight1 * existing['correlations'][ch_idx] + 
                            weight2 * patient_stats['correlations'][ch_idx]
                        )
                    else:
                        existing['correlations'][ch_idx] = patient_stats['correlations'][ch_idx]
                
                # Update variance ratios
                for ch_idx in channel_variances:
                    if ch_idx in existing['variances']:
                        weight1 = existing['n_segments'] / n_total
                        weight2 = patient_stats['n_segments'] / n_total
                        existing['variances'][ch_idx] = (
                            weight1 * existing['variances'][ch_idx] + 
                            weight2 * patient_stats['variances'][ch_idx]
                        )
                    else:
                        existing['variances'][ch_idx] = patient_stats['variances'][ch_idx]
                
                # Update counts
                existing['n_segments'] = n_total
                existing['n_spikes'] += patient_stats['n_spikes']
            
            return patient_stats
            
        except Exception as e:
            self.logger.warning(f"Error calculating channel statistics: {str(e)}")
            return {}

    def _update_channel_rankings(self) -> None:
        """Update the ranking of most informative channels."""
        if not self.channel_stats or not self.channels_info['per_patient']:
            return
            
        # Aggregate correlations and variances across patients
        all_correlations = defaultdict(list)
        all_variances = defaultdict(list)
        
        for patient_id, stats in self.channels_info['per_patient'].items():
            for ch_idx, corr in stats['correlations'].items():
                all_correlations[ch_idx].append(corr)
            for ch_idx, var_ratio in stats['variances'].items():
                all_variances[ch_idx].append(var_ratio)
        
        # Calculate average correlation and variance ratio for each channel
        avg_correlations = {}
        avg_variances = {}
        
        for ch_idx in all_correlations:
            avg_correlations[ch_idx] = np.mean(all_correlations[ch_idx])
            
        for ch_idx in all_variances:
            avg_variances[ch_idx] = np.mean(all_variances[ch_idx])
        
        # Store in global statistics
        self.channels_info['correlation_with_spikes'] = avg_correlations
        self.channels_info['variance_during_spikes'] = avg_variances
        
        # Rank channels by correlation (absolute value)
        channels_by_corr = sorted(
            avg_correlations.items(), 
            key=lambda x: abs(x[1]), 
            reverse=True
        )
        
        self.channels_info['most_informative'] = [
            {'channel_idx': ch_idx, 'correlation': corr} 
            for ch_idx, corr in channels_by_corr
        ]

    def _augment_segment(self, segment: np.ndarray, 
                         augmentation_types: Optional[List[str]] = None) -> np.ndarray:
        """Apply data augmentation to a segment.

        Args:
            segment: Segment data of shape (n_channels, n_samples)
            augmentation_types: List of augmentation types to apply (default: all enabled types)

        Returns:
            Augmented segment
        """
        if not self.augmentation['enabled']:
            return segment
            
        # Default: use all enabled augmentation types
        if augmentation_types is None:
            augmentation_types = []
            if self.augmentation.get('amplitude_scale', False):
                augmentation_types.append('amplitude_scale')
            if self.augmentation.get('add_noise', False):
                augmentation_types.append('add_noise')
            if self.augmentation.get('channel_dropout', False):
                augmentation_types.append('channel_dropout')
        
        # Make a copy to avoid modifying the original
        augmented = segment.copy()
        
        # Apply selected augmentations
        for aug_type in augmentation_types:
            if aug_type == 'amplitude_scale' and self.augmentation.get('amplitude_scale', False):
                # Scale amplitude
                scale_range = self.augmentation.get('amplitude_range', (0.8, 1.2))
                scale_factor = np.random.uniform(scale_range[0], scale_range[1])
                augmented = augmented * scale_factor
                
            elif aug_type == 'add_noise' and self.augmentation.get('add_noise', False):
                # Add Gaussian noise
                noise_level = self.augmentation.get('noise_level', 0.05)
                # Calculate standard deviation of the data for proportional noise
                data_std = np.std(augmented)
                noise = np.random.normal(0, noise_level * data_std, augmented.shape)
                augmented = augmented + noise
                
            elif aug_type == 'channel_dropout' and self.augmentation.get('channel_dropout', False):
                # Randomly zero out some channels
                dropout_rate = self.augmentation.get('channel_dropout_rate', 0.1)
                n_channels = augmented.shape[0]
                n_dropout = int(n_channels * dropout_rate)
                
                if n_dropout > 0:
                    # Select channels to drop
                    drop_channels = np.random.choice(
                        np.arange(n_channels), 
                        n_dropout, 
                        replace=False
                    )
                    augmented[drop_channels, :] = 0
        
        return augmented

    def _save_data_examples(self) -> None:
        """Generate and save example visualizations of processed data.
        
        Creates plots of random examples from each split to verify data quality.
        """
        if not self.processed_files['train'] and not self.processed_files['val'] and not self.processed_files['test']:
            self.logger.warning("No data to visualize for sanity check")
            return
            
        self.logger.info("Generating data example visualizations for sanity check...")
        
        # Create output directory for visualizations
        viz_dir = os.path.join(self.output_dir, "visualizations")
        os.makedirs(viz_dir, exist_ok=True)
        
        # Sample a few examples from each split
        examples_per_split = 2
        for split in ['train', 'val', 'test']:
            if not self.processed_files[split]:
                continue
                
            # Get sample of files, prioritizing both spike and non-spike examples
            split_dir = os.path.join(self.output_dir, split)
            samples = []
            spike_count = 0
            non_spike_count = 0
            
            # Use random sampling with a large enough pool to ensure we get both classes
            file_pool = np.random.choice(self.processed_files[split], 
                                       min(50, len(self.processed_files[split])), 
                                       replace=False)
            
            for file_name in file_pool:
                if len(samples) >= examples_per_split:
                    break
                    
                file_path = os.path.join(split_dir, file_name)
                try:
                    sample = torch.load(file_path, map_location='cpu')
                    label = sample['label'].item()
                    
                    # Ensure we get at least one example of each class if available
                    if label == 1 and spike_count < examples_per_split // 2:
                        samples.append(sample)
                        spike_count += 1
                    elif label == 0 and non_spike_count < examples_per_split // 2:
                        samples.append(sample)
                        non_spike_count += 1
                        
                except Exception as e:
                    self.logger.warning(f"Error loading sample for visualization: {e}")
            
            # Plot the examples
            for i, sample in enumerate(samples):
                try:
                    # Extract data and metadata
                    data = sample['data'].numpy()
                    label = sample['label'].item()
                    patient_id = sample.get('patient_id', 'unknown')
                    has_spike = "spike" if label == 1 else "non-spike"
                    
                    # Create plot
                    fig, ax = plt.subplots(figsize=(10, 6))
                    
                    # Plot multiple channels with different colors
                    n_channels = min(10, data.shape[0])  # Limit to 10 channels for clarity
                    time_axis = np.arange(data.shape[1]) / self.sampling_rate
                    
                    # Use a color cycle for multiple channels
                    colors = plt.cm.tab10(np.linspace(0, 1, n_channels))
                    
                    for ch_idx in range(n_channels):
                        ax.plot(time_axis, data[ch_idx] + ch_idx * 2, color=colors[ch_idx], 
                               label=f"Channel {ch_idx}")
                    
                    # Highlight spike position if available
                    if label == 1 and 'spike_position' in sample:
                        spike_pos = sample['spike_position']
                        spike_time = spike_pos / self.sampling_rate
                        ax.axvline(x=spike_time, color='r', linestyle='--', 
                                 label=f"Spike at {spike_time:.3f}s")
                    
                    # Add labels and title
                    ax.set_xlabel('Time (s)')
                    ax.set_ylabel('Amplitude (a.u.)')
                    ax.set_title(f"{split.capitalize()} example: Patient {patient_id}, {has_spike}")
                    
                    # Save the figure
                    fig_path = os.path.join(viz_dir, f"{split}_{i+1}_{has_spike}.png")
                    plt.tight_layout()
                    plt.savefig(fig_path)
                    plt.close(fig)
                    
                except Exception as e:
                    self.logger.warning(f"Error creating visualization: {e}")
        
        self.logger.info(f"Saved data example visualizations to {viz_dir}")

    def _create_time_shifted_segments(self, segment: np.ndarray, label: int, 
                                     start_position: int, spike_position: Optional[int] = None,
                                     sampling_rate: int = None) -> List[Tuple[np.ndarray, int, int, Optional[int]]]:
        """Create time-shifted versions of a segment for data augmentation.

        Args:
            segment: Original segment data
            label: Segment label (1 for spike, 0 for non-spike)
            start_position: Starting position of the segment in original data
            spike_position: Position of the spike within the segment (if label=1)
            sampling_rate: Sampling rate (defaults to self.sampling_rate)

        Returns:
            List of tuples containing (augmented_segment, label, new_start_position, new_spike_position)
        """
        if not self.augmentation['enabled'] or label == 0:
            # No augmentation for non-spike segments or if augmentation is disabled
            return []
            
        sampling_rate = sampling_rate or self.sampling_rate
        max_shifts = self.augmentation['max_shifts']
        max_shift_samples = int(self.augmentation['time_shift_ms'] * sampling_rate / 1000)
        
        # Don't shift if no room to shift or max_shift_samples is 0
        if max_shift_samples <= 0:
            return []
            
        augmented_segments = []
        
        # Generate shifts
        shifts = np.random.choice(
            np.concatenate([
                np.arange(-max_shift_samples, 0),
                np.arange(1, max_shift_samples + 1)
            ]),
            size=min(max_shifts, 2 * max_shift_samples),  # Can't have more shifts than possible positions
            replace=False
        )
        
        # Apply shifts
        for shift in shifts:
            # Apply the shift
            shifted_segment = np.zeros_like(segment)
            
            if shift > 0:
                # Shift right
                shifted_segment[:, shift:] = segment[:, :-shift]
            else:
                # Shift left
                shifted_segment[:, :shift] = segment[:, -shift:]
                
            # If we need to preserve spikes, check if spike is still visible
            new_spike_pos = None
            if self.augmentation['preserve_spikes'] and spike_position is not None:
                new_spike_pos = spike_position + shift
                if new_spike_pos < 0 or new_spike_pos >= segment.shape[1]:
                    # Spike would be outside the segment, skip this augmentation
                    continue
                    
            # Calculate new start position
            new_start_position = start_position - shift
            
            # Apply additional augmentations
            augmented_segment = self._augment_segment(shifted_segment)
            
            # Add to results
            augmented_segments.append((augmented_segment, label, new_start_position, new_spike_pos))
            
        return augmented_segments

    def _balance_segments(self, segments: List[np.ndarray], labels: List[int], 
                         start_positions: List[int], 
                         spike_positions: Optional[List[Optional[int]]] = None) -> Tuple[
                             List[np.ndarray], List[int], List[int], Optional[List[Optional[int]]]]:
        """Balance the dataset according to the specified method and target ratio.

        Args:
            segments: List of segment arrays
            labels: List of labels
            start_positions: List of starting positions
            spike_positions: Optional list of spike positions within segments

        Returns:
            Tuple containing balanced (segments, labels, start_positions, spike_positions)
        """
        if self.balance_method == 'none':
            return segments, labels, start_positions, spike_positions
            
        # Count the occurrences of each class
        label_array = np.array(labels)
        spike_indices = np.where(label_array == 1)[0]
        non_spike_indices = np.where(label_array == 0)[0]
        
        n_spikes = len(spike_indices)
        n_non_spikes = len(non_spike_indices)
        
        if n_spikes == 0 or n_non_spikes == 0:
            # Can't balance if one class is missing
            return segments, labels, start_positions, spike_positions
            
        # Calculate target sizes based on target_class_ratio
        # target_ratio = n_spikes / (n_spikes + n_non_spikes)
        total_segments = n_spikes + n_non_spikes
        
        if self.balance_method == 'undersample':
            # Undersample the majority class
            if n_spikes / total_segments < self.target_class_ratio:
                # We need to reduce non-spikes
                target_non_spikes = int(n_spikes * (1 - self.target_class_ratio) / self.target_class_ratio)
                target_spikes = n_spikes
            else:
                # We need to reduce spikes
                target_spikes = int(n_non_spikes * self.target_class_ratio / (1 - self.target_class_ratio))
                target_non_spikes = n_non_spikes
                
            # Randomly select indices with stratified sampling
            np.random.seed(self.random_state)
            if target_spikes < n_spikes:
                selected_spike_indices = np.random.choice(spike_indices, target_spikes, replace=False)
            else:
                selected_spike_indices = spike_indices
                
            if target_non_spikes < n_non_spikes:
                selected_non_spike_indices = np.random.choice(non_spike_indices, target_non_spikes, replace=False)
            else:
                selected_non_spike_indices = non_spike_indices
                
            # Combine selected indices
            selected_indices = np.sort(np.concatenate([selected_spike_indices, selected_non_spike_indices]))
            
        elif self.balance_method == 'oversample':
            # Oversample the minority class
            if n_spikes < n_non_spikes:
                # We need to increase spikes
                target_spikes = int(n_non_spikes * self.target_class_ratio / (1 - self.target_class_ratio))
                # Generate indices with replacement for oversampling
                extra_spike_indices = np.random.choice(spike_indices, target_spikes - n_spikes, replace=True)
                # Combine all indices
                selected_indices = np.concatenate([np.arange(len(segments)), extra_spike_indices])
            else:
                # We need to increase non-spikes
                target_non_spikes = int(n_spikes * (1 - self.target_class_ratio) / self.target_class_ratio)
                # Generate indices with replacement for oversampling
                extra_non_spike_indices = np.random.choice(non_spike_indices, target_non_spikes - n_non_spikes, replace=True)
                # Combine all indices
                selected_indices = np.concatenate([np.arange(len(segments)), extra_non_spike_indices])
        else:
            self.logger.warning(f"Unknown balance method: {self.balance_method}, using all segments")
            return segments, labels, start_positions, spike_positions
            
        # Select the segments, labels and positions
        balanced_segments = [segments[i] if i < len(segments) else segments[i - len(segments)] for i in selected_indices]
        balanced_labels = [labels[i] if i < len(labels) else labels[i - len(labels)] for i in selected_indices]
        balanced_start_positions = [start_positions[i] if i < len(start_positions) else start_positions[i - len(start_positions)] for i in selected_indices]
        
        # Handle spike positions if provided
        if spike_positions:
            balanced_spike_positions = [spike_positions[i] if i < len(spike_positions) else spike_positions[i - len(spike_positions)] for i in selected_indices]
        else:
            balanced_spike_positions = None
            
        self.logger.info(f"Balanced {len(segments)} segments "
                        f"({n_spikes} spikes, {n_non_spikes} non-spikes) to {len(balanced_segments)} "
                        f"({balanced_labels.count(1)} spikes, {balanced_labels.count(0)} non-spikes)")
                        
        return balanced_segments, balanced_labels, balanced_start_positions, balanced_spike_positions
    
    def preprocess_file(self, file_info: Dict[str, str], interpolate: bool,
                       save_immediately: bool = None) -> Tuple[
        Optional[List[np.ndarray]], Optional[List[int]], Optional[List[int]], Optional[List[Optional[int]]]]:
        """Preprocess a single .ds file.

        Args:
            file_info: Dictionary with file information
            interpolate: If True, interpolate missing channels. Else drops to keep only intersection of valid channels.
            save_immediately: Whether to save segments immediately after processing. 
                              If None, uses self.online_saving.

        Returns:
            If save_immediately is False:
                Tuple containing:
                - segments: List of preprocessed segments
                - labels: Labels for each segment (1 for spike, 0 for non-spike)
                - start_positions: Original starting position of each segment in samples
                - spike_positions: Position of the spike within each segment (None for non-spike segments)
            If save_immediately is True:
                Returns (None, None, None, None) to indicate data was saved directly.
            Returns None for all values if processing fails.
        """
        start_time = time.time()
        
        file_path = file_info['path']
        patient_id = file_info['patient_id']
        group = file_info['group']
        
        # Determine which split this file belongs to
        split = self.patient_splits.get(patient_id, 'train')
        is_test_set = (split == 'test') or (group == 'Holdout')
        
        # For test set, use different settings to preserve data integrity
        test_overlap = 0.0 if is_test_set else self.overlap
        
        self.logger.info(f"Processing {file_path} (split: {split}, group: {group})")

        # Use save_immediately parameter if provided, otherwise use class setting
        if save_immediately is None:
            save_immediately = self.online_saving

        # Load the raw data
        try:
            raw_file = mne.io.read_raw_ctf(file_path, preload=True)
        except Exception as e:
            self.logger.error(f"Error loading {file_path}: {e}")
            return None, None, None, None

        # Keep only the MEG data (no references, eeg, eog, etc.)
        raw_file = raw_file.pick('meg')

        if interpolate:
            # Interpolate missing channels
            self.logger.debug("Interpolating missing channels")
            # handle a particular subject that has strange signal on some channels
            if 'Liogier_AllDataset1200Hz' in file_path:
                self.logger.debug("Handling Liogier_AllDataset1200Hz")
                strange_channels = ['MRO22-2805', 'MRO23-2805', 'MRO24-2805']
                raw_file.drop_channels(strange_channels)
            raw_file = interpolate_missing_channels(raw_file, self.good_channels, self.loc_meg_channels)
        else:
            # Only keep the valid channels
            self.logger.debug("Keeping only intersection of valid channels")
            raw_file = raw_file.pick_channels(ch_names=self.good_channels)

        # Resample
        original_sfreq = raw_file.info['sfreq']
        if raw_file.info['sfreq'] != self.sampling_rate:
            self.logger.debug(f"Resampling from {original_sfreq}Hz to {self.sampling_rate}Hz")
            raw_file.resample(sfreq=self.sampling_rate)

        # Apply bandpass filter (1-70Hz)
        self.logger.debug(f"Applying bandpass filter ({self.l_freq}-{self.h_freq}Hz)")
        raw_file.filter(l_freq=self.l_freq, h_freq=self.h_freq)

        # Apply notch filter (50Hz, power line interference)
        self.logger.debug(f"Applying notch filter ({self.notch_freq}Hz)")
        raw_file.notch_filter(freqs=self.notch_freq)

        # Get the data
        meg_data = raw_file.get_data()
        self.logger.debug(f"Data shape after preprocessing: {meg_data.shape}")

        # Apply normalization based on configuration
        meg_data = self._normalize_data(meg_data)
        self.logger.debug(f"Normalized data using method: {self.normalization['method']}")

        # Get annotations
        annotations = raw_file.annotations
        if len(annotations) == 0:
            self.logger.warning(f"No annotations found in {file_path}")
            if self.process_no_spike_files:
                self.logger.info(f"Processing {file_path} as control data (no spikes)")
                # Process as control data (all segments as non-spikes)
                spike_onsets = []
                self.dataset_metadata['processing_stats']['no_spike_files_processed'] += 1
            else:
                return None, None, None, None
        else:
            # Extract spike annotations based on group rules
            spike_onsets = self._get_spike_annotations(annotations, group)
            
            if len(spike_onsets) == 0:
                self.logger.warning(f"No valid spike annotations found in {file_path} using rules for {group}")
                if self.process_no_spike_files:
                    self.logger.info(f"Processing {file_path} as control data (no spikes)")
                    # Process as control data (all segments as non-spikes)
                    self.dataset_metadata['processing_stats']['no_spike_files_processed'] += 1
                else:
                    return None, None, None, None
                
            # Check if file has enough spikes - don't filter test set files by spike count
            elif not is_test_set and len(spike_onsets) < self.min_spikes_per_file and not self.process_no_spike_files:
                self.logger.warning(f"File {file_path} has only {len(spike_onsets)} spikes, which is below the minimum threshold of {self.min_spikes_per_file}")
                self.dataset_metadata['processing_stats']['skipped_low_spikes'] += 1
                return None, None, None, None
                
        self.logger.info(f"Found {len(spike_onsets)} valid spike annotations in {file_path}")

        # Convert spike onsets from seconds to samples
        spike_onset_samples = [int(onset * self.sampling_rate) for onset in spike_onsets]

        # Get total length in samples
        n_samples = meg_data.shape[1]

        segments = []
        labels = []
        start_positions = []  # Track the original starting position of each segment
        spike_positions = []  # Track the position of spikes within segments (for augmentation)
        saved_files = []  # Track saved file paths if save_immediately is True

        # Use test-specific overlap value for test set
        current_overlap = test_overlap if is_test_set else self.overlap
        step_size = int(self.clip_length_samples * (1 - current_overlap))
        
        if is_test_set:
            self.logger.info(f"Test set processing: Using overlap={test_overlap}, no augmentation, no balancing")

        # Cut the data into overlapping segments
        start_samples = range(0, n_samples - self.clip_length_samples + 1, step_size)
        
        # Track segments to process in this iteration
        segments_batch = []
        labels_batch = []
        start_positions_batch = []
        spike_positions_batch = []
        
        for start in start_samples:
            end = start + self.clip_length_samples
            segment = meg_data[:, start:end]  # take the segment across channels

            # Check if any spike occurs in this segment
            segment_spike_positions = [spike_onset - start for spike_onset in spike_onset_samples 
                                      if start <= spike_onset < end]
            
            has_spike = len(segment_spike_positions) > 0
            
            if save_immediately:
                segments_batch.append(segment)
                labels_batch.append(1 if has_spike else 0)
                
                # Store spike position (None if no spike)
                if has_spike:
                    # If multiple spikes, take the position of the first one
                    spike_positions_batch.append(segment_spike_positions[0])
                else:
                    spike_positions_batch.append(None)
    
                # Store original position in samples (at original sampling rate)
                original_start = int(
                    start * original_sfreq / self.sampling_rate)  # convert back to original sampling rate
                start_positions_batch.append(original_start)
                
                # Save batch if it gets large enough to avoid memory issues
                if len(segments_batch) >= 1000:
                    # Apply augmentation, balancing, etc.
                    processed_batch = self._process_segment_batch(
                        segments_batch, labels_batch, start_positions_batch, spike_positions_batch,
                        file_info, split, is_test_set, original_sfreq
                    )
                    
                    # Save the processed segments
                    saved_batch = self._save_segment_batch(processed_batch, file_info, split)
                    saved_files.extend(saved_batch)
                    
                    # Clear batch
                    segments_batch = []
                    labels_batch = []
                    start_positions_batch = []
                    spike_positions_batch = []
            else:
                segments.append(segment)
                labels.append(1 if has_spike else 0)
                
                # Store spike position (None if no spike)
                if has_spike:
                    # If multiple spikes, take the position of the first one
                    spike_positions.append(segment_spike_positions[0])
                else:
                    spike_positions.append(None)
    
                # Store original position in samples (at original sampling rate)
                original_start = int(
                    start * original_sfreq / self.sampling_rate)  # convert back to original sampling rate
                start_positions.append(original_start)

        # Process any remaining segments in the batch
        if save_immediately and segments_batch:
            processed_batch = self._process_segment_batch(
                segments_batch, labels_batch, start_positions_batch, spike_positions_batch,
                file_info, split, is_test_set, original_sfreq
            )
            
            # Save the processed segments
            saved_batch = self._save_segment_batch(processed_batch, file_info, split)
            saved_files.extend(saved_batch)
        
        # Store timing statistics
        end_time = time.time()
        if self.benchmark:
            file_duration = end_time - start_time
            self.timing_stats['per_file'][file_path] = {
                'duration_seconds': file_duration,
                'n_segments': len(saved_files) if save_immediately else len(segments),
                'seconds_per_segment': file_duration / (len(saved_files) if save_immediately else (len(segments) or 1))
            }
            self.timing_stats['total_time'] += file_duration
            self.timing_stats['file_count'] += 1
            self.timing_stats['segment_count'] += len(saved_files) if save_immediately else len(segments)

        if save_immediately:
            # Return the list of saved files for tracking
            return None, None, None, None
        else:
            self.logger.info(f"Created {len(segments)} segments, {sum(labels)} with spikes")
            
            # Process segments (augmentation, balancing)
            if len(segments) > 0:
                original_segments = segments.copy()
                
                # Skip augmentation for test set
                if self.augmentation['enabled'] and not is_test_set:
                    augmented_segments = []
                    augmented_labels = []
                    augmented_start_positions = []
                    augmented_spike_positions = []
                    
                    for i, (segment, label, start_pos, spike_pos) in enumerate(zip(segments, labels, start_positions, spike_positions)):
                        # Only augment spike segments
                        if label == 1:
                            shifted_segments = self._create_time_shifted_segments(
                                segment, label, start_pos, spike_pos, self.sampling_rate
                            )
                            
                            for shifted_segment, shifted_label, shifted_start_pos, shifted_spike_pos in shifted_segments:
                                augmented_segments.append(shifted_segment)
                                augmented_labels.append(shifted_label)
                                augmented_start_positions.append(shifted_start_pos)
                                augmented_spike_positions.append(shifted_spike_pos)
                    
                    # Add augmented segments to the original ones
                    if augmented_segments:
                        self.logger.info(f"Added {len(augmented_segments)} augmented segments")
                        segments.extend(augmented_segments)
                        labels.extend(augmented_labels)
                        start_positions.extend(augmented_start_positions)
                        spike_positions.extend(augmented_spike_positions)
                        
                        # Update metadata
                        self.dataset_metadata['processing_stats']['augmented_segments'] += len(augmented_segments)
                
                # Apply class balancing if enabled, but skip for test set
                if self.balance_method != 'none' and not is_test_set:
                    segments, labels, start_positions, spike_positions = self._balance_segments(
                        segments, labels, start_positions, spike_positions
                    )
                    
                # Limit the number of segments if configured, but don't limit test set
                if self.max_segments_per_file and len(segments) > self.max_segments_per_file and not is_test_set:
                    self.logger.info(f"Limiting {len(segments)} segments to {self.max_segments_per_file}")
                    
                    # Stratified sampling to maintain class distribution
                    positive_indices = [i for i, label in enumerate(labels) if label == 1]
                    negative_indices = [i for i, label in enumerate(labels) if label == 0]
                    
                    # Calculate how many of each class to keep
                    n_positives = min(len(positive_indices), int(self.max_segments_per_file * sum(labels) / len(labels)))
                    n_negatives = self.max_segments_per_file - n_positives
                    
                    # Select random samples from each class
                    np.random.seed(self.random_state)
                    selected_positives = np.random.choice(positive_indices, n_positives, replace=False)
                    selected_negatives = np.random.choice(negative_indices, n_negatives, replace=False)
                    
                    # Combine and sort indices
                    selected_indices = np.sort(np.concatenate([selected_positives, selected_negatives]))
                    
                    # Extract selected samples
                    segments = [segments[i] for i in selected_indices]
                    labels = [labels[i] for i in selected_indices]
                    start_positions = [start_positions[i] for i in selected_indices]
                    spike_positions = [spike_positions[i] for i in selected_indices]
                
                if self.channel_stats:
                    # Calculate channel statistics using original (non-augmented) segments
                    self._calculate_channel_statistics(
                        np.array(original_segments), labels[:len(original_segments)], patient_id
                    )
                
            self.logger.info(f"Final segment count: {len(segments)} ({sum(labels)} spikes, {len(labels) - sum(labels)} non-spikes)")
            return segments, labels, start_positions, spike_positions

    def _process_segment_batch(self, segments: List[np.ndarray], labels: List[int],
                              start_positions: List[int], spike_positions: List[Optional[int]],
                              file_info: Dict[str, str], split: str, is_test_set: bool,
                              original_sfreq: float) -> Dict[str, Any]:
        """Process a batch of segments.

        Args:
            segments: List of segment arrays
            labels: List of segment labels
            start_positions: List of starting positions
            spike_positions: List of spike positions
            file_info: Dictionary with file information
            split: Data split ('train', 'val', or 'test')
            is_test_set: Whether the file belongs to the test set
            original_sfreq: Original sampling frequency

        Returns:
            Dictionary with processed segments, labels, etc.
        """
        if len(segments) == 0:
            return {
                'segments': [],
                'labels': [],
                'start_positions': [],
                'spike_positions': [],
                'augmented_flag': []
            }
            
        # Keep track of which segments were created through augmentation
        augmented_flag = [False] * len(segments)
            
        # Skip augmentation for test set
        if self.augmentation['enabled'] and not is_test_set:
            augmented_segments = []
            augmented_labels = []
            augmented_start_positions = []
            augmented_spike_positions = []
            
            for i, (segment, label, start_pos, spike_pos) in enumerate(zip(segments, labels, start_positions, spike_positions)):
                # Only augment spike segments
                if label == 1:
                    shifted_segments = self._create_time_shifted_segments(
                        segment, label, start_pos, spike_pos, self.sampling_rate
                    )
                    
                    for shifted_segment, shifted_label, shifted_start_pos, shifted_spike_pos in shifted_segments:
                        augmented_segments.append(shifted_segment)
                        augmented_labels.append(shifted_label)
                        augmented_start_positions.append(shifted_start_pos)
                        augmented_spike_positions.append(shifted_spike_pos)
                        
            # Add augmented segments to the original ones
            if augmented_segments:
                self.logger.debug(f"Added {len(augmented_segments)} augmented segments")
                segments.extend(augmented_segments)
                labels.extend(augmented_labels)
                start_positions.extend(augmented_start_positions)
                spike_positions.extend(augmented_spike_positions)
                augmented_flag.extend([True] * len(augmented_segments))
                
                # Update metadata
                self.dataset_metadata['processing_stats']['augmented_segments'] += len(augmented_segments)
        
        # Apply class balancing if enabled, but skip for test set
        if self.balance_method != 'none' and not is_test_set:
            result = self._balance_segments(
                segments, labels, start_positions, spike_positions
            )
            segments, labels, start_positions, spike_positions = result
            
            # Need to update augmented_flag to match the new segment count
            # This is approximate since we don't know exactly which segments were kept
            if len(augmented_flag) != len(segments):
                # If the length changed, we need to create a new array
                n_original = len(augmented_flag)
                n_new = len(segments)
                
                if n_new < n_original:
                    # If we're undersampling, just keep the first n_new flags
                    augmented_flag = augmented_flag[:n_new]
                else:
                    # If we're oversampling, use the original flags and add extras
                    extra_flags = []
                    for _ in range(n_new - n_original):
                        # Assume new segments come from oversampling originals
                        extra_flags.append(False)
                    augmented_flag.extend(extra_flags)
                    
        return {
            'segments': segments,
            'labels': labels,
            'start_positions': start_positions,
            'spike_positions': spike_positions,
            'augmented_flag': augmented_flag
        }

    def _save_segment_batch(self, batch: Dict[str, Any], file_info: Dict[str, str], 
                           split: str) -> List[str]:
        """Save a batch of processed segments to disk.

        Args:
            batch: Dictionary with segments, labels, etc.
            file_info: Dictionary with file information
            split: Data split ('train', 'val', or 'test')

        Returns:
            List of saved file paths
        """
        segments = batch['segments']
        labels = batch['labels']
        start_positions = batch['start_positions']
        spike_positions = batch['spike_positions']
        augmented_flag = batch['augmented_flag']
        
        if len(segments) == 0:
            return []
            
        patient_id = file_info['patient_id']
        filename_origin = file_info['filename'].split('.')[0]
        group = file_info['group']
        
        # Determine if this is part of the test set
        is_test_set = (split == 'test') or (group == 'Holdout')
        
        self.logger.debug(f"Saving {len(segments)} segments for patient {patient_id} to {split} split")
        
        saved_files = []

        # Save each segment
        for i in range(len(segments)):
            file_prefix = f"{patient_id}_{filename_origin}_{i:04d}"  # i:04d formats the output to have 4 numbers like 0001, 0002,...
            file_prefix = file_prefix.replace('__', '_')  # adding the origin might double the underscores
            file_path = os.path.join(self.output_dir, split, f"{file_prefix}.pt")

            # Tensor conversion
            segment_tensor = torch.tensor(segments[i], dtype=torch.float32)
            label_tensor = torch.tensor(labels[i], dtype=torch.long)
            
            # Extended metadata
            metadata = {
                'data': segment_tensor,
                'label': label_tensor,
                'patient_id': patient_id,
                'original_filename': filename_origin,
                'start_position': int(start_positions[i]),
                'original_index': i,
                'group': group,
                # Additional metadata
                'preprocessing_config': {
                    'sampling_rate': self.sampling_rate,
                    'l_freq': self.l_freq,
                    'h_freq': self.h_freq,
                    'notch_freq': self.notch_freq,
                    'normalization': self.normalization['method'],
                },
                'segment_length_s': self.clip_length_s,
                'segment_length_samples': self.clip_length_samples,
                'is_test_set': is_test_set,
            }
            
            # Add spike position metadata if available
            if spike_positions[i] is not None:
                metadata['spike_position'] = int(spike_positions[i])
                metadata['spike_time'] = spike_positions[i] / self.sampling_rate
                
            # Add augmentation flag if applicable and not a test sample
            if not is_test_set and i < len(augmented_flag) and augmented_flag[i]:
                metadata['augmented'] = True

            torch.save(metadata, file_path)
            saved_files.append(f"{file_prefix}.pt")
            
            # Update class distribution statistics
            self.dataset_metadata['class_distribution'][split][labels[i]] += 1

        self.logger.debug(f"Saved {len(segments)} segments for patient {patient_id}")
        return saved_files
        
    def save_segments(self,
                      segments: List[np.ndarray],
                      labels: List[int],
                      start_positions: List[int],
                      spike_positions: List[Optional[int]],
                      file_info: Dict[str, str],
                      split: str) -> None:
        """Save preprocessed segments to disk.

        Args:
            segments: Preprocessed segments.
            labels: Labels for each segment.
            start_positions: Original starting positions of each segment.
            spike_positions: Positions of spikes within segments (None for non-spike segments).
            file_info: Dictionary with file information.
            split: Data split ('train', 'val', or 'test').
        """
        patient_id = file_info['patient_id']
        filename_origin = file_info['filename'].split('.')[0]
        group = file_info['group']
        
        # Determine if this is part of the test set
        is_test_set = (split == 'test') or (group == 'Holdout')
        
        if len(segments) == 0:
            self.logger.warning(f"No segments found for patient {patient_id}")
            return

        self.logger.info(f"Saving {len(segments)} segments for patient {patient_id} to {split} split")

        # Save each segment
        for i in range(len(segments)):
            file_prefix = f"{patient_id}_{filename_origin}_{i:04d}"  # i:04d formats the output to have 4 numbers like 0001, 0002,...
            file_prefix = file_prefix.replace('__', '_')  # adding the origin might double the underscores
            file_path = os.path.join(self.output_dir, split, f"{file_prefix}.pt")

            # Tensor conversion
            segment_tensor = torch.tensor(segments[i], dtype=torch.float32)
            label_tensor = torch.tensor(labels[i], dtype=torch.long)
            
            # Extended metadata
            metadata = {
                'data': segment_tensor,
                'label': label_tensor,
                'patient_id': patient_id,
                'original_filename': filename_origin,
                'start_position': int(start_positions[i]),
                'original_index': i,
                'group': group,
                # Additional metadata
                'preprocessing_config': {
                    'sampling_rate': self.sampling_rate,
                    'l_freq': self.l_freq,
                    'h_freq': self.h_freq,
                    'notch_freq': self.notch_freq,
                    'normalization': self.normalization['method'],
                },
                'segment_length_s': self.clip_length_s,
                'segment_length_samples': self.clip_length_samples,
                'is_test_set': is_test_set,
            }
            
            # Add spike position metadata if available
            if spike_positions[i] is not None:
                metadata['spike_position'] = int(spike_positions[i])
                metadata['spike_time'] = spike_positions[i] / self.sampling_rate
                
            # Add augmentation flag if applicable and not a test sample
            if not is_test_set and i >= len(segments) - self.dataset_metadata['processing_stats'].get('augmented_segments', 0):
                metadata['augmented'] = True

            torch.save(metadata, file_path)

            self.processed_files[split].append(f"{file_prefix}.pt")
            
            # Update class distribution statistics
            self.dataset_metadata['class_distribution'][split][labels[i]] += 1

        self.logger.debug(f"Saved {len(segments)} segments for patient {patient_id}")
        
    def process_file_wrapper(self, file_info):
        """Process a single file for parallel execution.
        
        This has to be a class method (not a local function) to support pickling
        for multiprocessing.
        
        Args:
            file_info: Dictionary with file information
            
        Returns:
            Tuple containing success flag and file information
        """
        patient_id = file_info['patient_id']
        group = file_info['group']
        split = self.patient_splits.get(patient_id, 'train')
        
        # Determine if this file belongs to the test set
        is_test_set = (split == 'test') or (group == 'Holdout')
        
        if is_test_set:
            self.logger.info(f"Processing test set file: {file_info['path']} (minimal processing)")
        
        try:
            # Process the file with immediate saving
            segments, labels, start_positions, spike_positions = self.preprocess_file(
                file_info, interpolate=True, save_immediately=True
            )
            
            # Since we're using immediate saving, we don't have segment information here
            # Instead, collect segment info from already saved files
            split_dir = os.path.join(self.output_dir, split)
            prefix = f"{patient_id}_{file_info['filename'].split('.')[0]}"
            prefix = prefix.replace('__', '_')
            
            processed_files = []
            for filename in os.listdir(split_dir):
                if filename.startswith(prefix) and filename.endswith('.pt'):
                    processed_files.append(filename)
                    
            if processed_files:
                # Count class distribution by loading label from each file
                class_counts = {0: 0, 1: 0}
                
                # Only check a sample of files to avoid overhead
                sample_size = min(len(processed_files), 100)
                sampled_files = np.random.choice(processed_files, sample_size, replace=False)
                
                for filename in sampled_files:
                    file_path = os.path.join(split_dir, filename)
                    try:
                        sample = torch.load(file_path, map_location='cpu')
                        label = sample['label'].item()
                        class_counts[label] += 1
                    except:
                        pass
                        
                # Estimate full distribution
                scaling_factor = len(processed_files) / sample_size
                class_counts = {k: int(v * scaling_factor) for k, v in class_counts.items()}
                
                if is_test_set:
                    self.logger.info(f"Test set file processed: {file_info['path']} - {len(processed_files)} segments (approx. {class_counts[1]} spikes)")
                
                return True, {'split': split, 'files': processed_files, 'class_counts': class_counts}
            return False, None
        except Exception as e:
            self.logger.error(f"Error processing {file_info['path']}: {str(e)}")
            return False, None


    def process_all(self, interpolate: bool) -> None:
        """Process all .ds files in the root directory.

        This method:
        1. Splits patients across train/val/test sets
        2. Processes each .ds file in parallel
        3. Saves the processed segments to the appropriate split folder

        Args:
            interpolate: bool, whether to interpolate missing channels or drop them.

        Raises:
            ValueError: If no .ds files are found.
        """
        start_time = time.time()
        
        ds_files = self.find_ds_files()

        if not ds_files:
            error_msg = f"No .ds files found in {self.root_dir}"
            self.logger.error(error_msg)
            raise ValueError(error_msg)

        # Split patients before processing
        self.split_patients()

        # Process files
        if self.n_jobs > 1:
            # Parallel processing using a simpler technique that avoids pickling errors
            self.logger.info(f"Processing {len(ds_files)} files with {self.n_jobs} parallel jobs")
            
            # Disable debug logging during multiprocessing to avoid log contention
            logging_level = self.logger.level
            self.logger.setLevel(logging.WARNING)
            
            try:
                with mp.Pool(processes=self.n_jobs) as pool:
                    # Map our function across all files
                    results = list(tqdm(
                        pool.imap(self.process_file_wrapper, ds_files),
                        total=len(ds_files),
                        desc="Processing files"
                    ))
                    
                    # Process results
                    for success, result in results:
                        if success and result:
                            split = result['split']
                            # Update processed files
                            self.processed_files[split].extend(result['files'])
                            # Update class distribution
                            for label, count in result['class_counts'].items():
                                self.dataset_metadata['class_distribution'][split][label] += count
                            # Update count
                            self.dataset_metadata['processing_stats']['processed_files'] += 1
            finally:
                # Restore logging level
                self.logger.setLevel(logging_level)
            
        else:
            # Sequential processing
            for file_info in tqdm(ds_files, desc="Processing files"):
                # Process the file and get results
                success, result = self.process_file_wrapper(file_info)
                
                # Update tracking data if successful
                if success and result:
                    split = result['split']
                    self.processed_files[split].extend(result['files'])
                    for label, count in result['class_counts'].items():
                        self.dataset_metadata['class_distribution'][split][label] += count
                    self.dataset_metadata['processing_stats']['processed_files'] += 1

        # Update channel rankings
        if self.channel_stats:
            self.logger.info("Updating channel rankings...")
            self._update_channel_rankings()

        # Save the file lists
        for split_name in ['train', 'val', 'test']:
            split_file = os.path.join(self.output_dir, f"{split_name}_files.pkl")
            with open(split_file, 'wb') as f:
                pickle.dump(self.processed_files[split_name], f)
            self.logger.info(f"Saved {split_name} file list to {split_file}")

        # Save the patient splits for reference
        patient_splits_file = os.path.join(self.output_dir, "patient_splits.pkl")
        with open(patient_splits_file, 'wb') as f:
            pickle.dump(self.patient_splits, f)
        self.logger.info(f"Saved patient splits to {patient_splits_file}")

        # Save patient groups information
        patient_groups_file = os.path.join(self.output_dir, "patient_groups.pkl")
        with open(patient_groups_file, 'wb') as f:
            pickle.dump(self.patient_groups, f)
        self.logger.info(f"Saved patient groups to {patient_groups_file}")
        
        # Save extended dataset metadata
        metadata_file = os.path.join(self.output_dir, "dataset_metadata.pkl")
        with open(metadata_file, 'wb') as f:
            pickle.dump(self.dataset_metadata, f)
        self.logger.info(f"Saved dataset metadata to {metadata_file}")
        
        # Also save metadata as YAML for human readability
        metadata_yaml = os.path.join(self.output_dir, "dataset_metadata.yaml")
        with open(metadata_yaml, 'w') as f:
            yaml.dump(self.dataset_metadata, f, default_flow_style=False)
        self.logger.info(f"Saved human-readable metadata to {metadata_yaml}")
        
        # Save benchmark results if enabled
        if self.benchmark:
            total_time = time.time() - start_time
            self.timing_stats['total_pipeline_time'] = total_time
            
            # Calculate aggregated statistics
            files_per_second = self.timing_stats['file_count'] / total_time
            segments_per_second = self.timing_stats['segment_count'] / total_time
            
            # Add summary statistics
            self.timing_stats['summary'] = {
                'files_per_second': files_per_second,
                'segments_per_second': segments_per_second,
                'average_time_per_file': total_time / self.timing_stats['file_count'] if self.timing_stats['file_count'] > 0 else 0,
                'total_time_minutes': total_time / 60
            }
            
            # Save the timing statistics
            timing_file = os.path.join(self.output_dir, "timing_stats.json")
            with open(timing_file, 'w') as f:
                json.dump(self.timing_stats, f, indent=2)
            self.logger.info(f"Saved timing statistics to {timing_file}")
            
        # Save channel statistics if enabled
        if self.channel_stats:
            # Save the channel statistics
            channels_file = os.path.join(self.output_dir, "channel_stats.pkl")
            with open(channels_file, 'wb') as f:
                pickle.dump(self.channels_info, f)
            self.logger.info(f"Saved channel statistics to {channels_file}")
            
            # Save a human-readable version
            channels_yaml = os.path.join(self.output_dir, "channel_stats.yaml")
            with open(channels_yaml, 'w') as f:
                yaml.dump({
                    'most_informative_channels': self.channels_info['most_informative'][:20],
                    'summary': {
                        'total_patients': len(self.channels_info['per_patient'])
                    }
                }, f, default_flow_style=False)

        self.logger.info(f"Processing complete. Files saved to {self.output_dir}")
        self.logger.info(f"Train: {len(self.processed_files['train'])} files "
                      f"({self.dataset_metadata['class_distribution']['train'][1]} spikes, "
                      f"{self.dataset_metadata['class_distribution']['train'][0]} non-spikes)")
        self.logger.info(f"Val: {len(self.processed_files['val'])} files "
                      f"({self.dataset_metadata['class_distribution']['val'][1]} spikes, "
                      f"{self.dataset_metadata['class_distribution']['val'][0]} non-spikes)")
        self.logger.info(f"Test: {len(self.processed_files['test'])} files "
                      f"({self.dataset_metadata['class_distribution']['test'][1]} spikes, "
                      f"{self.dataset_metadata['class_distribution']['test'][0]} non-spikes)")
        
        # Generate class weights and save for model training
        class_weights = {}
        for split in ['train', 'val', 'test']:
            split_class_distr = self.dataset_metadata['class_distribution'][split]
            total = split_class_distr[0] + split_class_distr[1]
            if total > 0:
                class_weights[split] = {
                    # Weight is inverse of frequency
                    0: total / (2 * split_class_distr[0]) if split_class_distr[0] > 0 else 0,
                    1: total / (2 * split_class_distr[1]) if split_class_distr[1] > 0 else 0
                }
        
        # Save class weights
        weights_file = os.path.join(self.output_dir, "class_weights.pkl")
        with open(weights_file, 'wb') as f:
            pickle.dump(class_weights, f)
        self.logger.info(f"Saved class weights to {weights_file}")
        
        # Save a sanity check visualization
        self._save_data_examples()

def validate_dataset(raw_data_file_path: Path,
                     processed_data_root: Path,
                     processed_sfreq: int = 200,
                     good_channels_file_path: Optional[Path] = None,
                     loc_meg_channels_file_path: Optional[Path] = None,
                     logger: Optional[logging.Logger] = None,
                     show_plot: bool = False,
                     output_dir: Optional[Path] = None) -> None:
    """Reconstruct a MEG recording from the torch fragments to validate preprocessing.

    This function:
    1. Loads the original .ds file
    2. Loads all processed fragments for the patient
    3. Reconstructs the binary spike signal using the original position information
    4. Plots the original and reconstructed signals for comparison

    Args:
        raw_data_file_path: Path to the original .ds file
        processed_data_root: Path to the directory containing processed data
        processed_sfreq: Sampling frequency of the processed data
        good_channels_file_path: Path to the file with good channels
        loc_meg_channels_file_path: Path to the file with channel locations
        logger: Logger instance
        show_plot: If True, display the plot
        output_dir: Optional directory to save visualization output (defaults to processed_data_root)

    Raises:
        ValueError: If no processed fragments are found for the patient
    """
    logger = logger or logging.getLogger(__name__)
    import matplotlib
    import matplotlib.pyplot as plt
    if show_plot:
        matplotlib.use('TkAgg')

    # Extract patient ID from the directory structure
    # For the new structure, patient_id is the parent directory name
    patient_id = os.path.basename(os.path.dirname(raw_data_file_path))
    group = os.path.basename(os.path.dirname(os.path.dirname(raw_data_file_path)))
    filename_origin = os.path.basename(raw_data_file_path).split('.')[0]
    
    logger.info(f"Validating patient: {patient_id}, group: {group}, file: {filename_origin}")

    # Load the original .ds file
    try:
        raw_file = mne.io.read_raw_ctf(raw_data_file_path, preload=True)
        raw_file = raw_file.pick('meg')
    except Exception as e:
        logger.error(f"Error loading original file {raw_data_file_path}: {e}")
        return

    # Get sampling rate and total samples
    orig_sfreq = raw_file.info['sfreq']
    n_samples = len(raw_file.times)
    duration_s = n_samples / orig_sfreq
    logger.debug(f"Original recording: {duration_s:.2f}s at {orig_sfreq}Hz ({n_samples} samples)")

    # Create a binary array for original data based on annotations
    original_binary = np.zeros(n_samples)

    # Extract spike annotations based on group rules
    # Create a temporary MEGPreprocessor to reuse the annotation extraction logic
    temp_processor = MEGPreprocessor(
        root_dir="",
        output_dir="",
        good_channels_file_path=good_channels_file_path,
        loc_meg_channels_file_path=loc_meg_channels_file_path,
        logger=logger
    )
    
    # Get spike annotations using the same rules as during processing
    spike_onsets = temp_processor._get_spike_annotations(raw_file.annotations, group)
    
    logger.info(f"Found {len(spike_onsets)} spike annotations in original file")

    for onset in spike_onsets:
        onset_sample = int(onset * orig_sfreq)
        # Mark a window around the spike
        window_samples = int(1.0 * orig_sfreq)  # 1-second window
        start = max(0, onset_sample - window_samples // 2)
        end = min(n_samples, onset_sample + window_samples // 2)
        original_binary[start:end] = 1

    # Find all processed fragments for this patient
    logger.info("Looking for processed fragments...")
    all_fragments = []
    for split in ['train', 'val', 'test']:
        split_dir = os.path.join(processed_data_root, split)
        if os.path.exists(split_dir):
            for file in os.listdir(split_dir):
                if patient_id in file and filename_origin in file and file.endswith('.pt'):
                    fragment_path = os.path.join(split_dir, file)
                    try:
                        fragment = torch.load(fragment_path, map_location='cpu')
                        all_fragments.append(fragment)
                    except Exception as e:
                        logger.error(f"Error loading {fragment_path}: {e}")

    if not all_fragments:
        error_msg = f"No processed fragments found for patient {patient_id}, file {filename_origin}"
        logger.error(error_msg)
        raise ValueError(error_msg)

    logger.info(f"Found {len(all_fragments)} processed fragments")

    # Get fragment properties
    sample_fragment = all_fragments[0]
    clip_length_samples = sample_fragment['data'].shape[1]
    logger.debug(f"Fragment shape: {sample_fragment['data'].shape}")

    # Reconstruct binary signal from fragments using stored position information
    reconstructed_binary = np.zeros(int(duration_s * processed_sfreq))

    # Check if position information is available
    if 'start_position' in all_fragments[0]:
        logger.info("Using stored position information for reconstruction")
        for fragment in all_fragments:
            if fragment['label'] == 1:  # If this fragment has a spike
                # Convert original position to reconstructed timeline
                start_pos = int(fragment['start_position'] * processed_sfreq / orig_sfreq)
                end_pos = min(start_pos + clip_length_samples, len(reconstructed_binary))
                if start_pos < len(reconstructed_binary):
                    reconstructed_binary[start_pos:end_pos] = 1
    else:
        logger.warning("No position information found in fragments!")
        # Fall back to approximate method - try to match spike patterns
        logger.info("Attempting to match spike patterns based on timing")
        for onset in spike_onsets:
            spike_sample = int(onset * processed_sfreq)
            window_start = max(0, spike_sample - clip_length_samples // 2)
            window_end = min(len(reconstructed_binary), spike_sample + clip_length_samples // 2)
            reconstructed_binary[window_start:window_end] = 1

    # Create the plots
    logger.info("Creating validation plots")
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 8), sharex=True)

    # Plot original binary signal
    time_orig = np.arange(n_samples) / orig_sfreq
    ax1.plot(time_orig, original_binary, 'b-', label='Original')
    ax1.set_title(f'Original MEG Recording: {patient_id} ({len(spike_onsets)} spikes)')
    ax1.set_ylabel('Spike Present')

    # Add vertical lines for exact spike times
    for onset in spike_onsets:
        ax1.axvline(x=onset, color='r', linestyle='--', alpha=0.5)

    ax1.legend()

    # Plot reconstructed binary signal
    time_reconstructed = np.arange(len(reconstructed_binary)) / processed_sfreq
    ax2.plot(time_reconstructed, reconstructed_binary, 'g-', label='Reconstructed')
    ax2.set_title('Reconstructed from Processed Fragments')
    ax2.set_ylabel('Spike Present')
    ax2.set_xlabel('Time (s)')
    ax2.legend()

    # Adjust display
    plt.tight_layout()
    
    # Save to output_dir if provided, otherwise use processed_data_root
    save_dir = output_dir if output_dir else processed_data_root
    os.makedirs(save_dir, exist_ok=True)
    
    plot_path = os.path.join(save_dir, f"{patient_id}_{filename_origin}_validation.png")
    plt.savefig(plot_path)
    logger.info(f"Saved validation plot to {plot_path}")
    if show_plot:
        plt.show()

    # Report statistics
    spike_fragments = sum(1 for f in all_fragments if f['label'] == 1)
    orig_spike_percentage = (np.sum(original_binary) / len(original_binary)) * 100
    frag_spike_percentage = (spike_fragments / len(all_fragments)) * 100

    logger.info("Validation Summary:")
    logger.info(f"Original recording: {duration_s:.2f}s, {len(spike_onsets)} annotated spikes ({orig_spike_percentage:.2f}% of time)")
    logger.info(f"Processed fragments: {len(all_fragments)} total, {spike_fragments} with spikes ({frag_spike_percentage:.2f}%)")
    
    # Extended validation: examine spike positions within fragments
    if any('spike_position' in f for f in all_fragments):
        logger.info("Analyzing spike positions within fragments:")
        
        # Collect spike positions
        spike_positions = [f.get('spike_position') for f in all_fragments if f.get('spike_position') is not None]
        
        if spike_positions:
            # Convert to numpy array for statistics
            spike_positions = np.array(spike_positions)
            
            # Calculate distribution statistics
            mean_pos = np.mean(spike_positions)
            std_pos = np.std(spike_positions)
            median_pos = np.median(spike_positions)
            min_pos = np.min(spike_positions)
            max_pos = np.max(spike_positions)
            
            # Report
            logger.info(f"Spike position statistics (in samples):")
            logger.info(f"  Mean: {mean_pos:.1f}, Std: {std_pos:.1f}, Median: {median_pos:.1f}")
            logger.info(f"  Range: [{min_pos}, {max_pos}]")
            
            # Plot histogram of spike positions
            plt.figure(figsize=(10, 6))
            plt.hist(spike_positions, bins=20)
            plt.title(f"Spike Position Distribution - {patient_id}")
            plt.xlabel("Position within segment (samples)")
            plt.ylabel("Count")
            plt.axvline(clip_length_samples/2, color='r', linestyle='--', label="Segment Center")
            plt.legend()
            
            # Save to output_dir if provided, otherwise use processed_data_root
            save_dir = output_dir if output_dir else processed_data_root
            pos_plot_path = os.path.join(save_dir, f"{patient_id}_{filename_origin}_spike_positions.png")
            plt.savefig(pos_plot_path)
            logger.info(f"Saved spike position histogram to {pos_plot_path}")
            if show_plot:
                plt.show()


def test_datasets(output_dir: str, logger: Optional[logging.Logger] = None, 
                visualize: bool = True, n_samples: int = 2) -> Dict[str, Any]:
    """Test the MEG datasets to ensure they're correctly loaded.

    This function:
    1. Loads the file lists for each split
    2. Creates datasets for each split
    3. Tests loading a sample from each dataset
    4. Reports class distribution statistics
    5. Optionally creates visualizations of sample data

    Args:
        output_dir: Directory containing the processed data
        logger: Logger instance
        visualize: Whether to create visualizations
        n_samples: Number of samples to visualize per class

    Returns:
        Dictionary with test results
    """
    logger = logger or logging.getLogger(__name__)
    logger.info("Testing MEG Dataset...")

    # Load file lists
    try:
        with open(os.path.join(output_dir, "train_files.pkl"), 'rb') as f:
            train_files = pickle.load(f)
        with open(os.path.join(output_dir, "val_files.pkl"), 'rb') as f:
            val_files = pickle.load(f)
        with open(os.path.join(output_dir, "test_files.pkl"), 'rb') as f:
            test_files = pickle.load(f)
    except Exception as e:
        logger.error(f"Error loading file lists: {e}")
        return {}

    # Load class weights if available
    try:
        with open(os.path.join(output_dir, "class_weights.pkl"), 'rb') as f:
            class_weights = pickle.load(f)
        logger.info(f"Loaded class weights: {class_weights}")
    except:
        class_weights = None
        logger.info("No class weights found, not using weights")

    # Create datasets
    train_dataset = MEGDataset(
        root=os.path.join(output_dir, 'train'),
        files=train_files,
        class_weights=class_weights.get('train') if class_weights else None
    )
    val_dataset = MEGDataset(
        root=os.path.join(output_dir, 'val'),
        files=val_files,
        class_weights=class_weights.get('val') if class_weights else None
    )
    test_dataset = MEGDataset(
        root=os.path.join(output_dir, 'test'),
        files=test_files,
        class_weights=class_weights.get('test') if class_weights else None
    )

    # Test loading a sample from each dataset
    for name, dataset in [("Train", train_dataset), ("Validation", val_dataset), ("Test", test_dataset)]:
        logger.info(f"{name} Dataset:")
        logger.info(f"Number of samples: {len(dataset)}")

        if len(dataset) > 0:
            try:
                # Check if dataset returns weights
                result = dataset[0]
                if len(result) == 3:
                    data, label, weight = result
                    logger.info(f"Sample data shape: {data.shape}")
                    logger.info(f"Sample label: {label}")
                    logger.info(f"Sample weight: {weight}")
                else:
                    data, label = result
                    logger.info(f"Sample data shape: {data.shape}")
                    logger.info(f"Sample label: {label}")

                # Report class distribution
                logger.info("Computing class distribution statistics... (beware, can be very slow for large datasets)")
                labels = [dataset[i][1].item() for i in range(len(dataset))]
                unique_labels, counts = np.unique(labels, return_counts=True)
                logger.info("Class distribution:")
                for lbl, cnt in zip(unique_labels, counts):
                    logger.info(f"  Class {lbl}: {cnt} samples ({cnt / len(labels) * 100:.2f}%)")
            except Exception as e:
                logger.error(f"Error testing {name} dataset: {e}")
        else:
            logger.warning(f"No samples found in {name} dataset")

    # Check dataset metadata if available
    test_results = {
        'metadata': {},
        'splits': {},
        'patient_distribution': {},
        'class_distribution': {},
        'sample_data': {}
    }
    
    try:
        with open(os.path.join(output_dir, "dataset_metadata.pkl"), 'rb') as f:
            metadata = pickle.load(f)
        
        test_results['metadata'] = metadata
        
        logger.info("Dataset Metadata:")
        logger.info(f"  Sampling rate: {metadata['config']['sampling_rate']} Hz")
        logger.info(f"  Clip length: {metadata['config']['clip_length_s']} s")
        logger.info(f"  Overlap: {metadata['config']['overlap']}")
        logger.info(f"  Filtering: {metadata['config']['l_freq']}-{metadata['config']['h_freq']} Hz")
        logger.info(f"  Normalization: {metadata['config']['normalization']['method']}")
        
        logger.info("Processing Statistics:")
        stats = metadata['processing_stats']
        logger.info(f"  Total files found: {stats['total_files']}")
        logger.info(f"  Processed files: {stats['processed_files']}")
        logger.info(f"  Skipped (pattern match): {stats.get('skipped_pattern', 0)}")
        logger.info(f"  Skipped (too few spikes): {stats.get('skipped_low_spikes', 0)}")
        logger.info(f"  No-spike files processed: {stats.get('no_spike_files_processed', 0)}")
        logger.info(f"  Augmented segments: {stats.get('augmented_segments', 0)}")
        
        logger.info("Class Distribution:")
        test_results['class_distribution'] = metadata['class_distribution']
        
        for split, distr in metadata['class_distribution'].items():
            total = distr[0] + distr[1]
            if total > 0:
                logger.info(f"  {split.capitalize()}: {distr[1]} spikes ({distr[1]/total*100:.1f}%), "
                           f"{distr[0]} non-spikes ({distr[0]/total*100:.1f}%)")
    except Exception as e:
        logger.info(f"No detailed metadata available: {e}")
        
    # Load patient splits if available
    try:
        with open(os.path.join(output_dir, "patient_splits.pkl"), 'rb') as f:
            patient_splits = pickle.load(f)
        
        with open(os.path.join(output_dir, "patient_groups.pkl"), 'rb') as f:
            patient_groups = pickle.load(f)
            
        test_results['patient_splits'] = patient_splits
        test_results['patient_groups'] = patient_groups

        logger.info("Patient distribution across splits:")
        split_counts = {"train": 0, "val": 0, "test": 0}
        group_counts = defaultdict(lambda: {"train": 0, "val": 0, "test": 0})
        
        for patient, split in patient_splits.items():
            split_counts[split] += 1
            group = patient_groups.get(patient, "Unknown")
            group_counts[group][split] += 1

        for split, count in split_counts.items():
            logger.info(f"  {split.capitalize()}: {count} patients")
            
        logger.info("Patient distribution by group and split:")
        for group, splits in group_counts.items():
            logger.info(f"  {group}:")
            for split, count in splits.items():
                logger.info(f"    {split.capitalize()}: {count} patients")
                
        test_results['patient_distribution'] = {
            'split_counts': split_counts,
            'group_counts': dict(group_counts)
        }
    except FileNotFoundError:
        logger.warning("Patient split information not found")
        
    # Check timing statistics if available
    try:
        with open(os.path.join(output_dir, "timing_stats.json"), 'r') as f:
            timing_stats = json.load(f)
            
        logger.info("Processing Performance:")
        if 'summary' in timing_stats:
            summary = timing_stats['summary']
            logger.info(f"  Total processing time: {summary.get('total_time_minutes', 0):.1f} minutes")
            logger.info(f"  Files per second: {summary.get('files_per_second', 0):.2f}")
            logger.info(f"  Segments per second: {summary.get('segments_per_second', 0):.2f}")
            
        test_results['timing'] = timing_stats
    except FileNotFoundError:
        logger.debug("Timing statistics not found")
        
    # Check channel statistics if available
    try:
        with open(os.path.join(output_dir, "channel_stats.pkl"), 'rb') as f:
            channel_stats = pickle.load(f)
            
        if 'most_informative' in channel_stats and channel_stats['most_informative']:
            logger.info("Most informative channels:")
            for i, ch_info in enumerate(channel_stats['most_informative'][:5]):
                logger.info(f"  {i+1}. Channel {ch_info['channel_idx']}: correlation={ch_info['correlation']:.3f}")
                
        test_results['channel_stats'] = {
            'most_informative': channel_stats.get('most_informative', [])[:10]
        }
    except FileNotFoundError:
        logger.debug("Channel statistics not found")
        
    # Visualize a few examples if requested
    if visualize:
        logger.info("Visualizing sample data...")
        viz_dir = os.path.join(output_dir, "test_visualizations")
        os.makedirs(viz_dir, exist_ok=True)
        
        for name, dataset in [("Train", train_dataset), ("Validation", val_dataset), ("Test", test_dataset)]:
            if len(dataset) == 0:
                continue
                
            # Get samples of each class if possible
            spike_samples = []
            non_spike_samples = []
            
            # Try to find samples of both classes
            for i in np.random.choice(len(dataset), min(100, len(dataset)), replace=False):
                try:
                    result = dataset[i]
                    if len(result) >= 2:
                        data, label = result[0], result[1]
                        label_val = label.item()
                        
                        if label_val == 1 and len(spike_samples) < n_samples:
                            spike_samples.append((data, i))
                        elif label_val == 0 and len(non_spike_samples) < n_samples:
                            non_spike_samples.append((data, i))
                            
                        if len(spike_samples) >= n_samples and len(non_spike_samples) >= n_samples:
                            break
                except Exception as e:
                    logger.warning(f"Error loading sample {i}: {e}")
            
            # Create visualizations
            for sample_type, samples in [("spike", spike_samples), ("non_spike", non_spike_samples)]:
                for idx, (data, sample_idx) in enumerate(samples):
                    try:
                        fig, ax = plt.subplots(figsize=(10, 6))
                        
                        # Plot multiple channels with different colors
                        n_channels = min(8, data.shape[0])  # Limit to 8 channels for clarity
                        time_axis = np.arange(data.shape[1]) / processed_sfreq
                        
                        # Use a color cycle for multiple channels
                        colors = plt.cm.tab10(np.linspace(0, 1, n_channels))
                        
                        for ch_idx in range(n_channels):
                            # Offset each channel for better visualization
                            ax.plot(time_axis, data[ch_idx] + ch_idx * 2, color=colors[ch_idx], 
                                  label=f"Channel {ch_idx}")
                        
                        ax.set_xlabel('Time (s)')
                        ax.set_ylabel('Amplitude (a.u.)')
                        ax.set_title(f"{name} {sample_type.replace('_', ' ')} example (sample {sample_idx})")
                        
                        plt.tight_layout()
                        plt.savefig(os.path.join(viz_dir, f"{name.lower()}_{sample_type}_{idx}.png"))
                        plt.close(fig)
                        
                        # Store sample information
                        if name.lower() not in test_results['sample_data']:
                            test_results['sample_data'][name.lower()] = {}
                        
                        if sample_type not in test_results['sample_data'][name.lower()]:
                            test_results['sample_data'][name.lower()][sample_type] = []
                            
                        test_results['sample_data'][name.lower()][sample_type].append({
                            'sample_idx': int(sample_idx),
                            'shape': tuple(data.shape),
                            'visualization': f"{name.lower()}_{sample_type}_{idx}.png"
                        })
                        
                    except Exception as e:
                        logger.warning(f"Error creating visualization: {e}")
        
        logger.info(f"Saved test visualizations to {viz_dir}")
            
    return test_results


def setup_logging(log_level: str, log_file: Optional[str] = None) -> Dict[str, logging.Logger]:
    """Set up logging configuration.

    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional path to log file

    Returns:
        Dict containing configured loggers
    """
    # Convert string level to logging level
    numeric_level = getattr(logging, log_level.upper(), logging.INFO)

    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(numeric_level)

    # Clear existing handlers to avoid duplicates
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(numeric_level)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    # Create file handler if log file is specified
    if log_file:
        os.makedirs(os.path.dirname(os.path.abspath(log_file)), exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(numeric_level)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)

    # Create specific loggers
    loggers = {
        'main': logging.getLogger('main'),
        'processor': logging.getLogger('processor'),
        'validator': logging.getLogger('validator'),
        'dataset': logging.getLogger('dataset')
    }

    return loggers



def main():
    """Main function to run the MEG preprocessing pipeline."""
    import argparse

    parser = argparse.ArgumentParser(description='MEG Data Preprocessing')
    parser.add_argument('--config', type=str, help='Path to YAML configuration file')
    parser.add_argument('--root_dir', type=str, help='Directory containing .ds files')
    parser.add_argument('--output_dir', type=str, help='Directory to save processed data')
    parser.add_argument('--good_channels_file_path', type=str, help='Path to the file containing the list of good channels')
    parser.add_argument('--loc_meg_channels_file_path', type=str, help='Path to the file containing the list of locations of meg channels')
    parser.add_argument('--interpolate_miss_ch', type=bool, default=True, help='Interpolate missing channels (if false, we drop the missing channels)')
    parser.add_argument('--clip_length_s', type=float, default=1.0, help='Length of each clip in seconds')
    parser.add_argument('--overlap', type=float, default=0.0, help='Fraction of overlap between consecutive clips (0.0 to <1.0)')
    parser.add_argument('--sampling_rate', type=int, default=200, help='Target sampling rate in Hz')
    parser.add_argument('--l_freq', type=float, default=1.0, help='Low-pass filter frequency in Hz')
    parser.add_argument('--h_freq', type=float, default=70.0, help='High-pass filter frequency in Hz')
    parser.add_argument('--notch_freq', type=float, default=50.0, help='Notch filter frequency in Hz')
    parser.add_argument('--train_ratio', type=float, default=0.8, help='Ratio of patients for training')
    parser.add_argument('--val_ratio', type=float, default=0.2, help='Ratio of patients for validation')
    parser.add_argument('--random_state', type=int, default=42, help='Random seed')
    parser.add_argument('--validate', type=str, default=None, help='Path to a specific .ds file to validate after processing')
    parser.add_argument('--test', type=bool, default=False, help='Test the datasets after processing')
    parser.add_argument('--log_level', type=str, default='INFO', help='Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)')
    parser.add_argument('--log_file', type=str, default=None, help='Path to log file')
    
    # Preprocessing options
    parser.add_argument('--min_spikes_per_file', type=int, default=10, help='Minimum number of spikes required to process a file')
    parser.add_argument('--process_no_spike_files', type=bool, default=False, help='Process files with no valid spikes as control data')
    parser.add_argument('--skip_files', type=str, nargs='+', default=[], help='List of filenames or patterns to skip')
    
    # Class balancing options
    parser.add_argument('--target_class_ratio', type=float, default=0.5, help='Target ratio of spike to non-spike samples (0.5 means balanced)')
    parser.add_argument('--balance_method', type=str, default='undersample', choices=['undersample', 'oversample', 'none'], help='Method for class balancing')
    
    # Augmentation options
    parser.add_argument('--augmentation_enabled', type=bool, default=False, help='Enable data augmentation')
    parser.add_argument('--augmentation_time_shift_ms', type=int, default=50, help='Maximum time shift in milliseconds for augmentation')
    parser.add_argument('--augmentation_max_shifts', type=int, default=2, help='Number of time-shifted copies per spike segment')
    parser.add_argument('--augmentation_amplitude_scale', type=bool, default=False, help='Enable amplitude scaling augmentation')
    parser.add_argument('--augmentation_add_noise', type=bool, default=False, help='Enable noise addition augmentation')
    parser.add_argument('--augmentation_channel_dropout', type=bool, default=False, help='Enable channel dropout augmentation')
    
    # Normalization options
    parser.add_argument('--normalization_method', type=str, default='percentile', choices=['percentile', 'zscore', 'minmax'], help='Method for data normalization')
    parser.add_argument('--normalization_percentile', type=int, default=95, help='Percentile value for percentile normalization')
    
    # Performance options
    parser.add_argument('--n_jobs', type=int, default=1, help='Number of parallel jobs to use')
    parser.add_argument('--max_segments_per_file', type=int, default=None, help='Maximum number of segments to extract per file')
    parser.add_argument('--online_saving', type=bool, default=True, help='Save segments immediately after processing to reduce memory usage')
    parser.add_argument('--benchmark', type=bool, default=True, help='Collect timing statistics')
    
    # Additional analysis options
    parser.add_argument('--channel_stats', type=bool, default=True, help='Collect channel statistics')
    parser.add_argument('--validation_output_dir', type=str, default=None, help='Output directory for validation visualizations')
    parser.add_argument('--visualize_test', type=bool, default=True, help='Create visualizations when testing the dataset')

    args = parser.parse_args()

    # Load configuration from YAML if provided
    config = {}
    if args.config:
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
    
    # Merge CLI arguments with config file (CLI arguments take precedence)
    # Convert args to dict, skipping None values (not provided)
    args_dict = {k: v for k, v in vars(args).items() if v is not None and k != 'config'}
    
    # Merge args_dict with config (CLI args take precedence over config)
    for key, value in args_dict.items():
        config[key] = value
    
    # Check if required arguments are provided
    required_args = ['root_dir', 'output_dir', 'good_channels_file_path', 'loc_meg_channels_file_path']
    missing_args = [arg for arg in required_args if arg not in config]
    if missing_args:
        raise ValueError(f"Missing required arguments: {', '.join(missing_args)}")

    mne.set_log_level(verbose=logging.ERROR)  # Suppress MNE output

    # Check if ratios sum to 1
    train_ratio = config.get('train_ratio', 0.8)
    val_ratio = config.get('val_ratio', 0.2)
    
    if abs(train_ratio + val_ratio - 1.0) > 1e-7:
        raise ValueError(f"Train and val ratios must sum to 1. Got {train_ratio + val_ratio}")

    # Check if overlap is valid
    overlap = config.get('overlap', 0.0)
    if overlap < 0.0 or overlap >= 1.0:
        raise ValueError("Overlap must be between 0.0 and less than 1.0")

    # Set up logging
    log_level = config.get('log_level', 'INFO')
    log_file = config.get('log_file', None)
    loggers = setup_logging(log_level, log_file)

    # Create augmentation and normalization configs
    augmentation = {
        'enabled': config.get('augmentation_enabled', False),
        'time_shift_ms': config.get('augmentation_time_shift_ms', 50),
        'max_shifts': config.get('augmentation_max_shifts', 2),
        'preserve_spikes': True,
        'amplitude_scale': config.get('augmentation_amplitude_scale', False),
        'amplitude_range': (0.8, 1.2),
        'add_noise': config.get('augmentation_add_noise', False),
        'noise_level': 0.05,
        'channel_dropout': config.get('augmentation_channel_dropout', False),
        'channel_dropout_rate': 0.1
    }
    
    normalization = {
        'method': config.get('normalization_method', 'percentile'),
        'percentile': config.get('normalization_percentile', 95),
        'per_channel': True,
        'per_segment': False,
    }

    # Log all configuration parameters
    loggers['main'].info("Configuration:")
    for key, value in config.items():
        loggers['main'].info(f"  {key}: {value}")
        
    # Validate augmentation parameters if enabled
    if config.get('augmentation_enabled', False):
        # Check if time shift is too large for clip length
        clip_length_ms = config.get('clip_length_s', 1.0) * 1000
        time_shift_ms = config.get('augmentation_time_shift_ms', 50)
        
        if time_shift_ms > clip_length_ms / 2:
            loggers['main'].warning(
                f"Augmentation time shift ({time_shift_ms}ms) is too large for clip length ({clip_length_ms}ms). "
                f"Consider reducing to at most {clip_length_ms/2}ms."
            )
        
        # Check if max shifts is too high
        max_shifts = config.get('augmentation_max_shifts', 2)
        if max_shifts > 3:
            loggers['main'].warning(
                f"Augmentation max shifts ({max_shifts}) is quite high. Consider reducing to 2-3."
            )

    # Initialize and run preprocessor
    loggers['main'].info("Starting MEG preprocessing pipeline...")
    preprocessor = MEGPreprocessor(
        root_dir=config['root_dir'],
        output_dir=config['output_dir'],
        good_channels_file_path=config['good_channels_file_path'],
        loc_meg_channels_file_path=config['loc_meg_channels_file_path'],
        clip_length_s=config.get('clip_length_s', 1.0),
        overlap=overlap,
        sampling_rate=config.get('sampling_rate', 200),
        l_freq=config.get('l_freq', 1.0),
        h_freq=config.get('h_freq', 70.0),
        notch_freq=config.get('notch_freq', 50.0),
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        random_state=config.get('random_state', 42),
        min_spikes_per_file=config.get('min_spikes_per_file', 10),
        process_no_spike_files=config.get('process_no_spike_files', False),
        skip_files=config.get('skip_files', []),
        target_class_ratio=config.get('target_class_ratio', 0.5),
        balance_method=config.get('balance_method', 'undersample'),
        augmentation=augmentation,
        normalization=normalization,
        n_jobs=config.get('n_jobs', 1),
        max_segments_per_file=config.get('max_segments_per_file', None),
        online_saving=config.get('online_saving', True),
        benchmark=config.get('benchmark', True),
        channel_stats=config.get('channel_stats', True),
        logger=loggers['processor']
    )

    loggers['main'].info("Processing all files...")
    preprocessor.process_all(config.get('interpolate_miss_ch', True))

    # Validate a specific file if requested
    if 'validate' in config and config['validate']:
        loggers['main'].info("Validating by reconstructing a given .ds...")
        validate_dataset(
            raw_data_file_path=Path(config['validate']),
            processed_data_root=Path(config['output_dir']),
            processed_sfreq=config.get('sampling_rate', 200),
            good_channels_file_path=Path(config['good_channels_file_path']),
            loc_meg_channels_file_path=Path(config['loc_meg_channels_file_path']),
            logger=loggers['validator'],
            output_dir=Path(config.get('validation_output_dir', config['output_dir']))
        )

    # Test the datasets if requested
    if config.get('test', False):
        loggers['main'].info("Testing datasets by getting sample's distribution...")
        test_results = test_datasets(
            config['output_dir'], 
            logger=loggers['dataset'],
            visualize=config.get('visualize_test', True)
        )
        
        # Save test results
        test_results_file = os.path.join(config['output_dir'], "test_results.json")
        try:
            # Convert to JSON-serializable format
            serializable_results = {}
            for key, value in test_results.items():
                if isinstance(value, dict):
                    serializable_results[key] = value
                else:
                    serializable_results[key] = str(value)
                    
            with open(test_results_file, 'w') as f:
                json.dump(serializable_results, f, indent=2)
            loggers['main'].info(f"Saved test results to {test_results_file}")
        except Exception as e:
            loggers['main'].warning(f"Failed to save test results: {e}")
            
    loggers['main'].info("Pipeline execution completed successfully.")


if __name__ == "__main__":
    main()