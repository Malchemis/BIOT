import mne
import numpy as np
import os
import pickle
import torch
import logging
from pathlib import Path
from typing import List, Tuple, Dict, Optional
from tqdm import tqdm

from mne_interpolate import interpolate_missing_channels


class MEGDataset(torch.utils.data.Dataset):
    """PyTorch Dataset for loading preprocessed MEG data.

    This dataset loads preprocessed MEG data from disk, stored as PyTorch tensors.
    Each data sample contains the MEG signal and its associated label.

    Attributes:
        root: Root directory containing the processed data.
        files: List of filenames to use.
    """

    def __init__(self, root: str, files: List[str]):
        """Initialize the MEG dataset.

        Args:
            root: Root directory containing the processed data.
            files: List of filenames to use.
        """
        self.root = root
        self.files = files

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
        sample = torch.load(file_path)
        return sample['data'], sample['label']


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

    Attributes:
        root_dir: Directory containing the .ds files.
        output_dir: Directory to save the processed data.
        clip_length_s: Length of each clip in seconds.
        overlap: Fraction of overlap between consecutive clips.
        sampling_rate: Target sampling rate in Hz.
        train_ratio: Ratio of patients to use for training.
        val_ratio: Ratio of patients to use for validation.
        test_ratio: Ratio of patients to use for testing.
        random_state: Random seed for reproducibility.
        clip_length_samples: Length of each clip in samples.
        step_size: Step size in samples between consecutive clips.
        processed_files: Dictionary of processed files for each split.
        patient_splits: Dictionary mapping patient IDs to their data split.
        logger: Logger for this class.
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
            train_ratio: float = 0.7,
            val_ratio: float = 0.15,
            test_ratio: float = 0.15,
            random_state: int = 42,
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
            train_ratio: Ratio of patients to use for training.
            val_ratio: Ratio of patients to use for validation.
            test_ratio: Ratio of patients to use for testing.
            random_state: Random seed for reproducibility.
            logger: Logger instance for this class.

        Raises:
            ValueError: If the overlap is invalid or if the split ratios don't sum to 1.
        """
        self.root_dir = root_dir
        self.output_dir = output_dir
        self.clip_length_s = clip_length_s
        self.overlap = overlap
        self.sampling_rate = sampling_rate
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.random_state = random_state

        # Set up logging
        self.logger = logger or logging.getLogger(__name__)

        # Check if ratios sum to 1
        if abs(self.train_ratio + self.val_ratio + self.test_ratio - 1.0) > 1e-5:
            error_msg = f"Train, val, and test ratios must sum to 1. Got {self.train_ratio + self.val_ratio + self.test_ratio}"
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

        # Load the list of good channels
        self.logger.info(f"Loading good channels from {good_channels_file_path}")
        with open(good_channels_file_path, 'rb') as f:
            self.good_channels = pickle.load(f)

        # Load the channel locations
        self.logger.info(f"Loading channel locations from {loc_meg_channels_file_path}")
        with open(loc_meg_channels_file_path, 'rb') as f:
            self.loc_meg_channels = pickle.load(f)

        # Create output directories
        for split in ['train', 'val', 'test']:
            split_dir = os.path.join(output_dir, split)
            os.makedirs(split_dir, exist_ok=True)
            self.logger.debug(f"Created output directory: {split_dir}")

        # Keep track of processed files
        self.processed_files = {'train': [], 'val': [], 'test': []}
        # Keep track of which patient goes into which split
        self.patient_splits = {}

    # Main method
    def process_all(self, interpolate: bool) -> None:
        """Process all .ds files in the root directory.

        This method:
        1. Splits patients across train/val/test sets
        2. Processes each .ds file
        3. Saves the processed segments to the appropriate split folder

        Args:
            interpolate: bool, whether to interpolate missing channels or drop them.

        Raises:
            ValueError: If no .ds files are found.
        """
        ds_files = self.find_ds_files()

        if not ds_files:
            error_msg = f"No .ds files found in {self.root_dir}"
            self.logger.error(error_msg)
            raise ValueError(error_msg)

        # Split patients before processing
        self.split_patients()

        for file_path in tqdm(ds_files, desc="Processing files"):
            # get patient_id by same procedure as split
            patient_id = os.path.basename(file_path).split('.')[0].split('_')[0]

            # Determine which split this patient belongs to
            split = self.patient_splits.get(patient_id, 'train')  # Default to train if not found
            self.logger.info(f"Processing file {file_path} for patient {patient_id} (split: {split})")

            # Process the file
            segments, labels, start_positions = self.preprocess_file(file_path, interpolate=interpolate)

            if segments is not None and labels is not None and start_positions is not None:
                # Save the processed segments
                self.save_segments(segments, labels, start_positions, patient_id,
                                   os.path.basename(file_path).split('.')[0], split)
            else:
                self.logger.warning(f"Failed to process file {file_path} - no data returned")

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

        self.logger.info(f"Processing complete. Files saved to {self.output_dir}")
        self.logger.info(f"Train: {len(self.processed_files['train'])} files")
        self.logger.info(f"Val: {len(self.processed_files['val'])} files")
        self.logger.info(f"Test: {len(self.processed_files['test'])} files")

    def find_ds_files(self) -> List[str]:
        """Find all .ds directories in the root directory.

        Returns:
            List of paths to .ds directories.
        """
        ds_files = []
        self.logger.debug(f"Searching for .ds files in {self.root_dir}")
        for item in os.listdir(self.root_dir):
            if item.endswith('.ds') and os.path.isdir(os.path.join(self.root_dir, item)):
                ds_files.append(os.path.join(self.root_dir, item))

        self.logger.info(f"Found {len(ds_files)} .ds files")
        return ds_files

    def split_patients(self) -> Dict[str, List[str]]:
        """Split patients into train, validation, and test sets.

        This ensures that all data from one patient stays in the same split,
        preventing data leakage between splits.

        Returns:
            Dictionary mapping split names to lists of patient IDs.
        """
        ds_files = self.find_ds_files()
        # We split on '.' to remove the extension. If the patient only has one recording, then we don't need to split further
        # But if the patient has had multiple sessions, we need to split on '_' to get patient ids.
        patient_ids = [os.path.basename(file_path).split('.')[0].split('_')[0] for file_path in ds_files]

        # Get unique patient IDs (in case multiple recordings from same patient)
        unique_patients = list(set(patient_ids))
        self.logger.info(f"Found {len(unique_patients)} unique patients from {len(ds_files)} files")

        # Shuffle patients
        np.random.seed(self.random_state)
        np.random.shuffle(unique_patients)

        # Calculate split indices
        n_patients = len(unique_patients)
        n_train = int(n_patients * self.train_ratio)
        n_val = int(n_patients * self.val_ratio)

        # Split patients
        train_patients = unique_patients[:n_train]
        val_patients = unique_patients[n_train:n_train + n_val]
        test_patients = unique_patients[n_train + n_val:]

        # Store patient splits
        for patient in train_patients:
            self.patient_splits[patient] = 'train'
        for patient in val_patients:
            self.patient_splits[patient] = 'val'
        for patient in test_patients:
            self.patient_splits[patient] = 'test'

        self.logger.info(f"Patient split: {len(train_patients)} train, {len(val_patients)} val, {len(test_patients)} test")
        return {'train': train_patients, 'val': val_patients, 'test': test_patients}

    def preprocess_file(self, file_path: str, interpolate: bool) -> Tuple[
        Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
        """Preprocess a single .ds file.

        Args:
            file_path: Path to the .ds file.
            interpolate: If True, interpolate missing channels. Else drops to keep only intersection of valid channels.

        Returns:
            Tuple containing:
            - segments: Preprocessed segments of shape (n_segments, n_channels, clip_length_samples)
            - labels: Labels for each segment (1 for spike, 0 for non-spike)
            - start_positions: Original starting position of each segment in samples
            Returns None for all values if processing fails.
        """
        self.logger.info(f"Processing {file_path}")

        # Load the raw data
        try:
            raw_file = mne.io.read_raw_ctf(file_path, preload=True)
        except Exception as e:
            self.logger.error(f"Error loading {file_path}: {e}")
            return None, None, None

        # Keep only the MEG data (no references, eeg, eog, etc.)
        raw_file = raw_file.pick('meg')

        if interpolate:
            # Interpolate missing channels
            self.logger.debug("Interpolating missing channels")
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
        self.logger.debug("Applying bandpass filter (1-70Hz)")
        raw_file.filter(l_freq=1, h_freq=70)

        # Apply notch filter (50Hz, power line interference)
        self.logger.debug("Applying notch filter (50Hz)")
        raw_file.notch_filter(freqs=50)

        # Get the data
        meg_data = raw_file.get_data()
        self.logger.debug(f"Data shape after preprocessing: {meg_data.shape}")

        # Normalize channel-wise using 95th percentile
        q95 = np.quantile(np.abs(meg_data), q=0.95, axis=-1, keepdims=True)
        meg_data = meg_data / (q95 + 1e-8)  # we add a small value to prevent division by 0.
        self.logger.debug("Normalized data using 95th percentile")

        # Get annotations
        annotations = raw_file.annotations
        if len(annotations) == 0:
            self.logger.warning(f"No annotations found in {file_path}")
            return None, None, None

        # Extract spike annotations
        spike_onsets = []
        for i in range(len(annotations)):
            if 'spike' in annotations[i]['description'].lower():
                onset_sample = int(annotations[i]['onset'] * self.sampling_rate)
                spike_onsets.append(onset_sample)
        self.logger.info(f"Found {len(spike_onsets)} spike annotations in {file_path}")

        # Get total length in samples
        n_samples = meg_data.shape[1]

        segments = []
        labels = []
        start_positions = []  # Track the original starting position of each segment

        # Cut the data into overlapping segments
        start_samples = range(0, n_samples - self.clip_length_samples + 1, self.step_size)
        for start in start_samples:
            end = start + self.clip_length_samples
            segment = meg_data[:, start:end]  # take the segment across channels

            # Check if any spike occurs in this segment
            has_spike = any(start <= spike_onset < end for spike_onset in spike_onsets)
            segments.append(segment)
            labels.append(1 if has_spike else 0)

            # Store original position in samples (at original sampling rate)
            original_start = int(
                start * original_sfreq / self.sampling_rate)  # divide by the new sr and then multiply by old sr to get by the original starting positions.
            start_positions.append(original_start)

        self.logger.info(f"Created {len(segments)} segments, {sum(labels)} with spikes")
        # store the produced segments, labels, and positions for reconstruction if necessary.
        return np.array(segments), np.array(labels), np.array(start_positions)

    def save_segments(self,
                      segments: np.ndarray,
                      labels: np.ndarray,
                      start_positions: np.ndarray,
                      patient_id: str,
                      filename_origin: str,
                      split: str) -> None:
        """Save preprocessed segments to disk.

        Args:
            segments: Preprocessed segments.
            labels: Labels for each segment.
            start_positions: Original starting positions of each segment.
            patient_id: Patient identifier.
            filename_origin: The name of the original filename
            split: Data split ('train', 'val', or 'test').
        """
        if segments.shape[0] == 0:
            self.logger.warning(f"No segments found for patient {patient_id}")
            return

        self.logger.info(f"Saving {segments.shape[0]} segments for patient {patient_id} to {split} split")

        # Save each segment
        for i in range(segments.shape[0]):
            file_prefix = f"{patient_id}_{filename_origin}_{i:04d}"  # i:04d formats the output to have 4 numbers like 0001, 0002,...
            file_prefix = file_prefix.replace('__', '_')  # adding the origin might double the underscores
            file_path = os.path.join(self.output_dir, split, f"{file_prefix}.pt")

            torch.save({
                'data': torch.tensor(segments[i], dtype=torch.float32),
                'label': torch.tensor(labels[i], dtype=torch.long),
                'patient_id': patient_id,
                'original_filename': filename_origin,
                'start_position': int(start_positions[i]),
                'original_index': i
            }, file_path)

            self.processed_files[split].append(
                f"{file_prefix}.pt")  # keep information about the processed files locations

        self.logger.debug(f"Saved {segments.shape[0]} segments for patient {patient_id}")


def validate_dataset(raw_data_file_path: Path,
                     processed_data_root: Path,
                     processed_sfreq: int = 200,
                     logger: Optional[logging.Logger] = None,
                     show_plot=False) -> None:
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
        logger: Logger instance
        show_plot: If True, display the plot

    Raises:
        ValueError: If no processed fragments are found for the patient
    """
    logger = logger or logging.getLogger(__name__)
    import matplotlib
    import matplotlib.pyplot as plt
    matplotlib.use('TkAgg')

    # Extract patient ID from basename
    patient_id_plus_session = os.path.basename(raw_data_file_path).split('.')[0]
    logger.info(f"Validating patient: {patient_id_plus_session}")

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

    # Extract spike annotations
    spike_times = []
    for annot in raw_file.annotations:
        if 'spike' in annot['description'].lower():
            onset_sample = int(annot['onset'] * orig_sfreq)
            spike_times.append(annot['onset'])
            # Mark a window around the spike
            window_samples = int(1.0 * orig_sfreq)  # 1-second window
            start = max(0, onset_sample - window_samples // 2)
            end = min(n_samples, onset_sample + window_samples // 2)
            original_binary[start:end] = 1

    logger.info(f"Found {len(spike_times)} spike annotations in original file")

    # Find all processed fragments for this patient
    logger.info("Looking for processed fragments...")
    all_fragments = []
    for split in ['train', 'val', 'test']:
        split_dir = os.path.join(processed_data_root, split)
        if os.path.exists(split_dir):
            for file in os.listdir(split_dir):
                if patient_id_plus_session in file and file.endswith('.pt'):
                    fragment_path = os.path.join(split_dir, file)
                    try:
                        fragment = torch.load(fragment_path)
                        all_fragments.append(fragment)
                    except Exception as e:
                        logger.error(f"Error loading {fragment_path}: {e}")

    if not all_fragments:
        error_msg = f"No processed fragments found for patient {patient_id_plus_session}"
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
        for spike_time in spike_times:
            spike_sample = int(spike_time * processed_sfreq)
            window_start = max(0, spike_sample - clip_length_samples // 2)
            window_end = min(len(reconstructed_binary), spike_sample + clip_length_samples // 2)
            reconstructed_binary[window_start:window_end] = 1

    # Create the plots
    logger.info("Creating validation plots")
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 8), sharex=True)

    # Plot original binary signal
    time_orig = np.arange(n_samples) / orig_sfreq
    ax1.plot(time_orig, original_binary, 'b-', label='Original')
    ax1.set_title(f'Original MEG Recording: {patient_id_plus_session} ({len(spike_times)} spikes)')
    ax1.set_ylabel('Spike Present')

    # Add vertical lines for exact spike times
    for t in spike_times:
        ax1.axvline(x=t, color='r', linestyle='--', alpha=0.5)

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
    plot_path = os.path.join(processed_data_root, f"{patient_id_plus_session}_validation.png")
    plt.savefig(plot_path)
    logger.info(f"Saved validation plot to {plot_path}")
    if show_plot:
        plt.show()

    # Report statistics
    spike_fragments = sum(1 for f in all_fragments if f['label'] == 1)
    orig_spike_percentage = (np.sum(original_binary) / len(original_binary)) * 100
    frag_spike_percentage = (spike_fragments / len(all_fragments)) * 100

    logger.info("Validation Summary:")
    logger.info(f"Original recording: {duration_s:.2f}s, {len(spike_times)} annotated spikes ({orig_spike_percentage:.2f}% of time)")
    logger.info(f"Processed fragments: {len(all_fragments)} total, {spike_fragments} with spikes ({frag_spike_percentage:.2f}%)")


def test_datasets(output_dir: str, logger: Optional[logging.Logger] = None) -> None:
    """Test the MEG datasets to ensure they're correctly loaded.

    This function:
    1. Loads the file lists for each split
    2. Creates datasets for each split
    3. Tests loading a sample from each dataset
    4. Reports class distribution statistics

    Args:
        output_dir: Directory containing the processed data
        logger: Logger instance
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
        return

    # Create datasets
    train_dataset = MEGDataset(
        root=os.path.join(output_dir, 'train'),
        files=train_files
    )
    val_dataset = MEGDataset(
        root=os.path.join(output_dir, 'val'),
        files=val_files
    )
    test_dataset = MEGDataset(
        root=os.path.join(output_dir, 'test'),
        files=test_files
    )

    # Test loading a sample from each dataset
    for name, dataset in [("Train", train_dataset), ("Validation", val_dataset), ("Test", test_dataset)]:
        logger.info(f"{name} Dataset:")
        logger.info(f"Number of samples: {len(dataset)}")

        if len(dataset) > 0:
            try:
                data, label = dataset[0]
                logger.info(f"Sample data shape: {data.shape}")
                logger.info(f"Sample label: {label}")

                # Report class distribution
                labels = [dataset[i][1].item() for i in range(len(dataset))]
                unique_labels, counts = np.unique(labels, return_counts=True)
                logger.info("Class distribution:")
                for lbl, cnt in zip(unique_labels, counts):
                    logger.info(f"  Class {lbl}: {cnt} samples ({cnt / len(labels) * 100:.2f}%)")
            except Exception as e:
                logger.error(f"Error testing {name} dataset: {e}")
        else:
            logger.warning(f"No samples found in {name} dataset")

    # Load patient splits if available
    try:
        with open(os.path.join(output_dir, "patient_splits.pkl"), 'rb') as f:
            patient_splits = pickle.load(f)

        logger.info("Patient distribution across splits:")
        split_counts = {"train": 0, "val": 0, "test": 0}
        for patient, split in patient_splits.items():
            split_counts[split] += 1

        for split, count in split_counts.items():
            logger.info(f"  {split.capitalize()}: {count} patients")
    except FileNotFoundError:
        logger.warning("Patient split information not found")


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
    parser.add_argument('--root_dir', type=str, required=True, help='Directory containing .ds files')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to save processed data')
    parser.add_argument('--good_channels_file_path', type=str, required=True, help='Path to the file containing the list of good channels')
    parser.add_argument('--loc_meg_channels_file_path', type=str, required=True, help='Path to the file containing the list of locations of meg channels')
    parser.add_argument('--interpolate_miss_ch', action='store_true', help='Interpolate missing channels (if false, we drop the missing channels)')
    parser.add_argument('--clip_length_s', type=float, default=1.0, help='Length of each clip in seconds')
    parser.add_argument('--overlap', type=float, default=0.5, help='Fraction of overlap between consecutive clips (0.0 to <1.0)')
    parser.add_argument('--sampling_rate', type=int, default=200, help='Target sampling rate in Hz')
    parser.add_argument('--train_ratio', type=float, default=0.5, help='Ratio of patients for training')
    parser.add_argument('--val_ratio', type=float, default=0.25, help='Ratio of patients for validation')
    parser.add_argument('--test_ratio', type=float, default=0.25, help='Ratio of patients for testing')
    parser.add_argument('--random_state', type=int, default=42, help='Random seed')
    parser.add_argument('--validate', type=str, default=None, help='Path to a specific .ds file to validate after processing')
    parser.add_argument('--test', action='store_true', help='Test the datasets after processing')
    parser.add_argument('--log_level', type=str, default='INFO', help='Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)')
    parser.add_argument('--log_file', type=str, default=None, help='Path to log file')

    args = parser.parse_args()

    mne.set_log_level(verbose=logging.ERROR)  # Suppress MNE output

    # Check if ratios sum to 1
    if abs(args.train_ratio + args.val_ratio + args.test_ratio - 1.0) > 1e-5:
        raise ValueError(
            f"Train, val, and test ratios must sum to 1. Got {args.train_ratio + args.val_ratio + args.test_ratio}")

    # Check if overlap is valid
    if args.overlap < 0.0 or args.overlap >= 1.0:
        raise ValueError("Overlap must be between 0.0 and less than 1.0")

    # Set up logging
    loggers = setup_logging(args.log_level, args.log_file)

    # Initialize and run preprocessor
    loggers['main'].info("Starting MEG preprocessing pipeline...")
    preprocessor = MEGPreprocessor(
        root_dir=args.root_dir,
        output_dir=args.output_dir,
        good_channels_file_path=args.good_channels_file_path,
        loc_meg_channels_file_path=args.loc_meg_channels_file_path,
        clip_length_s=args.clip_length_s,
        overlap=args.overlap,
        sampling_rate=args.sampling_rate,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        random_state=args.random_state,
        logger=loggers['processor']
    )

    loggers['main'].info("Processing all files...")
    preprocessor.process_all(args.interpolate_miss_ch)

    # Validate a specific file if requested
    if args.validate:
        loggers['main'].info("Validating by reconstructing a given .ds...")
        validate_dataset(
            raw_data_file_path=Path(args.validate),
            processed_data_root=Path(args.output_dir),
            processed_sfreq=args.sampling_rate,
            logger=loggers['validator']
        )

    # Test the datasets if requested
    if args.test:
        loggers['main'].info("Testing datasets by getting sample's distribution...")
        test_datasets(args.output_dir, logger=loggers['dataset'])


if __name__ == "__main__":
    # Example arguments
    # args = {
    #     "root_dir": "/home/malchemis/PycharmProjects/bio-sig-analysis/data/raw/crnl-meg/sample-data",
    #     "output_dir": "/home/malchemis/PycharmProjects/bio-sig-analysis/data/processed/crnl-meg",
    #     "good_channels_file_path": "/home/malchemis/PycharmProjects/BIOT/datasets/CUSTOM/good_channels",
    #     "loc_meg_channels_file_path": "/home/malchemis/PycharmProjects/BIOT/datasets/CUSTOM/loc_meg_channels.pkl",
    #     "interpolate_miss_ch": True
    #     "clip_length_s": 1.0,
    #     "overlap": 0.5,
    #     "sampling_rate": 200,
    #     "train_ratio": 0.7,
    #     "val_ratio": 0.15,
    #     "test_ratio": 0.15,
    #     "random_state": 42,
    #     "validate": '/home/malchemis/PycharmProjects/bio-sig-analysis/data/raw/crnl-meg/sample-data/chada_Epi-001_20070124_03.ds',
    #     "test": True
    #     "log_level": "INFO",
    #     "log_file": "/home/malchemis/PycharmProjects/bio-sig-analysis/logs/meg_preprocessing.log"
    # }
    main()