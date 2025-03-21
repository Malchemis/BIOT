import random
import os
import argparse
import logging
from datetime import datetime

# # we want to debug so we set
# For debugging consider passing CUDA_LAUNCH_BLOCKING=1
# Compile with `TORCH_USE_CUDA_DSA` to enable device-side assertions.
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["TORCH_USE_CUDA_DSA"] = "1"

import torch
import numpy as np

import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pyhealth.metrics import binary_metrics_fn

from model import (
    SPaRCNet,
    ContraWR,
    CNNTransformer,
    FFCL,
    STTransformer,
    BIOTClassifier,
)
from utils import MEGDataset, TUABLoader, CHBMITLoader, PTBLoader, focal_loss, focal_loss_with_class_weights, BCE, weighted_BCE

import torch
import numpy as np
from sklearn.metrics import precision_score, recall_score, confusion_matrix, f1_score, accuracy_score

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
                if (pred[ind] == 1 or pred[ind+1] == 1):
                    relaxed_pred[ind] = 1
            elif pred[ind] == 1 and true[ind+1] == 0:
                relaxed_pred[ind] = 1
        # Right boundary condition
        elif ind == true.shape[0] - 1:
            if l == 1:
                if (pred[ind] == 1 or pred[ind-1] == 1):
                    relaxed_pred[ind] = 1
            elif pred[ind] == 1 and true[ind-1] == 0:
                relaxed_pred[ind] = 1        
        # General case
        elif l == 1:
            if (pred[ind-1] == 1 or pred[ind] == 1 or pred[ind+1] == 1):
                relaxed_pred[ind] = 1
        elif pred[ind] == 1 and true[ind+1] == 0 and true[ind-1] == 0:
            relaxed_pred[ind] = 1

    # Calculate relaxed metrics
    relaxed_tp = len(np.intersect1d(np.where(true == 1), np.where(relaxed_pred == 1)))
    relaxed_tn = len(np.intersect1d(np.where(true == 0), np.where(relaxed_pred == 0)))
    relaxed_fp = len(np.intersect1d(np.where(true == 0), np.where(relaxed_pred == 1)))
    relaxed_fn = len(np.intersect1d(np.where(true == 1), np.where(relaxed_pred == 0)))
    
    # Calculate metrics, handling edge cases
    relaxed_f1 = (2*relaxed_tp)/(2*relaxed_tp+relaxed_fp+relaxed_fn) if (2*relaxed_tp+relaxed_fp+relaxed_fn) else 1
    relaxed_precision = (relaxed_tp / (relaxed_tp+relaxed_fp)) if (relaxed_tp+relaxed_fp) else 1
    relaxed_recall = relaxed_sens = (relaxed_tp / (relaxed_tp+relaxed_fn)) if (relaxed_tp+relaxed_fn) else 1
    relaxed_spec = (relaxed_tn / (relaxed_tn+relaxed_fp)) if (relaxed_tn+relaxed_fp) else 1
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

class LitModel_finetune(pl.LightningModule):
    """PyTorch Lightning module for fine-tuning binary classification models.
    
    This class wraps various neural network models to enable training, validation,
    and testing with PyTorch Lightning.
    
    Args:
        args (argparse.Namespace): Command-line arguments
        model: The neural network model to use
    """
    def __init__(self, args: argparse.Namespace, model):
        super().__init__()
        self.model = model
        self.threshold = 0.5
        self.args = args
        self.validation_step_outputs = []
        self.test_step_outputs = []
        self.custom_logger = logging.getLogger(__name__)

    def training_step(self, batch, batch_idx):
        """Perform a single training step.
        
        Args:
            batch: A batch of data (inputs and targets)
            batch_idx: Index of the batch
            
        Returns:
            The calculated loss
        """
        X, y = batch
        prob = self.model(X)
        
        # Option 1: Original BCE loss (no weighting)
        loss = BCE(prob, y)
        
        # Option 2: Weighted BCE with class-specific weights
        # loss = weighted_BCE(prob, y, self.class_weights)
        
        # Option 3: Original focal loss
        # loss = focal_loss(prob, y, alpha=0.8, gamma=0.7)
        
        # Option 4: Focal loss with class weights
        # loss = focal_loss_with_class_weights(prob, y, self.class_weights, gamma=2.0)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        """Perform a single validation step.
        
        Args:
            batch: A batch of data (inputs and targets)
            batch_idx: Index of the batch
            
        Returns:
            Tuple of predictions and ground truth
        """
        X, y = batch
        with torch.no_grad():
            prob = self.model(X)
            step_result = torch.sigmoid(prob).cpu().numpy()
            step_gt = y.cpu().numpy()
        # Append the result to the list
        self.validation_step_outputs.append((step_result, step_gt))
        return step_result, step_gt

    def on_validation_epoch_start(self):
        """Initialize validation outputs collection at the start of validation."""
        self.validation_step_outputs = []

    def on_validation_epoch_end(self):
        """Process validation outputs at the end of validation epoch with scikit-learn optimized threshold."""
        from sklearn.metrics import precision_recall_curve
        
        val_step_outputs = self.validation_step_outputs
        result = np.array([])
        gt = np.array([])
        for out in val_step_outputs:
            result = np.append(result, out[0])
            gt = np.append(gt, out[1])

        if sum(gt) * (len(gt) - sum(gt)) != 0:  # to prevent all 0 or all 1 cases
            # Get precision, recall, and thresholds from precision-recall curve
            precision, recall, pr_thresholds = precision_recall_curve(gt, result)
            
            # Calculate F1 score for each threshold
            # Convert to list and handle the case where pr_thresholds has one less element than precision/recall
            pr_thresholds = np.append(pr_thresholds, 1.0)
            f1_scores = [2 * p * r / (p + r + 1e-10) for p, r in zip(precision, recall)]
            
            # Find threshold with highest F1 score
            best_idx = np.argmax(f1_scores)
            self.threshold = pr_thresholds[best_idx]
            
            # Calculate and log metrics using the selected threshold
            result_metrics = binary_metrics_fn(
                gt,
                result,
                metrics=["pr_auc", "roc_auc", "accuracy", "balanced_accuracy", "f1"],
                threshold=self.threshold,
            )
            
            # Additional insight: log the performance at a standard 0.5 threshold for comparison
            standard_metrics = binary_metrics_fn(
                gt,
                result,
                metrics=["accuracy", "balanced_accuracy", "f1"],
                threshold=0.5,
            )
            self.log("val_f1_standard", standard_metrics["f1"], sync_dist=True)
            self.log("val_acc_standard", standard_metrics["accuracy"], sync_dist=True)
        else:
            result_metrics = {
                "accuracy": 0.0,
                "balanced_accuracy": 0.0,
                "pr_auc": 0.0,
                "roc_auc": 0.0,
                "f1": 0.0,
            }
            self.threshold = 0.5  # Default threshold if only one class is present
        
        # Log all metrics for TensorBoard
        self.log("val_acc", result_metrics["accuracy"], sync_dist=True)
        self.log("val_bacc", result_metrics["balanced_accuracy"], sync_dist=True)
        self.log("val_pr_auc", result_metrics["pr_auc"], sync_dist=True)
        self.log("val_auroc", result_metrics["roc_auc"], sync_dist=True)
        self.log("val_f1", result_metrics["f1"], sync_dist=True)
        self.log("best_threshold", self.threshold, sync_dist=True)
        
        self.custom_logger.info(f"Validation metrics: {result_metrics}")
        self.custom_logger.info(f"Optimal threshold: {self.threshold:.4f}")

    def test_step(self, batch, batch_idx):
        """Perform a single test step.
        
        Args:
            batch: A batch of data (inputs and targets)
            batch_idx: Index of the batch
            
        Returns:
            Tuple of predictions and ground truth
        """
        X, y = batch
        with torch.no_grad():
            convScore = self.model(X)
            step_result = torch.sigmoid(convScore).cpu().numpy()
            step_gt = y.cpu().numpy()
        # Append to the test outputs list
        self.test_step_outputs.append((step_result, step_gt))
        return step_result, step_gt

    def on_test_epoch_start(self):
        """Initialize test outputs collection at the start of testing."""
        self.test_step_outputs = []

    def on_test_epoch_end(self):
        """Process test outputs at the end of test epoch with extended metrics."""
        test_step_outputs = self.test_step_outputs
        result = np.array([])
        gt = np.array([])
        for out in test_step_outputs:
            result = np.append(result, out[0])
            gt = np.append(gt, out[1])
        
        # Convert continuous predictions to binary using the best threshold
        binary_pred = (result >= self.threshold).astype(int)
        
        metrics = {}
        if sum(gt) * (len(gt) - sum(gt)) != 0:  # to prevent all 0 or all 1 and raise the AUROC error
            # Calculate standard metrics
            metrics = binary_metrics_fn(
                gt,
                result,
                metrics=["pr_auc", "roc_auc", "accuracy", "balanced_accuracy", "f1"],
                threshold=self.threshold,
            )
            
            # Calculate precision and recall manually
            metrics["precision"] = precision_score(gt, binary_pred)
            metrics["recall"] = recall_score(gt, binary_pred)
            
            # Calculate confusion matrix (TP, FP, TN, FN)
            tn, fp, fn, tp = confusion_matrix(gt, binary_pred, labels=[0, 1]).ravel()
            metrics["true_positives"] = int(tp)
            metrics["false_positives"] = int(fp)
            metrics["true_negatives"] = int(tn)
            metrics["false_negatives"] = int(fn)
            
            # Calculate relaxed scores
            relaxed_metrics = relaxed_scores(gt, binary_pred)
            
            # Add relaxed metrics to the metrics dict
            metrics.update(relaxed_metrics)
            
        else:
            # Default metrics if only one class is present
            metrics = {
                "accuracy": 0.0,
                "balanced_accuracy": 0.0,
                "pr_auc": 0.0,
                "roc_auc": 0.0,
                "f1": 0.0,
                "precision": 0.0,
                "recall": 0.0,
                "true_positives": 0,
                "false_positives": 0,
                "true_negatives": 0,
                "false_negatives": 0,
                "relaxed_f1": 0.0,
                "relaxed_precision": 0.0,
                "relaxed_recall": 0.0,
                "relaxed_sensitivity": 0.0,
                "relaxed_specificity": 0.0,
                "relaxed_accuracy": 0.0,
                "relaxed_tp": 0,
                "relaxed_tn": 0,
                "relaxed_fp": 0,
                "relaxed_fn": 0
            }
        
        # Log all metrics for TensorBoard
        self.log("test_acc", metrics["accuracy"], sync_dist=True)
        self.log("test_bacc", metrics["balanced_accuracy"], sync_dist=True)
        self.log("test_pr_auc", metrics["pr_auc"], sync_dist=True)
        self.log("test_auroc", metrics["roc_auc"], sync_dist=True)
        self.log("test_f1", metrics["f1"], sync_dist=True)
        self.log("test_precision", metrics["precision"], sync_dist=True)
        self.log("test_recall", metrics["recall"], sync_dist=True)
        
        # Log confusion matrix metrics
        self.log("test_true_positives", metrics["true_positives"], sync_dist=True)
        self.log("test_false_positives", metrics["false_positives"], sync_dist=True)
        self.log("test_true_negatives", metrics["true_negatives"], sync_dist=True)
        self.log("test_false_negatives", metrics["false_negatives"], sync_dist=True)
        
        # Log relaxed metrics
        self.log("test_relaxed_f1", metrics["relaxed_f1"], sync_dist=True)
        self.log("test_relaxed_precision", metrics["relaxed_precision"], sync_dist=True)
        self.log("test_relaxed_recall", metrics["relaxed_recall"], sync_dist=True)
        self.log("test_relaxed_sensitivity", metrics["relaxed_sensitivity"], sync_dist=True)
        self.log("test_relaxed_specificity", metrics["relaxed_specificity"], sync_dist=True)
        self.log("test_relaxed_accuracy", metrics["relaxed_accuracy"], sync_dist=True)
        
        # Log detailed results
        confusion_matrix_str = f"""
        Confusion Matrix:
        ┌───────────────┬────────────────┬────────────────┐
        │               │ Predicted +ve  │ Predicted -ve  │
        ├───────────────┼────────────────┼────────────────┤
        │ Actual +ve    │ {metrics['true_positives']} (TP)    │ {metrics['false_negatives']} (FN)    │
        ├───────────────┼────────────────┼────────────────┤
        │ Actual -ve    │ {metrics['false_positives']} (FP)    │ {metrics['true_negatives']} (TN)    │
        └───────────────┴────────────────┴────────────────┘
        
        Relaxed Confusion Matrix:
        ┌───────────────┬────────────────┬────────────────┐
        │               │ Predicted +ve  │ Predicted -ve  │
        ├───────────────┼────────────────┼────────────────┤
        │ Actual +ve    │ {metrics['relaxed_tp']} (TP)    │ {metrics['relaxed_fn']} (FN)    │
        ├───────────────┼────────────────┼────────────────┤
        │ Actual -ve    │ {metrics['relaxed_fp']} (FP)    │ {metrics['relaxed_tn']} (TN)    │
        └───────────────┴────────────────┴────────────────┘
        """
        
        self.custom_logger.info(f"Test metrics:")
        self.custom_logger.info(f"Standard metrics: {metrics}")
        self.custom_logger.info(confusion_matrix_str)
        
        return metrics

    def configure_optimizers(self):
        """Configure optimizer for training.
        
        Returns:
            List of optimizers
        """
        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.args.lr,
            weight_decay=self.args.weight_decay,
        )

        return [optimizer]  # , [scheduler]


def seed_worker(worker_id):
    """Set seed for workers to ensure reproducibility.
    
    Args:
        worker_id: ID of the worker
    """
    # Set seed for Python and NumPy in each worker
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def prepare_custom_dataloader(args):
    """Prepare dataloaders for MEG dataset.
    
    Args:
        args: Command-line arguments containing dataset parameters
        
    Returns:
        Tuple of train, test, and validation data loaders
    """
    # Set random seed for reproducibility
    seed = args.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    logger = logging.getLogger(__name__)
    
    # Use the specified data root path
    root = args.data_dir

    train_files = os.listdir(os.path.join(root, "train"))
    np.random.shuffle(train_files)
    val_files = os.listdir(os.path.join(root, "val"))
    test_files = os.listdir(os.path.join(root, "test"))

    logger.info(f"Size of train, val, and test file list: {len(train_files)}, {len(val_files)}, {len(test_files)}")

    # Prepare training and test data loader
    train_loader = torch.utils.data.DataLoader(
        MEGDataset(os.path.join(root, "train"), train_files),
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=args.num_workers,
        persistent_workers=True,
        worker_init_fn=seed_worker
    )
    test_loader = torch.utils.data.DataLoader(
        MEGDataset(os.path.join(root, "test"), test_files),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        persistent_workers=True,
        worker_init_fn=seed_worker
    )
    val_loader = torch.utils.data.DataLoader(
        MEGDataset(os.path.join(root, "val"), val_files),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        persistent_workers=True,
        worker_init_fn=seed_worker
    )
    logger.info(f"Size of train, val, and test loaders: {len(train_loader)}, {len(val_loader)}, {len(test_loader)}")
    # test each loader to get a batch
    for _, (X, y) in enumerate(train_loader):
        logger.info(f"Train loader: {X.shape}, {y.shape}")
        break
    for _, (X, y) in enumerate(val_loader):
        logger.info(f"Val loader: {X.shape}, {y.shape}")
        break
    for _, (X, y) in enumerate(test_loader):
        logger.info(f"Test loader: {X.shape}, {y.shape}")
        break
    return train_loader, test_loader, val_loader


def prepare_TUAB_dataloader(args):
    """Prepare dataloaders for TUAB dataset.
    
    Args:
        args: Command-line arguments containing dataset parameters
        
    Returns:
        Tuple of train, test, and validation data loaders
    """
    # Set random seed for reproducibility
    seed = args.seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

    logger = logging.getLogger(__name__)
    
    # Use the specified data root path
    root = os.path.join(args.data_dir, "tuh_eeg_abnormal/v3.0.0/edf/processed")

    train_files = os.listdir(os.path.join(root, "train"))
    np.random.shuffle(train_files)
    val_files = os.listdir(os.path.join(root, "val"))
    test_files = os.listdir(os.path.join(root, "test"))

    logger.info(f"Size of train, val, and test file list: {len(train_files)}, {len(val_files)}, {len(test_files)}")

    # Prepare training and test data loader
    train_loader = torch.utils.data.DataLoader(
        TUABLoader(os.path.join(root, "train"),
                   train_files, args.sampling_rate),
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=args.num_workers,
        persistent_workers=True,
    )
    test_loader = torch.utils.data.DataLoader(
        TUABLoader(os.path.join(root, "test"), test_files, args.sampling_rate),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        persistent_workers=True,
    )
    val_loader = torch.utils.data.DataLoader(
        TUABLoader(os.path.join(root, "val"), val_files, args.sampling_rate),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        persistent_workers=True,
    )
    logger.info(f"Size of train, val, and test loaders: {len(train_loader)}, {len(val_loader)}, {len(test_loader)}")
    return train_loader, test_loader, val_loader


def prepare_CHB_MIT_dataloader(args):
    """Prepare dataloaders for CHB-MIT dataset.
    
    Args:
        args: Command-line arguments containing dataset parameters
        
    Returns:
        Tuple of train, test, and validation data loaders
    """
    # Set random seed for reproducibility
    seed = args.seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

    logger = logging.getLogger(__name__)
    
    # Use the specified data root path
    root = os.path.join(args.data_dir, "physionet.org/files/chbmit/1.0.0/clean_segments")

    train_files = os.listdir(os.path.join(root, "train"))
    val_files = os.listdir(os.path.join(root, "val"))
    test_files = os.listdir(os.path.join(root, "test"))

    logger.info(f"Size of train, val, and test file list: {len(train_files)}, {len(val_files)}, {len(test_files)}")

    # Prepare training and test data loader
    train_loader = torch.utils.data.DataLoader(
        CHBMITLoader(os.path.join(root, "train"),
                     train_files, args.sampling_rate),
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=args.num_workers,
        persistent_workers=True,
    )
    test_loader = torch.utils.data.DataLoader(
        CHBMITLoader(os.path.join(root, "test"),
                     test_files, args.sampling_rate),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        persistent_workers=True,
    )
    val_loader = torch.utils.data.DataLoader(
        CHBMITLoader(os.path.join(root, "val"), val_files, args.sampling_rate),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        persistent_workers=True,
    )
    logger.info(f"Size of train, val, and test loaders: {len(train_loader)}, {len(val_loader)}, {len(test_loader)}")
    return train_loader, test_loader, val_loader


def prepare_PTB_dataloader(args):
    """Prepare dataloaders for PTB dataset.
    
    Args:
        args: Command-line arguments containing dataset parameters
        
    Returns:
        Tuple of train, test, and validation data loaders
    """
    # Set random seed for reproducibility
    seed = args.seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

    logger = logging.getLogger(__name__)
    
    # Use the specified data root path
    root = os.path.join(args.data_dir, "WFDB/processed2")

    train_files = os.listdir(os.path.join(root, "train"))
    val_files = os.listdir(os.path.join(root, "val"))
    test_files = os.listdir(os.path.join(root, "test"))

    logger.info(f"Size of train, val, and test file list: {len(train_files)}, {len(val_files)}, {len(test_files)}")

    # Prepare training and test data loader
    train_loader = torch.utils.data.DataLoader(
        PTBLoader(os.path.join(root, "train"),
                  train_files, args.sampling_rate),
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=args.num_workers,
        persistent_workers=True,
    )
    test_loader = torch.utils.data.DataLoader(
        PTBLoader(os.path.join(root, "test"), test_files, args.sampling_rate),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        persistent_workers=True,
    )
    val_loader = torch.utils.data.DataLoader(
        PTBLoader(os.path.join(root, "val"), val_files, args.sampling_rate),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        persistent_workers=True,
    )
    logger.info(f"Size of train, val, and test loaders: {len(train_loader)}, {len(val_loader)}, {len(test_loader)}")
    return train_loader, test_loader, val_loader


def supervised(args):
    """Main function to run supervised training and evaluation.
    
    Args:
        args: Command-line arguments
    """
    logger = logging.getLogger(__name__)
    
    # Setup logging
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_file = os.path.join(args.log_dir, f"{args.dataset}-{args.model}-{timestamp}.log")
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logger.addHandler(file_handler)
    
    # Get data loaders
    logger.info(f"Preparing data loaders for dataset: {args.dataset}")
    if args.dataset == "MEG":
        train_loader, test_loader, val_loader = prepare_custom_dataloader(args)
    elif args.dataset == "TUAB":
        train_loader, test_loader, val_loader = prepare_TUAB_dataloader(args)
    elif args.dataset == "CHB-MIT":
        train_loader, test_loader, val_loader = prepare_CHB_MIT_dataloader(args)
    elif args.dataset == "PTB":
        train_loader, test_loader, val_loader = prepare_PTB_dataloader(args)
    else:
        raise NotImplementedError(f"Dataset {args.dataset} not implemented")

    # Define the model
    logger.info(f"Initializing model: {args.model}")
    if args.model == "SPaRCNet":
        model = SPaRCNet(
            in_channels=args.in_channels,
            sample_length=int(args.sampling_rate * args.sample_length),
            n_classes=args.n_classes,
            block_layers=4,
            growth_rate=16,
            bn_size=16,
            drop_rate=0.5,
            conv_bias=True,
            batch_norm=True,
        )

    elif args.model == "ContraWR":
        model = ContraWR(
            in_channels=args.in_channels,
            n_classes=args.n_classes,
            fft=args.token_size,
            steps=args.hop_length // 5,
        )

    elif args.model == "CNNTransformer":
        model = CNNTransformer(
            in_channels=args.in_channels,
            n_classes=args.n_classes,
            fft=args.sampling_rate,
            steps=args.hop_length // 5,
            dropout=0.2,
            nhead=4,
            emb_size=256,
        )

    elif args.model == "FFCL":
        model = FFCL(
            in_channels=args.in_channels,
            n_classes=args.n_classes,
            fft=args.token_size,
            steps=args.hop_length // 5,
            sample_length=int(args.sampling_rate * args.sample_length),
            shrink_steps=20,
        )

    elif args.model == "STTransformer":
        model = STTransformer(
            emb_size=256,
            depth=4,
            n_classes=args.n_classes,
            channel_legnth=int(
                args.sampling_rate * args.sample_length
            ),  # (sampling_rate * duration)
            n_channels=args.in_channels,
        )

    elif args.model == "BIOT":
        model = BIOTClassifier(
            n_classes=args.n_classes,
            n_channels=args.in_channels,
            n_fft=args.token_size,
            hop_length=args.hop_length,
            raw=args.raw, 
            patch_size=args.patch_size, 
            overlap=args.overlap,
        )

    else:
        raise NotImplementedError(f"Model {args.model} not implemented")
        
    lightning_model = LitModel_finetune.load_from_checkpoint(args.pretrain_model_path, args=args, model=model) if args.pretrain_model_path else LitModel_finetune(args, model)

    # Logger and callbacks
    version = f"{args.dataset}-{args.model}-{args.lr}-{args.batch_size}-{args.sampling_rate}-{args.token_size}-{args.hop_length}-{args.sample_length}-{args.epochs}-{timestamp}"
    tensorboard_logger = TensorBoardLogger(
        save_dir=args.model_log_dir,
        version=version,
        name="",
    )

    dirpath = tensorboard_logger.log_dir
    os.makedirs(dirpath, exist_ok=True)

    # Early stopping to monitor PR AUC
    early_stop_callback = EarlyStopping(
        monitor="val_pr_auc", patience=args.patience, verbose=True, mode="max"
    )

    # Save the best 3 models, and monitor PR AUC
    best_checkpoint_callback = ModelCheckpoint(
        dirpath=dirpath,
        filename="best-{epoch:02d}-{val_pr_auc:.4f}",
        save_top_k=3,
        monitor="val_pr_auc",
        mode="max",
        save_last=False,
        verbose=True
    )

    # Ensure the last model is being saved correctly
    last_checkpoint_callback = ModelCheckpoint(
        dirpath=dirpath,
        filename="last-{epoch:02d}",
        save_top_k=1,
        save_last=True,
        verbose=True
    )

    trainer = pl.Trainer(
        devices=[int(str_id) for str_id in args.gpus if str_id.isdigit()],
        accelerator="auto",
        strategy=DDPStrategy(find_unused_parameters=True),
        benchmark=True,
        enable_checkpointing=True,
        logger=tensorboard_logger,
        max_epochs=args.epochs,
        callbacks=[early_stop_callback, best_checkpoint_callback, last_checkpoint_callback],
    )

    # Train the model
    logger.info("Starting model training...")
    trainer.fit(
        lightning_model, train_dataloaders=train_loader, val_dataloaders=val_loader
    )
    logger.info("Training completed")

    # Test with pretrained model
    if args.pretrain_model_path:
        path = args.pretrain_model_path
        logger.info(f"Testing pretrained model: {path}")
        result = trainer.test(model=lightning_model, ckpt_path=path, dataloaders=test_loader)[0]
        logger.info(f"Pretrained model test results:")
        logger.info(f"{result}")

    # Test with the best models
    logger.info("Testing with best models...")
    if hasattr(best_checkpoint_callback, 'best_k_models') and best_checkpoint_callback.best_k_models:
        # Sort models by score (higher PR AUC is better)
        sorted_models = sorted(
            [(score, path) for path, score in best_checkpoint_callback.best_k_models.items()],
            reverse=True  # Higher score is better since we're using max mode
        )

        logger.info(f"Found {len(sorted_models)} best models")
        for i, (score, path) in enumerate(sorted_models):
            logger.info(f"Testing best model {i + 1}/{len(sorted_models)}: {path}")
            result = trainer.test(
                model=lightning_model, ckpt_path=path, dataloaders=test_loader
            )[0]
            logger.info(f"Best model {i + 1} test results (PR AUC: {score.item():.4f}):")
            logger.info(f"{result}")
    else:
        logger.info("No best model checkpoints found")

    # Test with the last model
    logger.info("Testing with last model...")
    last_model_path = os.path.join(dirpath, "last.ckpt")
    if os.path.exists(last_model_path):
        logger.info(f"Found last model at: {last_model_path}")
        last_result = trainer.test(
            model=lightning_model, ckpt_path=last_model_path, dataloaders=test_loader
        )[0]
        logger.info("Last model test results:")
        logger.info(f"{last_result}")
    else:
        # Fallback - look for the filename pattern if the default path doesn't exist
        last_checkpoints = [f for f in os.listdir(dirpath) if f.startswith("last-epoch=")]
        if last_checkpoints:
            last_model_path = os.path.join(dirpath, last_checkpoints[0])
            logger.info(f"Found last model at alternative path: {last_model_path}")
            last_result = trainer.test(
                model=lightning_model, ckpt_path=last_model_path, dataloaders=test_loader
            )[0]
            logger.info("Last model test results:")
            logger.info(f"{last_result}")
        else:
            logger.info("No last model checkpoint found")


def setup_logging(args):
    """Set up logging configuration.
    
    Args:
        args: Command-line arguments with logging parameters
    """
    # Create log directory if it doesn't exist
    os.makedirs(args.log_dir, exist_ok=True)
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(args.log_level)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(args.log_level)
    console_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    root_logger.addHandler(console_handler)
    
    # File handler for the main log file
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    main_log_file = os.path.join(args.log_dir, f"main-{timestamp}.log")
    file_handler = logging.FileHandler(main_log_file)
    file_handler.setLevel(args.log_level)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    root_logger.addHandler(file_handler)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and evaluate binary classification models")
    
    # General parameters
    parser.add_argument("--seed", type=int, default=12345, help="random seed for reproducibility")
    parser.add_argument("--log_dir", type=str, default="./logs", help="directory to save logs")
    parser.add_argument("--model_log_dir", type=str, default="./models", help="directory to save model checkpoints and related logs")
    parser.add_argument("--data_dir", type=str, default="./data/processed", help="base directory for datasets")
    parser.add_argument("--log_level", type=str, default="INFO", 
                        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
                        help="logging level")
    
    # Training parameters
    parser.add_argument("--epochs", type=int, default=100, help="number of epochs")
    parser.add_argument("--lr", type=float, default=1e-3, help="learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-5, help="weight decay")
    parser.add_argument("--batch_size", type=int, default=512, help="batch size")
    parser.add_argument("--num_workers", type=int, default=32, help="number of workers")
    parser.add_argument("--patience", type=int, default=5, help="patience for early stopping")
    parser.add_argument("--gpus", type=list, default=[0], help="GPU devices to use")
    
    # Dataset parameters
    parser.add_argument("--dataset", type=str, default="MEG", 
                        choices=["MEG", "TUAB", "CHB-MIT", "PTB"],
                        help="which dataset to use")
    
    # Model parameters
    parser.add_argument("--model", type=str, default="SPaRCNet", 
                       choices=["SPaRCNet", "ContraWR", "CNNTransformer", "FFCL", "STTransformer", "BIOT"],
                       help="which supervised model to use")
    parser.add_argument("--in_channels", type=int, default=16, help="number of input channels")
    parser.add_argument("--sample_length", type=float, default=10, help="length (s) of sample")
    parser.add_argument("--n_classes", type=int, default=1, help="number of output classes")
    parser.add_argument("--sampling_rate", type=int, default=200, help="sampling rate (r)")
    parser.add_argument("--token_size", type=int, default=200, help="token size (t)")
    parser.add_argument("--hop_length", type=int, default=100, help="token hop length (t - p)")
    parser.add_argument("--raw", type=bool, default=False, help="Whether to use raw data/time series or stft")
    parser.add_argument("--patch_size", type=int, default=100, help="Size of a patch/ time segment in case raw data is used")
    parser.add_argument("--overlap", type=float, default=0.0, help="overlap percentage for patches in case raw data is used")
    parser.add_argument("--pretrain_model_path", type=str, default="", help="pretrained model path")
    
    arguments = parser.parse_args()
    
    # Set up logging first
    setup_logging(arguments)
    
    # Get logger for this module
    logger = logging.getLogger(__name__)
    logger.info(f"Arguments: {arguments}")
    
    # Run the supervised training process
    supervised(arguments)