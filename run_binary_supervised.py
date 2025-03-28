import random
import os
import argparse
import logging
from datetime import datetime
from tqdm import tqdm

# # we want to debug so we set
# For debugging consider passing CUDA_LAUNCH_BLOCKING=1
# Compile with `TORCH_USE_CUDA_DSA` to enable device-side assertions.
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["TORCH_USE_CUDA_DSA"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import lightning as pl
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.strategies import DDPStrategy
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from pyhealth.metrics import binary_metrics_fn

from model import BIOTHierarchicalEncoder
from utils import MEGDataset, focal_loss, focal_loss_with_class_weights, weighted_BCE

import torch
import numpy as np
from sklearn.metrics import precision_score, recall_score, confusion_matrix#, f1_score, accuracy_score


class ProjectionLayerCallback(pl.Callback):
    """Callback to handle channel projection layer freezing.
    
    This callback implements a strategy to freeze the projection layer
    after a certain number of epochs or when validation metrics reach a plateau.
    """
    
    def __init__(self, freeze_after_epochs=10, freeze_on_plateau=True, patience=3):
        """Initialize the projection layer callback.
        
        Args:
            freeze_after_epochs: Number of epochs after which to freeze the projection
            freeze_on_plateau: Whether to freeze on validation metric plateau
            patience: Number of epochs to wait for improvement before freezing on plateau
        """
        super().__init__()
        self.freeze_after_epochs = freeze_after_epochs
        self.freeze_on_plateau = freeze_on_plateau
        self.patience = patience
        self.best_val_metric = 0
        self.epochs_without_improvement = 0
        self.frozen = False
        self.logger = logging.getLogger(__name__ + ".ProjectionLayerCallback")
    
    def on_train_epoch_end(self, trainer, pl_module):
        """Check whether to freeze the projection layer at the end of each epoch."""
        current_epoch = trainer.current_epoch
        
        # Check if we should freeze based on epoch count
        if not self.frozen and current_epoch >= self.freeze_after_epochs:
            self._freeze_projection(pl_module)
            return
            
        # Check if we should freeze based on validation plateau
        if self.freeze_on_plateau and not self.frozen:
            current_val_metric = trainer.callback_metrics.get('val_pr_auc', 0)
            
            if current_val_metric > self.best_val_metric:
                self.best_val_metric = current_val_metric
                self.epochs_without_improvement = 0
            else:
                self.epochs_without_improvement += 1
                
            if self.epochs_without_improvement >= self.patience:
                self.logger.info(f"Validation metric plateaued for {self.patience} epochs.")
                self._freeze_projection(pl_module)
    
    def _freeze_projection(self, pl_module):
        """Freeze the projection layer in the model."""
        if hasattr(pl_module.model, 'freeze_projection'):
            pl_module.model.freeze_projection()
            self.frozen = True
            self.logger.info("Projection layer frozen after training phase.")
        else:
            self.logger.warning("Model does not have freeze_projection method.")


class LitModel_finetune(pl.LightningModule):
    def __init__(self, args: argparse.Namespace, model, class_weights=None):
        super().__init__()
        self.model = model
        self.threshold = 0.5
        self.args = args
        self.class_weights = class_weights
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
        logits = self.model(X)

        # Option 1: Original BCE loss (no weighting)
        # loss = BCE(prob, y)
        # Option 2: Weighted BCE with class-specific weights
        #loss = weighted_BCE(logits, y, self.class_weights)
        # Option 3: Original focal loss
        # loss = focal_loss(prob, y, gamma=2.0, alpha=0.25)
        # Option 4: Focal loss with class weights
        loss = focal_loss_with_class_weights(logits, y, self.class_weights, gamma=2.0)
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
            logits = self.model(X)
            step_probs = torch.sigmoid(logits).cpu().numpy()
            step_gt = y.cpu().numpy()
        # Append the result to the list
        self.validation_step_outputs.append((step_probs, step_gt))
        return step_probs, step_gt

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
            result = np.append(result, out[0].flatten())
            gt = np.append(gt, out[1].flatten())

        num_positives = np.sum(gt)
        num_negatives = len(gt) - num_positives
        self.custom_logger.debug(f"Number of positives: {num_positives}, Number of negatives: {num_negatives}")
        if num_positives * num_negatives != 0:  # to prevent all 0 or all 1 cases
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
            result_metrics = binary_metrics_fn(gt, result,
                metrics=["pr_auc", "roc_auc", "accuracy", "balanced_accuracy", "f1"],
                threshold=self.threshold,
            )

            # Additional insight: log the performance at a standard 0.5 threshold for comparison
            standard_metrics = binary_metrics_fn(gt, result,
                metrics=["accuracy", "balanced_accuracy", "f1"],
                threshold=0.5,
            )
            self.log("val_f1_standard", standard_metrics["f1"], sync_dist=True)
            self.log("val_acc_standard", standard_metrics["accuracy"], sync_dist=True)
        else:
            self.custom_logger.info(f"Could not compute metrics as the ground truth is all 0 or all 1")
            result_metrics = {
                "accuracy": 0.0, "balanced_accuracy": 0.0, "pr_auc": 0.0,
                "roc_auc": 0.0, "f1": 0.0,
            }
            self.threshold = 0.5
        
        # Log metrics
        for key, value in result_metrics.items():
            self.log(f"val_{key}", value, sync_dist=True)
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
        """Calculate test metrics including relaxed scores for spike detection."""
        test_step_outputs = self.test_step_outputs
        result = np.array([])
        gt = np.array([])
        
        for out in test_step_outputs:
            result = np.append(result, out[0].flatten())
            gt = np.append(gt, out[1].flatten())
        
        # Convert continuous predictions to binary using the best threshold
        binary_pred = (result >= self.threshold).astype(int)
        num_positives = np.sum(gt)
        num_negatives = len(gt) - num_positives
        self.custom_logger.debug(f"Number of positives: {num_positives}, Number of negatives: {num_negatives}")
        if num_positives * num_negatives != 0:  # to prevent all 0 or all 1 cases
            # Standard metrics
            metrics = binary_metrics_fn(gt, result,
                metrics=["pr_auc", "roc_auc", "accuracy", "balanced_accuracy", "f1"],
                threshold=self.threshold,
            )
            
            # Additional metrics
            metrics["precision"] = precision_score(gt, binary_pred)
            metrics["recall"] = recall_score(gt, binary_pred)
            
            # Confusion matrix
            tn, fp, fn, tp = confusion_matrix(gt, binary_pred, labels=[0, 1]).ravel()
            for key, value in zip(["true_positives", "false_positives", "true_negatives", "false_negatives"], [tp, fp, tn, fn]):
                metrics[key] = int(value)
        else:
            self.custom_logger.info(f"Could not compute metrics as the ground truth is all 0 or all 1")
            # Default metrics for one-class case
            metrics = {
                "accuracy": 0.0, "balanced_accuracy": 0.0, "pr_auc": 0.0, "roc_auc": 0.0, 
                "f1": 0.0, "precision": 0.0, "recall": 0.0,
                "true_positives": 0, "false_positives": 0, "true_negatives": 0, "false_negatives": 0,
            }
        
        # Log all metrics
        for key, value in metrics.items():
            self.log(f"test_{key}", value, sync_dist=True)
        
        # Log detailed results
        confusion_matrix_str = f"""
        Confusion Matrix:
        ┌───────────────┬────────────────┬────────────────┐
        │               │ Predicted +ve  │ Predicted -ve  │
        ├───────────────┼────────────────┼────────────────┤
        │ Actual +ve    │ {metrics['true_positives']} (TP)        │ {metrics['false_negatives']} (FN)        │
        ├───────────────┼────────────────┼────────────────┤
        │ Actual -ve    │ {metrics['false_positives']} (FP)        │ {metrics['true_negatives']} (TN)        │
        └───────────────┴────────────────┴────────────────┘
        """
        
        self.custom_logger.info(f"Test metrics: {metrics}")
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

        # Learning rate scheduler
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", factor=0.5, patience=3
        )

        return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "val_pr_auc"}


def seed_worker(worker_id):
    """Set seed for workers to ensure reproducibility.
    
    Args:
        worker_id: ID of the worker
    """
    # Set seed for Python and NumPy in each worker
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def prepare_meg_dataloader(args):
    """Prepare dataloaders for MEG dataset
    
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
    
    # Load datasets
    train_set = MEGDataset.from_processed_dir(root, "train", use_weights=True, cache_size=100, metadata_only=False)
    val_set = MEGDataset.from_processed_dir(root, "val", use_weights=True, cache_size=100, metadata_only=False)
    test_set = MEGDataset.from_processed_dir(root, "test", use_weights=True, cache_size=100, metadata_only=False)
    
    logger.info(f"Size of train, val, and test sets: {len(train_set)}, {len(val_set)}, {len(test_set)}")
    logger.info(f"Size of a sample from train, val, and test sets: {train_set[0][0].size()}, {val_set[0][0].size()}, {test_set[0][0].size()}")
    
    # Create data loaders
    train_loader = torch.utils.data.DataLoader(
        train_set,
        shuffle=True,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        persistent_workers=True,
        worker_init_fn=seed_worker
    )
    
    # For validation and test, we don't need shuffling
    val_loader = torch.utils.data.DataLoader(
        val_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        persistent_workers=True,
        worker_init_fn=seed_worker
    )
    
    test_loader = torch.utils.data.DataLoader(
        test_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        persistent_workers=True,
        worker_init_fn=seed_worker
    )
    
    logger.info(f"Size of train, val, and test loaders: {len(train_loader)}, {len(val_loader)}, {len(test_loader)}")
    
    # Test each loader to get a batch
    for _, (X, y) in enumerate(train_loader):
        logger.info(f"Train loader: {X.shape}, {y.shape}")
        break
    for _, (X, y) in enumerate(val_loader):
        logger.info(f"Val loader: {X.shape}, {y.shape}")
        break
    for _, (X, y) in enumerate(test_loader):
        logger.info(f"Test loader: {X.shape}, {y.shape}")
        break
    
    return train_loader, test_loader, val_loader, X.shape, y.shape, train_set.class_weights


def supervised(args):
    """Main function to run supervised training and evaluation.
    
    Args:
        args: Command-line arguments
    """
    logger = logging.getLogger(__name__)
    
    #  Setup logging
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_file = os.path.join(args.log_dir, f"{args.dataset}-{args.model}-projection-{timestamp}.log")
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logger.addHandler(file_handler)
    
    # Get data loaders with stratified batching if enabled
    logger.info(f"Preparing data loaders for dataset: {args.dataset}")
    train_loader, test_loader, val_loader, input_shape, output_shape, class_weights = prepare_meg_dataloader(args)
    
    # Log information about data shapes
    batch_size, n_segments, n_channels, n_samples_per_segment = input_shape
    batch_size, n_segments = output_shape
    logger.info(f"Input shape: Batch size: {batch_size}, Segments: {n_segments}, Channels: {n_channels}, Samples per segment: {n_samples_per_segment}")
    logger.info(f"Output shape: Batch size: {batch_size}, Segments: {n_segments}")
    logger.info(f"Will project {n_channels} original channels to {args.n_projected_channels} channels using '{args.projection_strategy}' strategy")

    # Define the model with channel projection
    logger.info(f"Initializing model: {args.model}")
    if args.model == "BIOT":
        model = BIOTHierarchicalEncoder(
            n_classes=1,
            n_channels=n_channels,
            n_projected_channels=args.n_projected_channels,
            n_segments=n_segments,
            token_size=args.token_size,
            overlap=args.overlap,
            raw=args.raw,
            emb_size=256,
            heads=8,
            segment_encoder_depth=4,
            inter_segment_depth=4,
            projection_strategy=args.projection_strategy,
            freeze_projection=args.freeze_projection,
            log_dir=args.log_dir
        )
    else:
        raise NotImplementedError(f"Model {args.model} not implemented")
        
    # Create lightning model with projection samples
    lightning_model = LitModel_finetune(
        args=args, 
        model=model, 
        class_weights=class_weights,
    )

    # Logger and callbacks
    projection_info = f"proj{args.n_projected_channels}-{args.projection_strategy}"
    version = f"{args.dataset}-{args.model}-{projection_info}-{timestamp}"
    
    tensorboard_logger = TensorBoardLogger(
        save_dir=args.model_log_dir,
        version=version,
        name="",
    )

    dirpath = tensorboard_logger.log_dir
    os.makedirs(dirpath, exist_ok=True)

    # Callbacks
    callbacks = [
        # Early stopping to monitor PR AUC
        EarlyStopping(
            monitor="val_pr_auc", 
            patience=args.patience, 
            verbose=True, 
            mode="max"
        ),
        
        # Save the best 3 models
        ModelCheckpoint(
            dirpath=dirpath,
            filename="best-{epoch:02d}-{val_pr_auc:.4f}",
            save_top_k=3,
            monitor="val_pr_auc",
            mode="max",
            save_last=False,
            verbose=True
        ),
        
        # Save the last model
        ModelCheckpoint(
            dirpath=dirpath,
            filename="last-{epoch:02d}",
            save_top_k=1,
            save_last=True,
            verbose=True
        )
    ]
    
    # Add projection layer callback if using channel projection
    if hasattr(model, 'channel_projection') and args.n_projected_channels < n_channels:
        projection_callback = ProjectionLayerCallback(
            freeze_after_epochs=args.freeze_projection_after,
            freeze_on_plateau=args.freeze_projection_on_plateau,
            patience=3
        )
        callbacks.append(projection_callback)
        logger.info(f"Added projection layer callback: freeze after {args.freeze_projection_after} epochs or on plateau: {args.freeze_projection_on_plateau}")

    trainer = pl.Trainer(
        devices=[int(str_id) for str_id in args.gpus if str_id.isdigit()],
        accelerator="auto",
        strategy=DDPStrategy(find_unused_parameters=True),
        benchmark=True,
        enable_checkpointing=True,
        logger=tensorboard_logger,
        max_epochs=args.epochs,
        callbacks=callbacks,
        log_every_n_steps=1,
    )

    # Train the model
    logger.info("Starting model training...")
    trainer.fit(
        lightning_model, train_dataloaders=train_loader, val_dataloaders=val_loader
    )
    logger.info("Training completed")

    # Test best models
    logger.info("Testing with best models...")
    best_checkpoint_callback = [cb for cb in callbacks if isinstance(cb, ModelCheckpoint) and cb.monitor == "val_pr_auc"][0]
    
    if hasattr(best_checkpoint_callback, 'best_k_models') and best_checkpoint_callback.best_k_models:
        # Sort models by score
        sorted_models = sorted(
            [(score, path) for path, score in best_checkpoint_callback.best_k_models.items()],
            reverse=True
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
        # Fallback - look for the filename pattern
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
    parser.add_argument("--weight_decay", type=float, default=3e-5, help="weight decay")
    parser.add_argument("--batch_size", type=int, default=512, help="batch size")
    parser.add_argument("--num_workers", type=int, default=32, help="number of workers")
    parser.add_argument("--patience", type=int, default=5, help="patience for early stopping")
    parser.add_argument("--gpus", type=list, default=[0], help="GPU devices to use")
    
    # Dataset parameters
    parser.add_argument("--dataset", type=str, default="MEG", choices=["MEG",],
                        help="which dataset to use")
    
    # Model parameters
    parser.add_argument("--model", type=str, default="BIOT", choices=["BIOT",],
                        help="which supervised model to use")
    parser.add_argument("--sampling_rate", type=int, default=200, help="sampling rate (r)")
    parser.add_argument("--token_size", type=int, default=200, help="token size (t) in samples")
    parser.add_argument("--overlap", type=float, default=0.0, help="overlap percentage of tokens")
    parser.add_argument("--raw", type=bool, default=False, help="Whether to use raw data/time series or stft")
    parser.add_argument("--pretrain_model_path", type=str, default="", help="pretrained model path")

    # Channel projection parameters
    parser.add_argument("--n_projected_channels", type=int, default=64, 
                        help="Number of channels after projection (default: 64)")
    parser.add_argument("--projection_strategy", type=str, default="learned", 
                        choices=["learned"],
                        help="Strategy for channel projection (default: learned)")
    parser.add_argument("--freeze_projection", action="store_true", 
                        help="Whether to freeze projection layer initially")
    parser.add_argument("--freeze_projection_after", type=int, default=10, 
                        help="Freeze projection layer after this many epochs (default: 10)")
    parser.add_argument("--freeze_projection_on_plateau", action="store_true", 
                        help="Freeze projection when validation metric plateaus")
    
    arguments = parser.parse_args()
    
    # Set up logging first
    setup_logging(arguments)
    
    # Get logger for this module
    logger = logging.getLogger(__name__)
    logger.info(f"Arguments: {arguments}")
    
    # Run the supervised training process
    supervised(arguments)