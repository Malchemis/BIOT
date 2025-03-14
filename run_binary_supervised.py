import random
import os
import argparse
import logging
from pathlib import Path
from datetime import datetime

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
from utils import MEGDataset, TUABLoader, CHBMITLoader, PTBLoader, focal_loss, BCE
torch.set_float32_matmul_precision('high') # Take advantage of tensor cores (trade off between speed and precision)


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
        #loss = focal_loss(prob, y)
        loss = BCE(prob, y)
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
        """Process validation outputs at the end of validation epoch."""
        val_step_outputs = self.validation_step_outputs
        result = np.array([])
        gt = np.array([])
        for out in val_step_outputs:
            result = np.append(result, out[0])
            gt = np.append(gt, out[1])

        if sum(gt) * (len(gt) - sum(gt)) != 0:  # to prevent all 0 or all 1 and raise the AUROC error
            self.threshold = np.sort(result)[-int(np.sum(gt))]
            result = binary_metrics_fn(
                gt,
                result,
                metrics=["pr_auc", "roc_auc", "accuracy", "balanced_accuracy", "f1"],
                threshold=self.threshold,
            )
        else:
            result = {
                "accuracy": 0.0,
                "balanced_accuracy": 0.0,
                "pr_auc": 0.0,
                "roc_auc": 0.0,
                "f1": 0.0,
            }
        self.log("val_acc", result["accuracy"], sync_dist=True)
        self.log("val_bacc", result["balanced_accuracy"], sync_dist=True)
        self.log("val_pr_auc", result["pr_auc"], sync_dist=True)
        self.log("val_auroc", result["roc_auc"], sync_dist=True)
        self.log("val_f1", result["f1"], sync_dist=True)
        self.log("best_threshold", self.threshold, sync_dist=True)
        self.custom_logger.info(f"Validation metrics: {result}")

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
        """Process test outputs at the end of test epoch."""
        test_step_outputs = self.test_step_outputs
        result = np.array([])
        gt = np.array([])
        for out in test_step_outputs:
            result = np.append(result, out[0])
            gt = np.append(gt, out[1])
        if sum(gt) * (len(gt) - sum(gt)) != 0:  # to prevent all 0 or all 1 and raise the AUROC error
            result = binary_metrics_fn(
                gt,
                result,
                metrics=["pr_auc", "roc_auc", "accuracy", "balanced_accuracy", "f1"],
                threshold=self.threshold,
            )
        else:
            result = {
                "accuracy": 0.0,
                "balanced_accuracy": 0.0,
                "pr_auc": 0.0,
                "roc_auc": 0.0,
                "f1": 0.0,
            }
        self.log("test_acc", result["accuracy"], sync_dist=True)
        self.log("test_bacc", result["balanced_accuracy"], sync_dist=True)
        self.log("test_pr_auc", result["pr_auc"], sync_dist=True)
        self.log("test_auroc", result["roc_auc"], sync_dist=True)
        self.log("test_f1", result["f1"], sync_dist=True)
        self.custom_logger.info(f"Test metrics: {result}")
        
        return result

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
    root = os.path.join(args.data_dir, "crnl-meg")

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
        if args.pretrain_model_path and (args.sampling_rate == 200):
            model.biot.load_state_dict(torch.load(args.pretrain_model_path))
            logger.info(f"Loaded pretrained model from {args.pretrain_model_path}")

    else:
        raise NotImplementedError(f"Model {args.model} not implemented")
        
    lightning_model = LitModel_finetune(args, model)

    # Logger and callbacks
    version = f"{args.dataset}-{args.model}-{args.lr}-{args.batch_size}-{args.sampling_rate}-{args.token_size}-{args.hop_length}"
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
        devices=args.gpus,
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