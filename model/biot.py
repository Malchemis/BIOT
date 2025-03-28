import logging
import math
import os
from typing import Optional, Tuple

import torch
import torch.nn as nn
from torch.nn import Identity
import numpy as np
from linear_attention_transformer import LinearAttentionTransformer


class PatchFrequencyEmbedding(nn.Module):
    """Module for embedding frequency domain data.
    
    This module projects frequency-domain data to a fixed embedding size.
    
    Attributes:
        projection (nn.Linear): Linear projection layer from frequency to embedding space.
    """
    
    def __init__(self, emb_size: int = 256, n_freq: int = 101, log_dir: Optional[str] = None):
        """Initialize the patch frequency embedding.
        
        Args:
            emb_size: Size of the embedding vector.
            n_freq: Number of frequency components.
            log_dir: Optional directory for log files. If None, logs to console only.
        """
        super().__init__()
        self.projection = nn.Linear(n_freq, emb_size)
        self.logger = logging.getLogger(__name__ + ".PatchFrequencyEmbedding")
        
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
            file_handler = logging.FileHandler(os.path.join(log_dir, "patch_freq_embedding.log"))
            file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
            self.logger.addHandler(file_handler)
            
        self.logger.debug(f"Initialized with emb_size={emb_size}, n_freq={n_freq}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the module.
        
        Args:
            x: Input tensor of shape (batch, freq, time).
            
        Returns:
            Embedded tensor of shape (batch, time, emb_size).
        """
        # Permute to (batch, time, freq) for linear projection along freq dimension
        x = x.permute(0, 2, 1)
        x = self.projection(x)
        return x


class PatchTimeEmbedding(nn.Module):
    """Module for embedding time domain data.
    
    This module processes raw time-series data by splitting it into possibly overlapping
    patches and projecting each patch to a fixed embedding size.
    
    Attributes:
        patch_size (int): Size of each time patch.
        overlap (float): Amount of overlap between adjacent patches (0.0-1.0).
        projection (nn.Linear): Linear projection layer from patch to embedding space.
    """
    
    def __init__(self, emb_size: int = 256, patch_size: int = 100, overlap: float = 0.0, 
                 log_dir: Optional[str] = None):
        """Initialize the patch time embedding.
        
        Args:
            emb_size: Size of the embedding vector.
            patch_size: Size of each time patch.
            overlap: Amount of overlap between adjacent patches (0.0-1.0).
            log_dir: Optional directory for log files. If None, logs to console only.
        """
        super().__init__()
        self.patch_size = patch_size
        self.overlap = overlap
        self.projection = nn.Linear(patch_size, emb_size)
        self.logger = logging.getLogger(__name__ + ".PatchTimeEmbedding")
        
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
            file_handler = logging.FileHandler(os.path.join(log_dir, "patch_time_embedding.log"))
            file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
            self.logger.addHandler(file_handler)
            
        self.logger.debug(f"Initialized with emb_size={emb_size}, patch_size={patch_size}, overlap={overlap}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the module.
        
        Args:
            x: Input tensor of shape (batch, channel, time).
            
        Returns:
            Embedded tensor of shape (batch, n_patches, emb_size).
        """
        time_steps = x.shape[2]

        # Calculate stride based on overlap
        stride = int(self.patch_size * (1 - self.overlap)) # e.g., 20 samples * (1 - 0.75) = 5 samples
        stride = max(1, stride)  # Ensure stride is at least 1 sample

        # Ensure we have enough time steps/samples for at least one patch
        if time_steps < self.patch_size:
            error_msg = f"Input length ({time_steps}) must be >= patch_size ({self.patch_size})"
            self.logger.error(error_msg)
            raise ValueError(error_msg)

        # Use unfold with the calculated stride to create overlapping patches
        x = x.squeeze(1)  # Remove channel dim: (batch, time)
        x = x.unfold(1, self.patch_size, stride)  # (batch, n_patches, patch_size)

        # Project to embedding dimension
        x = self.projection(x)  # (batch, n_patches, emb_size)
        return x


class ClassificationHead(nn.Sequential):
    """Module for classification head.
    
    This module takes the embeddings and produces class probabilities.
    
    Attributes:
        clshead (nn.Sequential): Sequential module with the classification layers.
    """
    
    def __init__(self, emb_size: int, n_classes: int, log_dir: Optional[str] = None):
        """Initialize the classification head.
        
        Args:
            emb_size: Size of the input embedding.
            n_classes: Number of output classes.
            log_dir: Optional directory for log files. If None, logs to console only.
        """
        super().__init__()
        self.clshead = nn.Sequential(
            nn.ELU(),
            nn.Linear(emb_size, n_classes),
        )
        self.logger = logging.getLogger(__name__ + ".ClassificationHead")
        self.logger.debug(f"Initialized with emb_size={emb_size}, n_classes={n_classes}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the module.
        
        Args:
            x: Input embedding tensor.
            
        Returns:
            Class logits.
        """
        return self.clshead(x)


class PositionalEncoding(nn.Module):
    """Module for adding positional encoding to embeddings.
    
    This implementation uses sinusoidal position encoding to give the model
    information about the position of tokens in the sequence.
    
    Attributes:
        dropout (nn.Dropout): Dropout layer.
        pe (torch.Tensor): Precomputed positional encoding. (registered as to not be a model parameter)
    """
    
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 100, log_dir: Optional[str] = None):
        """Initialize the positional encoding.
        
        Args:
            d_model: Dimensionality of the model.
            dropout: Dropout rate.
            max_len: sequence length.
            log_dir: Optional directory for log files. If None, logs to console only.
        """
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.logger = logging.getLogger(__name__ + ".PositionalEncoding")

        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
            file_handler = logging.FileHandler(os.path.join(log_dir, "positional_encoding.log"))
            file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
            self.logger.addHandler(file_handler)

        self.logger.debug(f"Initialized with d_model={d_model}, dropout={dropout}, max_len={max_len}")

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding to the input embeddings.
        
        Args:
            x: Input embeddings tensor of shape (batch, max_len, d_model).
            
        Returns:
            Embeddings with positional encoding added.
        """
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)
    

def stft(sample: torch.Tensor, n_fft, hop_length) -> torch.Tensor:
        """Compute the Short-Time Fourier Transform.
        
        Args:
            sample: Input tensor of shape (batch_size, 1, ts).
            
        Returns:
            Magnitude of the STFT of shape (batch_size, freq, time).
        """
        spectral = torch.stft(
            input=sample.squeeze(1),  # from shape (batch_size, 1, ts) to (batch_size, ts)
            n_fft=n_fft,
            hop_length=hop_length,
            window=torch.ones(n_fft, device=sample.device),
            center=False,
            onesided=True,
            return_complex=True,
        )
        return torch.abs(spectral)


class BIOTEncoder(nn.Module):
    """Biomedical Input-Output Transformer (BIOT) Encoder.
    
    This encoder processes biomedical signals using either spectral or raw time-domain
    approaches and generates embeddings.
    
    Attributes:
        n_fft (int): Number of FFT points.
        hop_length (int): Hop length for STFT.
        raw (bool): Whether to use raw time-domain processing.
        patch_frequency_embedding (PatchFrequencyEmbedding): Embedding module for spectral data.
        patch_time_embedding (PatchTimeEmbedding): Embedding module for raw time data.
        transformer (LinearAttentionTransformer): Transformer model.
        positional_encoding (PositionalEncoding): Positional encoding module.
        channel_tokens (nn.Embedding): Embedding for channel tokens.
        index (nn.Parameter): Parameter for channel indices.
    """
    
    def __init__(
            self,
            emb_size: int = 256,
            heads: int = 8,
            depth: int = 4,
            n_channels: int = 16,
            token_size: int = 200,
            overlap: float = 0.0,
            raw: bool = False,       
            log_dir: Optional[str] = None,
            **kwargs
    ):
        """Initialize the BIOT encoder.
        
        Args:
            emb_size: Size of the embedding vectors.
            heads: Number of attention heads.
            depth: Number of transformer layers.
            n_channels: Number of input channels.
            token_size: Number of FFT points for Spectral mode / Number of samples for Raw mode.
            raw: Whether to use raw time-domain processing.
            overlap: Overlap between patches for raw data mode.
            log_dir: Optional directory for log files. If None, logs to console only.
            **kwargs: Additional keyword arguments.
        """
        super().__init__()
        self.logger = logging.getLogger(__name__ + ".BIOTEncoder")
        
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
            file_handler = logging.FileHandler(os.path.join(log_dir, "biot_encoder.log"))
            file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
            self.logger.addHandler(file_handler)

        self.n_fft = token_size
        self.hop_length = int(token_size * overlap)
        self.raw = raw  # Store the raw mode flag
        
        self.logger.info(f"Initialized with emb_size={emb_size}, heads={heads}, depth={depth}, "
                         f"n_channels={n_channels}, token_size={token_size}, overlap={overlap}")
        self.logger.info(f"Processing mode: {'Raw time-domain' if raw else 'Spectral'}")

        # Create embedding modules for both spectral and raw data processing
        self.patch_frequency_embedding = PatchFrequencyEmbedding(
            emb_size=emb_size, n_freq=self.n_fft // 2 + 1, log_dir=log_dir
        )

        # Add the patch time embedding for raw data
        self.patch_time_embedding = PatchTimeEmbedding(
            emb_size=emb_size, patch_size=token_size, overlap=overlap, log_dir=log_dir
        )

        self.transformer = LinearAttentionTransformer(
            dim=emb_size,
            heads=heads,
            depth=depth,
            max_seq_len=1024,
            attn_layer_dropout=0.2,  # dropout right after self-attention layer
            attn_dropout=0.2,  # dropout post-attention
        )
        self.positional_encoding = PositionalEncoding(emb_size, log_dir=log_dir)

        # Channel token, N_channels >= your actual channels
        self.channel_tokens = nn.Embedding(n_channels, emb_size)
        self.index = nn.Parameter(
            torch.LongTensor(range(n_channels)), requires_grad=False
        )

    def forward(self, x: torch.Tensor, n_channel_offset: int = 0, perturb: bool = False) -> torch.Tensor:
        """Forward pass of the BIOT encoder.
        
        Args:
            x: Input tensor of shape (batch_size, channel, ts).
            n_channel_offset: Offset for channel tokens.
            perturb: Whether to randomly perturb the sequence.
            
        Returns:
            Encoded tensor of shape (batch_size, emb_size).
        """
        emb_seq = []
        # For each channel
        for i in range(x.shape[1]):
            # Get the current channel data
            channel_data = x[:, i:i + 1, :]

            # Segmentation (shape goes from (batch, 1, ts) to (batch, ts, emb_size))
            if self.raw:
                # Raw time-series data processing
                channel_emb = self.patch_time_embedding(channel_data)
            else:
                # Spectral data processing
                channel_spec = stft(channel_data, n_fft=self.n_fft, hop_length=self.hop_length) # shape: batch, freq, ts
                channel_emb = self.patch_frequency_embedding(channel_spec)

            batch_size, ts, _ = channel_emb.shape

            # Channel token embedding: We get the channel embedding, make it match the segment shape, and apply.
            channel_token_emb = (
                self.channel_tokens(
                    self.index[i + n_channel_offset]
                )
                .unsqueeze(0) # shape from (emb) to (1, emb)
                .unsqueeze(0) # (1, 1, emb)
                .repeat(batch_size, ts, 1) # repeat the channel embedding for all corresponding segments
            )

            # Positional encoding: We add segment and channel emb, then we apply relative positional emb.
            channel_emb = self.positional_encoding(channel_emb + channel_token_emb)

            # perturb (create random mask)
            if perturb:
                ts = channel_emb.shape[1]
                ts_new = np.random.randint(ts // 2, ts)
                selected_ts = np.random.choice(range(ts), ts_new, replace=False)
                channel_emb = channel_emb[:, selected_ts]
            emb_seq.append(channel_emb)

        # (batch_size, channels*ts, emb)
        emb = torch.cat(emb_seq, dim=1)
        # (batch_size, emb)
        emb = self.transformer(emb).mean(dim=1) # collapse the time dimension 
        return emb

        
class BIOTClassifier(nn.Module):
    """Biomedical Input-Output Transformer (BIOT) Classifier.
    
    This model uses the BIOT encoder for feature extraction and adds a classification head.
    
    Attributes:
        biot (BIOTEncoder): BIOT encoder for feature extraction.
        classifier (ClassificationHead): Classification head.
    """
    
    def __init__(self, emb_size: int = 256, heads: int = 8, depth: int = 4, n_classes: int = 6, 
                 raw: bool = False, overlap: float = 0.0, log_dir: Optional[str] = None, **kwargs):
        """Initialize the BIOT classifier.
        
        Args:
            emb_size: Size of the embedding vectors.
            heads: Number of attention heads.
            depth: Number of transformer layers.
            n_classes: Number of output classes.
            raw: Whether to use raw time-domain processing.
            overlap: Overlap between patches for raw data mode.
            log_dir: Optional directory for log files. If None, logs to console only.
            **kwargs: Additional parameters passed to BIOTEncoder.
        """
        super().__init__()
        self.logger = logging.getLogger(__name__ + ".BIOTClassifier")
        
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
            file_handler = logging.FileHandler(os.path.join(log_dir, "biot_classifier.log"))
            file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
            self.logger.addHandler(file_handler)
            
        self.logger.info(f"Initializing BIOT classifier with emb_size={emb_size}, heads={heads}, "
                         f"depth={depth}, n_classes={n_classes}, raw={raw}")
        
        self.biot = BIOTEncoder(emb_size=emb_size, heads=heads, depth=depth, raw=raw, 
                               overlap=overlap, log_dir=log_dir, **kwargs)
        self.classifier = ClassificationHead(emb_size, n_classes, log_dir=log_dir)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the BIOT classifier.
        
        Args:
            x: Input tensor of shape (batch_size, channel, ts).
            
        Returns:
            Logits for each class.
        """
        x = self.biot(x)
        x = self.classifier(x)
        return x


class ChannelProjection(nn.Module):
    """Channel projection module to reduce dimensionality of MEG channels.
    
    This module implements different channel projection strategies:
    1. Learned: Full learnable projection using a linear layer
    
    Attributes:
        projection: The projection layer (linear transformation)
        strategy: Which projection strategy is being used
        frozen: Whether the projection is frozen or trainable
    """
    
    def __init__(self, n_channels, n_projected_channels, strategy='learned', 
                 data_samples=None, freeze=False, log_dir=None):
        """Initialize the channel projection module.
        
        Args:
            n_channels: Original number of channels
            n_projected_channels: Target number of channels after projection
            strategy: Projection strategy ('learned', 'pca', 'selection')
            data_samples: Representative data samples for PCA initialization
            freeze: Whether to freeze the projection layer
            log_dir: Directory for logging
        """
        super().__init__()
        self.n_channels = n_channels
        self.n_projected_channels = n_projected_channels
        self.strategy = strategy
        self.logger = logging.getLogger(__name__ + ".ChannelProjection")
        
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
            file_handler = logging.FileHandler(os.path.join(log_dir, "channel_projection.log"))
            file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
            self.logger.addHandler(file_handler)
            
        # Initialize projection based on strategy
        if strategy == 'learned':
            # Fully learnable projection
            self.projection = nn.Linear(n_channels, n_projected_channels, bias=False)
            nn.init.orthogonal_(self.projection.weight)
            self.logger.info(f"Initialized learnable projection: {n_channels} -> {n_projected_channels}")
        else:
            raise ValueError(f"Unknown projection strategy: {strategy}")
            
        # Freeze if requested
        self.frozen = freeze
        if freeze:
            for param in self.projection.parameters():
                param.requires_grad = False
            self.logger.info("Projection layer frozen (not trainable)")
    
    def forward(self, x):
        """Apply channel projection.
        
        Args:
            x: Input tensor of shape (..., n_channels, ...)
            
        Returns:
            Projected tensor with reduced channel dimension
        """        
        # Get original shape and number of dimensions
        orig_shape = x.shape
        ndim = len(orig_shape)
        
        # For batched data with segments
        if ndim == 4:  # (batch_size, n_segments, n_channels, n_samples)
            # Reshape to bring channels to last dimension for linear projection
            batch_size, n_segments, _, n_samples = orig_shape
            x_reshaped = x.permute(0, 1, 3, 2).reshape(-1, self.n_channels)
            
            # Apply projection
            x_projected = self.projection(x_reshaped)
            
            # Reshape back to original format with reduced channels
            x_out = x_projected.reshape(batch_size, n_segments, n_samples, self.n_projected_channels)
            x_out = x_out.permute(0, 1, 3, 2)
            
        # For batched data without segments (only individual clips)
        elif ndim == 3:  # (batch_size, n_channels, n_samples)
            # Reshape to bring channels to last dimension
            batch_size, _, n_samples = orig_shape
            x_reshaped = x.permute(0, 2, 1).reshape(-1, self.n_channels)
            
            # Apply projection
            x_projected = self.projection(x_reshaped)
            
            # Reshape back
            x_out = x_projected.reshape(batch_size, n_samples, self.n_projected_channels)
            x_out = x_out.permute(0, 2, 1)
            
        else:
            raise ValueError(f"Unsupported input tensor shape: {orig_shape}")
            
        return x_out
    
    def freeze(self):
        """Freeze the projection layer."""
        for param in self.projection.parameters():
            param.requires_grad = False
        self.frozen = True
        self.logger.info("Projection layer frozen")
    
    def unfreeze(self):
        """Unfreeze the projection layer."""
        for param in self.projection.parameters():
            param.requires_grad = True
        self.frozen = False
        self.logger.info("Projection layer unfrozen")


class BIOTHierarchicalEncoder(nn.Module):
    """BIOT Hierarchical Encoder with Channel Projection.

    This encoder processes chunks of biomedical signals using a hierarchical attention approach:
    1. Intra-segment attention with a CLS token to obtain a summary of each segment
    2. Inter-segment attention on the segment summaries for context across segments
    
    This architecture is designed specifically for MEG data with many channels (275+)
    and enables processing segments with temporal context for improved generalization.

    We add a channel projection layer to reduce the number of channels before processing.
    
    Attributes:
        segment_encoder: Modified BIOT encoder for processing individual segments
        segment_positional_encoding: Positional encoding for segment positions
        inter_segment_transformer: Transformer for processing segment representations
        classifier: Classification head for final prediction
    """
    
    def __init__(
            self,
            emb_size: int = 256,
            heads: int = 8,
            segment_encoder_depth: int = 4,
            inter_segment_depth: int = 2,
            n_channels: int = 275,
            n_projected_channels: int = 64,  # New parameter for projected channels
            token_size: int = 200,
            overlap: float = 0.0,
            raw: bool = False,
            n_segments: int = 100,
            n_classes: int = 2,
            projection_strategy: str = 'learned',  # Projection strategy
            freeze_projection: bool = False,  # Whether to initially freeze projection
            log_dir: Optional[str] = None,
            **kwargs
    ):
        """Initialize the BIOT hierarchical encoder with channel projection.
        
        Args:
            emb_size: Size of the embedding vectors.
            heads: Number of attention heads.
            segment_encoder_depth: Number of transformer layers for intra-segment attention.
            inter_segment_depth: Number of transformer layers for inter-segment attention.
            n_channels: Number of input channels (275 for MEG).
            n_projected_channels: Number of channels after projection.
            token_size: Number of FFT points for Spectral mode / Number of samples for Raw mode.
            overlap: Overlap between patches for raw data mode.
            raw: Whether to use raw time-domain processing.
            n_segments: number of segments in a chunk.
            n_classes: Number of output classes.
            projection_strategy: Strategy for channel projection ('learned', only that choice for now but we could extend later).
            freeze_projection: Whether to freeze the projection layer initially.
            log_dir: Optional directory for log files. If None, logs to console only.
            **kwargs: Additional keyword arguments.
        """
        super().__init__()
        self.logger = logging.getLogger(__name__ + ".BIOTHierarchicalEncoderWithProjection")
        
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
            file_handler = logging.FileHandler(os.path.join(log_dir, "biot_hierarchical_encoder_with_projection.log"))
            file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
            self.logger.addHandler(file_handler)

        self.logger.info(f"Initialized hierarchical encoder with channel projection")
        self.logger.info(f"Channel projection: {n_channels} -> {n_projected_channels} using {projection_strategy} strategy")
        
        # Channel projection layer (to be initialized with data later)
        if n_projected_channels < n_channels:
            self.channel_projection = ChannelProjection(
                n_channels=n_channels,
                n_projected_channels=n_projected_channels,
                strategy=projection_strategy,
                freeze=freeze_projection,
                log_dir=log_dir
            )
        else:
            self.channel_projection = Identity()
            self.logger.info("No channel projection needed (n_projected_channels >= n_channels)")
        
        # Save channel dimensions
        self.n_channels = n_channels
        self.n_projected_channels = n_projected_channels
        
        # Modified BIOT encoder for segment-level processing with CLS token
        # Now works with projected channels
        self.segment_encoder = ModifiedBIOTEncoder(
            emb_size=emb_size,
            heads=heads,
            depth=segment_encoder_depth,
            n_channels=n_projected_channels,  # Use projected channel count
            token_size=token_size,
            overlap=overlap,
            raw=raw,
            log_dir=log_dir
        )
        
        # Segment positional encoding for inter-segment attention
        self.segment_positional_encoding = PositionalEncoding(
            d_model=emb_size,
            max_len=n_segments,
            log_dir=log_dir
        )
        
        # Inter-segment transformer
        self.inter_segment_transformer = LinearAttentionTransformer(
            dim=emb_size,
            heads=heads,
            depth=inter_segment_depth,
            max_seq_len=n_segments,
            attn_layer_dropout=0.2,
            attn_dropout=0.2,
        )
        
        # Classification head
        self.classifier = ClassificationHead(
            emb_size=emb_size,
            n_classes=n_classes,
            log_dir=log_dir
        )
    
    def freeze_projection(self):
        """Freeze the channel projection layer."""
        self.channel_projection.freeze()
    
    def unfreeze_projection(self):
        """Unfreeze the channel projection layer."""
        self.channel_projection.unfreeze()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the BIOT hierarchical encoder with channel projection.
        
        Args:
            x: Input tensor of shape (batch_size, n_segments, n_channels, n_samples_per_segment).
            
        Returns:
            Classification logits for each segment of shape (batch_size, n_segments, n_classes).
        """
        batch_size, n_segments, n_channels, n_samples = x.shape
        
        # Apply channel projection
        x_projected = self.channel_projection(x)
        
        # Process each segment individually to get segment embeddings
        segment_embeddings = []
        
        for i in range(n_segments):
            # Extract the current segment with projected channels
            segment = x_projected[:, i, :, :]  # (batch_size, n_projected_channels, n_samples)
            
            # Get the segment representation using the modified BIOT encoder with CLS token
            segment_emb = self.segment_encoder(segment)  # (batch_size, emb_size)
            segment_embeddings.append(segment_emb.unsqueeze(1))  # (batch_size, 1, emb_size)
        
        # Concatenate segment embeddings
        segment_embeddings = torch.cat(segment_embeddings, dim=1)  # (batch_size, n_segments, emb_size)
        
        # Add segment positional encodings for temporal context
        segment_embeddings = self.segment_positional_encoding(segment_embeddings)
        
        # Process with inter-segment transformer to get contextualized segment representations
        output_embeddings = self.inter_segment_transformer(segment_embeddings)  # (batch_size, n_segments, emb_size)
        
        # Apply classification head to each segment embedding
        # reshape to (batch_size * n_segments, emb_size)
        output_embeddings = output_embeddings.view(-1, output_embeddings.size(-1))
        logits = self.classifier(output_embeddings)  # (batch_size * n_segments, n_classes)
        # reshape back to (batch_size, n_segments, n_classes)
        logits = logits.view(batch_size, n_segments, -1)
        return logits


class ModifiedBIOTEncoder(nn.Module):
    """Modified BIOT Encoder with CLS token for intra-segment attention.
    
    This is a modification of the original BIOT encoder to include a CLS token
    that aggregates information from all channels and temporal patches.
    
    Attributes:
        patch_frequency_embedding: Embedding module for spectral data
        patch_time_embedding: Embedding module for raw time data
        cls_token: Learnable CLS token added to each segment
        transformer: Transformer model for intra-segment attention
        positional_encoding: Positional encoding module
        channel_tokens: Embedding for channel tokens
    """
    
    def __init__(
            self,
            emb_size: int = 256,
            heads: int = 8,
            depth: int = 4,
            n_channels: int = 275,
            token_size: int = 200,
            overlap: float = 0.0,
            raw: bool = False,
            log_dir: Optional[str] = None,
            **kwargs
    ):
        """Initialize the modified BIOT encoder.
        
        Args:
            emb_size: Size of the embedding vectors.
            heads: Number of attention heads.
            depth: Number of transformer layers.
            n_channels: Number of input channels.
            token_size: Number of FFT points for Spectral mode / Number of samples for Raw mode.
            overlap: Overlap between patches for raw data mode.
            raw: Whether to use raw time-domain processing.
            log_dir: Optional directory for log files. If None, logs to console only.
            **kwargs: Additional keyword arguments.
        """
        super().__init__()
        self.logger = logging.getLogger(__name__ + ".ModifiedBIOTEncoder")
        
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
            file_handler = logging.FileHandler(os.path.join(log_dir, "modified_biot_encoder.log"))
            file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
            self.logger.addHandler(file_handler)

        self.n_fft = token_size
        self.hop_length = int(token_size * overlap)
        self.raw = raw
        
        self.logger.info(f"Modified BIOT encoder initialized")
        self.logger.info(f"Processing mode: {'Raw time-domain' if raw else 'Spectral'}")

        # Create embedding modules for both spectral and raw data processing
        self.patch_frequency_embedding = PatchFrequencyEmbedding(
            emb_size=emb_size, n_freq=self.n_fft // 2 + 1, log_dir=log_dir
        )

        # Add the patch time embedding for raw data
        self.patch_time_embedding = PatchTimeEmbedding(
            emb_size=emb_size, patch_size=token_size, overlap=overlap, log_dir=log_dir
        )

        # CLS token for aggregating segment information
        self.cls_token = nn.Parameter(torch.randn(1, 1, emb_size))

        # Transformer with increased max_seq_len to accommodate the extra CLS token
        self.transformer = LinearAttentionTransformer(
            dim=emb_size,
            heads=heads,
            depth=depth,
            max_seq_len=1025,  # 1024 + 1 for CLS token
            attn_layer_dropout=0.2,
            attn_dropout=0.2,
        )
        self.positional_encoding = PositionalEncoding(emb_size, log_dir=log_dir)

        # Channel token embeddings
        self.channel_tokens = nn.Embedding(n_channels, emb_size)
        self.index = nn.Parameter(
            torch.LongTensor(range(n_channels)), requires_grad=False
        )

    def forward(self, x: torch.Tensor, n_channel_offset: int = 0, perturb: bool = False) -> torch.Tensor:
        """Forward pass of the modified BIOT encoder.
        
        Args:
            x: Input tensor of shape (batch_size, channel, ts).
            n_channel_offset: Offset for channel tokens.
            perturb: Whether to randomly perturb the sequence.
            
        Returns:
            CLS token embedding of shape (batch_size, emb_size).
        """
        batch_size = x.shape[0]
        emb_seq = []
        
        # Process each channel
        for i in range(x.shape[1]):
            # Get the current channel data
            channel_data = x[:, i:i + 1, :]

            # Process channel data based on mode (raw or spectral)
            if self.raw:
                # Raw time-series data processing
                channel_emb = self.patch_time_embedding(channel_data)
            else:
                # Spectral data processing
                channel_spec = stft(channel_data, n_fft=self.n_fft, hop_length=self.hop_length)
                channel_emb = self.patch_frequency_embedding(channel_spec)

            batch_size_inner, ts, _ = channel_emb.shape

            # Add channel token embedding
            channel_token_emb = (
                self.channel_tokens(
                    self.index[i + n_channel_offset]
                )
                .unsqueeze(0)
                .unsqueeze(0)
                .repeat(batch_size_inner, ts, 1)
            )

            # Add positional encoding
            channel_emb = self.positional_encoding(channel_emb + channel_token_emb)

            # Apply perturbation if specified
            if perturb:
                ts = channel_emb.shape[1]
                ts_new = np.random.randint(ts // 2, ts)
                selected_ts = np.random.choice(range(ts), ts_new, replace=False)
                channel_emb = channel_emb[:, selected_ts]
                
            emb_seq.append(channel_emb)

        # Concatenate embeddings from all channels
        emb = torch.cat(emb_seq, dim=1)  # (batch_size, channels*ts, emb_size)
        
        # Add CLS token at the beginning
        cls_tokens = self.cls_token.repeat(batch_size, 1, 1)  # (batch_size, 1, emb_size)
        emb_with_cls = torch.cat([cls_tokens, emb], dim=1)  # (batch_size, 1 + channels*ts, emb_size)
        
        # Process with transformer
        output = self.transformer(emb_with_cls)  # (batch_size, 1 + channels*ts, emb_size)
        
        # Extract CLS token representation for segment summary
        cls_output = output[:, 0, :]  # (batch_size, emb_size)
        
        return cls_output


def setup_logging(log_dir: Optional[str] = None, log_level: int = logging.INFO):
    """Set up logging configuration.
    
    Args:
        log_dir: Directory for log files. If None, logs to console only.
        log_level: Logging level.
    """
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    
    # Remove existing handlers to avoid duplicate logs
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    console_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    root_logger.addHandler(console_handler)
    
    # File handler if log_dir is provided
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
        file_handler = logging.FileHandler(os.path.join(log_dir, "biot.log"))
        file_handler.setLevel(log_level)
        file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
        root_logger.addHandler(file_handler)


if __name__ == "__main__":
    # Set up command-line argument parsing
    import argparse
    
    parser = argparse.ArgumentParser(description="Test BIOT models")
    parser.add_argument("--log_dir", type=str, default="./logs", help="Directory for log files")
    parser.add_argument("--log_level", type=str, default="INFO", 
                       choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
                       help="Logging level")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for testing")
    parser.add_argument("--channels", type=int, default=2, help="Number of channels in test data")
    parser.add_argument("--time_steps", type=int, default=2000, help="Number of time steps in test data")
    parser.add_argument("--n_fft", type=int, default=200, help="FFT size")
    parser.add_argument("--hop_length", type=int, default=100, help="Hop length for STFT")
    parser.add_argument("--patch_size", type=int, default=100, help="Patch size for raw mode")
    parser.add_argument("--overlap", type=float, default=0.0, help="Overlap for raw mode (0.0-1.0)")
    parser.add_argument("--raw", action="store_true", help="Use raw time-domain processing")
    
    args = parser.parse_args()
    
    # Set up logging
    log_level = getattr(logging, args.log_level)
    setup_logging(args.log_dir, log_level)
    logger = logging.getLogger(__name__)
    
    logger.info("Testing BIOT models")
    logger.info(f"Arguments: {args}")
    
    # Create test data
    x = torch.randn(args.batch_size, args.channels, args.time_steps)
    logger.info(f"Test data shape: {x.shape}")

    # Test with spectral processing
    if not args.raw:
        logger.info("Testing spectral processing model")
        model = BIOTClassifier(
            n_fft=args.n_fft, 
            hop_length=args.hop_length, 
            depth=4, 
            heads=8, 
            raw=False,
            log_dir=args.log_dir
        )
        out = model(x)
        logger.info(f"Spectral output shape: {out.shape}")

    # Test with raw data processing (no overlap)
    logger.info("Testing raw processing model (no overlap)")
    model_raw = BIOTClassifier(
        n_fft=args.n_fft, 
        hop_length=args.hop_length, 
        depth=4, 
        heads=8, 
        raw=True, 
        patch_size=args.patch_size, 
        overlap=0.0,
        log_dir=args.log_dir
    )
    out_raw = model_raw(x)
    logger.info(f"Raw output shape (no overlap): {out_raw.shape}")

    # Test with raw data processing (with overlap)
    if args.overlap > 0:
        logger.info(f"Testing raw processing model (with {args.overlap*100}% overlap)")
        model_raw_overlap = BIOTClassifier(
            n_fft=args.n_fft, 
            hop_length=args.hop_length, 
            depth=4, 
            heads=8, 
            raw=True, 
            patch_size=args.patch_size,
            overlap=args.overlap,
            log_dir=args.log_dir
        )
        out_raw_overlap = model_raw_overlap(x)
        logger.info(f"Raw output shape (with overlap): {out_raw_overlap.shape}")
    
    logger.info("All tests completed successfully")