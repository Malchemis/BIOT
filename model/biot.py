import logging
import math
import os
from typing import Optional, Tuple

import torch
import torch.nn as nn
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
    
class SequentialClassificationHead(nn.Module):
    """Classification head for multi-segment prediction.
    
    This module processes the sequence of embeddings from the transformer 
    and produces a prediction for each segment using context from the whole chunk.
    
    Attributes:
        method (str): Method for sequence processing ('lstm', 'attention', or 'gating').
        processor: Sequential processing module (LSTM, Self-Attention, or Context Gating).
        classifier: Final classification layer.
    """
    
    def __init__(self, emb_size: int, n_classes: int, n_segments: int, 
                 method: str = 'lstm', hidden_size: int = 128, 
                 num_layers: int = 1, bidirectional: bool = True,
                 num_heads: int = 8, dropout: float = 0.1,
                 log_dir: Optional[str] = None):
        """Initialize the sequential classification head.
        
        Args:
            emb_size: Size of the input embeddings.
            n_classes: Number of output classes.
            n_segments: Number of segments to predict.
            method: Method for processing ('lstm', 'attention', or 'gating').
            hidden_size: Size of LSTM hidden layers.
            num_layers: Number of LSTM layers.
            bidirectional: Whether to use bidirectional LSTM.
            num_heads: Number of attention heads.
            dropout: Dropout rate.
            log_dir: Optional directory for log files.
        """
        super().__init__()
        self.n_segments = n_segments
        self.method = method
        self.logger = logging.getLogger(__name__ + ".SequentialClassificationHead")
        
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
            file_handler = logging.FileHandler(os.path.join(log_dir, "sequential_head.log"))
            file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
            self.logger.addHandler(file_handler)
            
        self.logger.info(f"Initializing with method={method}, n_segments={n_segments}")
        
        # Choose the sequence processing method
        if method == 'lstm':
            self.processor = nn.LSTM(
                input_size=emb_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
                bidirectional=bidirectional,
                dropout=dropout if num_layers > 1 else 0
            )
            output_size = hidden_size * (2 if bidirectional else 1)
        elif method == 'attention':
            self.processor = nn.MultiheadAttention(
                embed_dim=emb_size,
                num_heads=num_heads,
                dropout=dropout,
                batch_first=True
            )
            # Add a normalization layer for attention
            self.norm = nn.LayerNorm(emb_size)
            output_size = emb_size
        elif method == 'gating':
            self.processor = nn.Sequential(
                nn.Linear(emb_size, emb_size),
                nn.Sigmoid()
            )
            output_size = emb_size
        else:
            raise ValueError(f"Unknown method: {method}")
        
        # Final classification layer
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(output_size, n_classes)
        )
        
        # Additional projection if we need to align the sequence length
        self.needs_alignment = True
        if self.needs_alignment:
            self.align_layer = nn.Linear(1, 1)  # Placeholder, will be properly initialized in forward

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Process sequence and classify each segment.
        
        Args:
            x: Input tensor from transformer (batch_size, seq_len, emb_size).
            
        Returns:
            Logits for each segment (batch_size, n_segments, n_classes).
        """
        batch_size, seq_len, emb_size = x.shape
        
        # Process the sequence with the chosen method
        if self.method == 'lstm':
            # Process through LSTM
            lstm_out, _ = self.processor(x)
            processed = lstm_out
        elif self.method == 'attention':
            # Self-attention processing
            attn_out, _ = self.processor(x, x, x)
            processed = self.norm(x + attn_out)  # Residual connection
        elif self.method == 'gating':
            # Context gating
            gates = self.processor(x)
            processed = x * gates
        
        # Align sequence length with the number of segments if needed
        if seq_len != self.n_segments:
            # Initialize alignment layer if this is the first forward pass
            if hasattr(self, 'align_layer') and isinstance(self.align_layer, nn.Linear) and self.align_layer.weight.shape[0] == 1:
                self.align_layer = nn.Linear(seq_len, self.n_segments)
                if x.is_cuda:
                    self.align_layer = self.align_layer.cuda()
                self.logger.info(f"Initialized alignment layer: {seq_len} → {self.n_segments}")
            
            # Transpose, align, and transpose back
            # (batch_size, seq_len, emb_size) → (batch_size, emb_size, seq_len)
            transposed = processed.transpose(1, 2)
            # (batch_size, emb_size, seq_len) → (batch_size, emb_size, n_segments)
            aligned = self.align_layer(transposed)
            # (batch_size, emb_size, n_segments) → (batch_size, n_segments, emb_size)
            processed = aligned.transpose(1, 2)
            
            self.logger.debug(f"Aligned sequence: {seq_len} → {self.n_segments}")
        
        # Process through the classifier to get logits for each segment
        logits = self.classifier(processed)  # (batch_size, n_segments, n_classes)
        
        return logits


class PositionalEncoding(nn.Module):
    """Module for adding positional encoding to embeddings.
    
    This implementation uses sinusoidal position encoding to give the model
    information about the position of tokens in the sequence.
    
    Attributes:
        dropout (nn.Dropout): Dropout layer.
        pe (torch.Tensor): Precomputed positional encoding. (registered as to not be a model parameter)
    """
    
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 1000, log_dir: Optional[str] = None):
        """Initialize the positional encoding.
        
        Args:
            d_model: Dimensionality of the model.
            dropout: Dropout rate.
            max_len: Maximum sequence length.
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
            n_fft: int = 200,
            hop_length: int = 100,
            raw: bool = False,      # Parameter to toggle between spectral and raw data
            patch_size: int = 100,  # Patch size for raw data mode
            overlap: float = 0.0,   # Overlap between patches for raw data mode
            log_dir: Optional[str] = None,
            **kwargs
    ):
        """Initialize the BIOT encoder.
        
        Args:
            emb_size: Size of the embedding vectors.
            heads: Number of attention heads.
            depth: Number of transformer layers.
            n_channels: Number of input channels.
            n_fft: Number of FFT points.
            hop_length: Hop length for STFT.
            raw: Whether to use raw time-domain processing.
            patch_size: Patch size for raw data mode.
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

        self.n_fft = n_fft
        self.hop_length = hop_length
        self.raw = raw  # Store the raw mode flag
        
        self.logger.info(f"Initialized with emb_size={emb_size}, heads={heads}, depth={depth}, "
                         f"n_channels={n_channels}, n_fft={n_fft}, hop_length={hop_length}")
        self.logger.info(f"Processing mode: {'Raw time-domain' if raw else 'Spectral'}")
        
        if raw:
            self.logger.info(f"Raw mode parameters: patch_size={patch_size}, overlap={overlap}")

        # Create embedding modules for both spectral and raw data processing
        self.patch_frequency_embedding = PatchFrequencyEmbedding(
            emb_size=emb_size, n_freq=self.n_fft // 2 + 1, log_dir=log_dir
        )

        # Add the patch time embedding for raw data
        self.patch_time_embedding = PatchTimeEmbedding(
            emb_size=emb_size, patch_size=patch_size, overlap=overlap, log_dir=log_dir
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
        self.channel_tokens = nn.Embedding(n_channels, 256)
        self.index = nn.Parameter(
            torch.LongTensor(range(n_channels)), requires_grad=False
        )

    def stft(self, sample: torch.Tensor) -> torch.Tensor:
        """Compute the Short-Time Fourier Transform.
        
        Args:
            sample: Input tensor of shape (batch_size, 1, ts).
            
        Returns:
            Magnitude of the STFT of shape (batch_size, freq, time).
        """
        spectral = torch.stft(
            input=sample.squeeze(1),  # from shape (batch_size, 1, ts) to (batch_size, ts)
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            # window=torch.hann_window(self.n_fft).to(sample.device),
            center=False,
            onesided=True,
            return_complex=True,
        )
        return torch.abs(spectral)

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
                channel_spec = self.stft(channel_data) # shape: batch, freq, ts
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
        emb = self.transformer(emb)#.mean(dim=1) # don't collapse the time dimension 
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


class BIOTSequenceClassifier(nn.Module):
    """Biomedical Input-Output Transformer (BIOT) for multi-segment classification.
    
    This model processes an entire chunk of data and outputs predictions 
    for each segment while considering the context of the whole chunk.
    
    Attributes:
        biot (BIOTEncoder): BIOT encoder for feature extraction.
        classifier (SequentialClassificationHead): Classification head.
    """
    
    def __init__(self, emb_size: int = 256, heads: int = 8, depth: int = 4, 
                 n_classes: int = 1, n_channels: int = 16, n_segments: int = 1, 
                 seq_method: str = 'lstm', n_fft: int = 200, hop_length: int = 100,
                 raw: bool = False, patch_size: int = 100, overlap: float = 0.0, 
                 log_dir: Optional[str] = None, **kwargs):
        """Initialize the BIOT sequence classifier.
        
        Args:
            emb_size: Size of the embedding vectors.
            heads: Number of attention heads.
            depth: Number of transformer layers.
            n_classes: Number of output classes.
            n_channels: Number of input channels.
            n_segments: Number of segments to predict.
            seq_method: Method for sequence processing ('lstm', 'attention', 'gating').
            n_fft: Number of FFT points.
            hop_length: Hop length for STFT.
            raw: Whether to use raw time-domain processing.
            patch_size: Patch size for raw data.
            overlap: Overlap between patches.
            log_dir: Optional directory for log files.
            **kwargs: Additional keyword arguments.
        """
        super().__init__()
        self.logger = logging.getLogger(__name__ + ".BIOTSequenceClassifier")
        
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
            file_handler = logging.FileHandler(os.path.join(log_dir, "biot_sequence.log"))
            file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
            self.logger.addHandler(file_handler)
            
        self.logger.info(f"Initializing with n_segments={n_segments}, method={seq_method}")
        
        # Initialize the BIOT encoder (no modifications needed here)
        self.biot = BIOTEncoder(
            emb_size=emb_size, 
            heads=heads, 
            depth=depth, 
            n_channels=n_channels,
            n_fft=n_fft, 
            hop_length=hop_length,
            raw=raw, 
            patch_size=patch_size, 
            overlap=overlap, 
            log_dir=log_dir,
            **kwargs
        )
        
        # Initialize the sequential classification head
        self.classifier = SequentialClassificationHead(
            emb_size=emb_size,
            n_classes=n_classes,
            n_segments=n_segments,
            method=seq_method,
            log_dir=log_dir
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the model.
        
        Args:
            x: Input tensor of shape (batch_size, channel, ts).
            
        Returns:
            Logits for each segment (batch_size, n_segments, n_classes).
        """
        # Process through BIOT encoder (which now preserves temporal info)
        embeddings = self.biot(x)  # (batch_size, seq_len, emb_size)
        
        # Process through sequential classification head
        logits = self.classifier(embeddings)  # (batch_size, n_segments, n_classes)
        return logits


class UnsupervisedPretrain(nn.Module):
    """Unsupervised pretraining model for BIOT.
    
    This model uses contrastive learning to pretrain the BIOT encoder.
    
    Attributes:
        biot (BIOTEncoder): BIOT encoder for feature extraction.
        prediction (nn.Sequential): MLP for feature prediction.
    """
    
    def __init__(self, emb_size: int = 256, heads: int = 8, depth: int = 4, 
                 n_channels: int = 18, raw: bool = False, log_dir: Optional[str] = None, **kwargs):
        """Initialize the unsupervised pretraining model.
        
        Args:
            emb_size: Size of the embedding vectors.
            heads: Number of attention heads.
            depth: Number of transformer layers.
            n_channels: Number of input channels.
            raw: Whether to use raw time-domain processing.
            log_dir: Optional directory for log files. If None, logs to console only.
            **kwargs: Additional parameters passed to BIOTEncoder.
        """
        super(UnsupervisedPretrain, self).__init__()
        self.logger = logging.getLogger(__name__ + ".UnsupervisedPretrain")
        
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
            file_handler = logging.FileHandler(os.path.join(log_dir, "unsupervised_pretrain.log"))
            file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
            self.logger.addHandler(file_handler)
            
        self.logger.info(f"Initializing unsupervised pretraining model with emb_size={emb_size}, "
                         f"heads={heads}, depth={depth}, n_channels={n_channels}, raw={raw}")
        
        self.biot = BIOTEncoder(emb_size, heads, depth, n_channels, raw=raw, 
                               log_dir=log_dir, **kwargs)
        self.prediction = nn.Sequential(
            nn.Linear(256, 256),
            nn.GELU(),
            nn.Linear(256, 256),
        )

    def forward(self, x: torch.Tensor, n_channel_offset: int = 0) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass of the unsupervised pretraining model.
        
        Args:
            x: Input tensor of shape (batch_size, channel, ts).
            n_channel_offset: Offset for channel tokens.
            
        Returns:
            Tuple of (predicted embeddings, target embeddings).
        """
        emb = self.biot(x, n_channel_offset, perturb=True)
        emb = self.prediction(emb)
        pred_emb = self.biot(x, n_channel_offset)
        return emb, pred_emb


class SupervisedPretrain(nn.Module):
    """Supervised pretraining model for BIOT.
    
    This model trains the BIOT encoder on multiple tasks simultaneously.
    
    Attributes:
        biot (BIOTEncoder): BIOT encoder for feature extraction.
        classifier_chb_mit (ClassificationHead): Classification head for CHB-MIT dataset.
        classifier_iiic_seizure (ClassificationHead): Classification head for IIIC seizure dataset.
        classifier_tuab (ClassificationHead): Classification head for TUAB dataset.
        classifier_tuev (ClassificationHead): Classification head for TUEV dataset.
    """
    
    def __init__(self, emb_size: int = 256, heads: int = 8, depth: int = 4, 
                 raw: bool = False, log_dir: Optional[str] = None, **kwargs):
        """Initialize the supervised pretraining model.
        
        Args:
            emb_size: Size of the embedding vectors.
            heads: Number of attention heads.
            depth: Number of transformer layers.
            raw: Whether to use raw time-domain processing.
            log_dir: Optional directory for log files. If None, logs to console only.
            **kwargs: Additional parameters passed to BIOTEncoder.
        """
        super().__init__()
        self.logger = logging.getLogger(__name__ + ".SupervisedPretrain")
        
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
            file_handler = logging.FileHandler(os.path.join(log_dir, "supervised_pretrain.log"))
            file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
            self.logger.addHandler(file_handler)
            
        self.logger.info(f"Initializing supervised pretraining model with emb_size={emb_size}, "
                         f"heads={heads}, depth={depth}, raw={raw}")
        
        self.biot = BIOTEncoder(emb_size=emb_size, heads=heads, depth=depth, raw=raw, 
                               log_dir=log_dir, **kwargs)
        self.classifier_chb_mit = ClassificationHead(emb_size, 1, log_dir=log_dir)
        self.classifier_iiic_seizure = ClassificationHead(emb_size, 6, log_dir=log_dir)
        self.classifier_tuab = ClassificationHead(emb_size, 1, log_dir=log_dir)
        self.classifier_tuev = ClassificationHead(emb_size, 6, log_dir=log_dir)

    def forward(self, x: torch.Tensor, task: str = "chb-mit") -> torch.Tensor:
        """Forward pass of the supervised pretraining model.
        
        Args:
            x: Input tensor of shape (batch_size, channel, ts).
            task: Which task to perform. Options are "chb-mit", "iiic-seizure", "tuab", or "tuev".
            
        Returns:
            Logits for the specific task.
        """
        x = self.biot(x)
        if task == "chb-mit":
            x = self.classifier_chb_mit(x)
        elif task == "iiic-seizure":
            x = self.classifier_iiic_seizure(x)
        elif task == "tuab":
            x = self.classifier_tuab(x)
        elif task == "tuev":
            x = self.classifier_tuev(x)
        else:
            error_msg = f"Task {task} is not implemented. Choose from: chb-mit, iiic-seizure, tuab, tuev"
            self.logger.error(error_msg)
            raise NotImplementedError(error_msg)
        return x


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

    # Test unsupervised pretraining
    logger.info("Testing unsupervised pretraining model")
    model_unsup = UnsupervisedPretrain(
        n_fft=args.n_fft, 
        hop_length=args.hop_length, 
        depth=4, 
        heads=8, 
        raw=args.raw, 
        patch_size=args.patch_size,
        overlap=args.overlap,
        log_dir=args.log_dir
    )
    out1, out2 = model_unsup(x)
    logger.info(f"Unsupervised output shapes: {out1.shape}, {out2.shape}")
    
    logger.info("All tests completed successfully")