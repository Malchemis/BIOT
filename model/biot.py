import logging
import math
import os
from typing import Optional, Tuple

import torch
import torch.nn as nn
from torch.nn import MultiheadAttention
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
    
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 800, log_dir: Optional[str] = None):
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
        self.channel_tokens = nn.Embedding(n_channels, emb_size)
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
            window=torch.ones(self.n_fft, device=sample.device),
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


class MultiSegmentBIOTEncoder(nn.Module):
    """Enhanced BIOT Encoder for processing multiple segments simultaneously.
    
    This encoder processes chunks of MEG data by treating them as one long sequence
    with segment tokens to mark boundaries between segments. It preserves the
    channel token mechanism while adding contextual information across segments.
    
    Attributes:
        base_encoder (BIOTEncoder): The underlying BIOT encoder
        segment_tokens (nn.Embedding): Embedding for segment tokens
        segment_indices (nn.Parameter): Parameter for segment indices
    """
    
    def __init__(
            self,
            emb_size: int = 256,
            heads: int = 8,
            depth: int = 4,
            n_channels: int = 275,
            n_projected_channels: int = 64,
            max_segments: int = 300,  # Maximum number of segments in a chunk
            n_fft: int = 200,
            hop_length: int = 100,
            raw: bool = False,
            patch_size: int = 100,
            overlap: float = 0.0,
            log_dir: Optional[str] = None,
            **kwargs
    ):
        """Initialize the multi-segment BIOT encoder.
        
        Args:
            emb_size: Size of the embedding vectors
            heads: Number of attention heads
            depth: Number of transformer layers
            n_channels: Number of input channels
            max_segments: Maximum number of segments per chunk
            n_fft: Number of FFT points
            hop_length: Hop length for STFT
            raw: Whether to use raw time-domain processing
            patch_size: Patch size for raw data mode
            overlap: Overlap between patches for raw data mode
            log_dir: Optional directory for log files
            **kwargs: Additional keyword arguments
        """
        super().__init__()
        self.logger = logging.getLogger(__name__ + ".MultiSegmentBIOTEncoder")
        
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
            file_handler = logging.FileHandler(os.path.join(log_dir, "multi_segment_biot_encoder.log"))
            file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
            self.logger.addHandler(file_handler)

        # Reduction spatial dimension to n_projected_channels
        self.channel_reduction = nn.Conv1d(
            in_channels=n_channels,
            out_channels=n_projected_channels,
            kernel_size=1,
            stride=1,
        ) # linear combination across channels
        
        # Create the base BIOT encoder
        self.base_encoder = BIOTEncoder(
            emb_size=emb_size,
            heads=heads,
            depth=depth,
            n_channels=n_projected_channels, # Reduced channels
            n_fft=n_fft,
            hop_length=hop_length,
            raw=raw,
            patch_size=patch_size,
            overlap=overlap,
            log_dir=log_dir,
            **kwargs
        )
        
        # Add segment tokens embeddings
        self.segment_tokens = nn.Embedding(max_segments, emb_size)
        self.segment_indices = nn.Parameter(
            torch.arange(max_segments), requires_grad=False
        )

        self.segment_attention = MultiheadAttention(emb_size, heads)
        self.segment_layer_norm = nn.LayerNorm(emb_size)

        
        # Save parameters
        self.raw = raw
        self.n_channels = n_channels
        self.projected_channels = n_projected_channels
        self.max_segments = max_segments
        self.emb_size = emb_size
        self.segment_length = None  # Will be determined during first forward pass
        
        self.logger.info(f"Initialized with max_segments={max_segments}, n_channels={n_channels}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the multi-segment BIOT encoder.
        
        Args:
            x: Input tensor of shape (batch_size, n_channels, total_time)
                where total_time = n_segments * segment_length
            
        Returns:
            Segment embeddings of shape (batch_size, n_segments, emb_size)
        """
        batch_size, n_channels, total_time = x.shape
        
        # Calculate exact segment length
        self.segment_length = total_time // self.max_segments
        actual_segments = min(total_time // self.segment_length, self.max_segments)
        actual_segments = torch.tensor(actual_segments, device=x.device, dtype=torch.long)
        
        # Channel projection
        projected = self.channel_reduction(x)  # [B, C_proj, T]
        
        # Reshape to segments [B, C_proj, S, L]
        segments = projected.view(
            batch_size, 
            self.projected_channels,
            actual_segments,
            self.segment_length
        )
        
        segment_embeddings = []
        for seg_idx in range(actual_segments):
            seg_data = segments[:, :, seg_idx, :]  # [B, C_proj, L]
            
            if self.raw:
                # Process each projected channel individually for raw data
                channel_embs = []
                for chan_idx in range(self.projected_channels):
                    chan_data = seg_data[:, chan_idx:chan_idx+1, :]  # [B, 1, L]
                    
                    # Process through patch_time_embedding
                    chan_emb = self.base_encoder.patch_time_embedding(chan_data)
                    
                    # Add channel token
                    chan_token = self.base_encoder.channel_tokens(
                        self.base_encoder.index[chan_idx].long()
                    ).view(1, 1, -1)
                    chan_emb += chan_token
                    
                    channel_embs.append(chan_emb)
                
                # Combine channel embeddings
                emb = torch.stack(channel_embs, dim=1).mean(dim=1)  # [B, T, E]
            else:
                # Process each projected channel individually for spectral processing
                channel_embs = []
                for chan_idx in range(self.projected_channels):
                    # Extract single channel data [B, 1, L]
                    chan_data = seg_data[:, chan_idx:chan_idx+1, :]
                    
                    # STFT processing
                    spec = self.base_encoder.stft(chan_data)  # [B, Freq, Time]
                    chan_emb = self.base_encoder.patch_frequency_embedding(spec)
                    
                    # Add channel embedding
                    chan_token = self.base_encoder.channel_tokens(
                        self.base_encoder.index[chan_idx].long() # ensure long type
                    ).unsqueeze(0).unsqueeze(0)  # [1, 1, E]
                    chan_emb += chan_token
                    
                    channel_embs.append(chan_emb)
                
                # Combine channel embeddings
                emb = torch.stack(channel_embs, dim=1).mean(dim=1)  # [B, T, E]
            
            # Add segment embedding and positional encoding
            seg_idx_tensor = torch.tensor([seg_idx], device=x.device, dtype=torch.long)
            emb += self.segment_tokens(seg_idx_tensor).view(1, 1, -1)  # from [1, E] to [1, 1, E] for broadcasting
            emb = self.base_encoder.positional_encoding(emb)
            
            # Transformer processing
            transformed = self.base_encoder.transformer(emb)
            segment_embeddings.append(transformed.mean(dim=1))  # [B, E]

        # Rest of the code remains the same
        all_segments = torch.stack(segment_embeddings, dim=1)
        attn_output, _ = self.segment_attention(all_segments, all_segments, all_segments)
        return self.segment_layer_norm(all_segments + attn_output)
        

class AttClassificationHead(nn.Module):
    def __init__(self, emb_size, n_classes):
        super().__init__()
        self.attention_pool = nn.MultiheadAttention(emb_size, 1)
        self.fc = nn.Sequential(
            nn.LayerNorm(emb_size),
            nn.Linear(emb_size, n_classes)
        )

    def forward(self, x):
        # x shape: [B, S, E]
        x = x.permute(1, 0, 2)  # [S, B, E] for MHA
        attn_output, _ = self.attention_pool(x, x, x)
        attn_output = attn_output.permute(1, 0, 2)  # [B, S, E]
        return self.fc(attn_output)  # [B, S, n_classes]


class MultiSegmentBIOTClassifier(nn.Module):
    """BIOT Classifier for multi-segment classification.
    
    This classifier processes chunks of MEG data and outputs a classification
    score for each segment in the chunk, leveraging contextual information
    across segments.
    
    Attributes:
        encoder (MultiSegmentBIOTEncoder): The encoder for multi-segment processing
        classifier (ClassificationHead): Classification head for segment-level predictions
    """
    
    def __init__(
            self,
            n_classes: int = 1,
            n_channels: int = 16,
            max_segments: int = 16,
            n_fft: int = 200,
            hop_length: int = 100,
            raw: bool = False,
            patch_size: int = 100,
            overlap: float = 0.0,
            emb_size: int = 256,
            heads: int = 8,
            depth: int = 4,
            log_dir: Optional[str] = None,
            **kwargs
    ):
        """Initialize the multi-segment BIOT classifier.
        
        Args:
            n_classes: Number of output classes (1 for binary classification)
            n_channels: Number of input channels
            max_segments: Maximum number of segments per chunk
            n_fft: Number of FFT points
            hop_length: Hop length for STFT
            raw: Whether to use raw time-domain processing
            patch_size: Patch size for raw data mode
            overlap: Overlap between patches for raw data mode
            emb_size: Size of the embedding vectors
            heads: Number of attention heads
            depth: Number of transformer layers
            log_dir: Optional directory for log files
            **kwargs: Additional keyword arguments
        """
        super().__init__()
        self.logger = logging.getLogger(__name__ + ".MultiSegmentBIOTClassifier")
        
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
            file_handler = logging.FileHandler(os.path.join(log_dir, "multi_segment_biot_classifier.log"))
            file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
            self.logger.addHandler(file_handler)
        
        # Multi-segment encoder
        self.encoder = MultiSegmentBIOTEncoder(
            emb_size=emb_size,
            heads=heads,
            depth=depth,
            n_channels=n_channels,
            max_segments=max_segments,
            n_fft=n_fft,
            hop_length=hop_length,
            raw=raw,
            patch_size=patch_size,
            overlap=overlap,
            log_dir=log_dir,
            **kwargs
        )
        
        # Classification head that uses attention pooling
        self.classifier = AttClassificationHead(
            emb_size=emb_size,
            n_classes=n_classes,
        )
        
        self.logger.info(f"Initialized with n_classes={n_classes}, max_segments={max_segments}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the multi-segment BIOT classifier.
        
        Args:
            x: Input tensor of shape (batch_size, n_channels, total_time)
                where total_time = n_segments * segment_length
            
        Returns:
            Logits of shape (batch_size, n_segments, n_classes)
        """
        # Get all segment embeddings [B, S, E]
        segment_embeddings = self.encoder(x)  
        
        # Process all segments simultaneously
        logits = self.classifier(segment_embeddings)  # [B, S, 1]
        return logits.squeeze(-1)  # [B, S]


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