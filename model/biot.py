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

class ChunkClassificationHead(nn.Module):
    def __init__(self, emb_size, chunk_size, mode='attention'):
        super().__init__()
        self.mode = mode
        
        if mode == 'sequential':
            # Bidirectional LSTM to capture temporal dependencies between segments
            self.context_layers = nn.LSTM(
                input_size=emb_size,
                hidden_size=emb_size // 2,
                num_layers=2,  # Use 2 layers for better context modeling
                batch_first=True,
                bidirectional=True,
                dropout=0.2
            )
        elif mode == 'attention':
            # Multi-head self-attention to capture relationships between segments
            self.context_layers = nn.MultiheadAttention(
                embed_dim=emb_size,
                num_heads=8,  # More heads for detailed attention
                batch_first=True,
                dropout=0.2
            )
            self.norm1 = nn.LayerNorm(emb_size)
            
            # Add feed-forward network after attention for better representation
            self.feed_forward = nn.Sequential(
                nn.Linear(emb_size, emb_size*4),
                nn.GELU(),
                nn.Dropout(0.2),
                nn.Linear(emb_size*4, emb_size)
            )
            self.norm2 = nn.LayerNorm(emb_size)
        else:  # 'independent' mode
            self.context_layers = nn.Identity()
        
        # Segment-wise classifier (applied to each segment separately)
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(emb_size, 1)  # Binary classification per segment
        )

    def forward(self, x):
        """
        Args:
            x: Segment embeddings of shape (batch_size, chunk_size, emb_size)
            
        Returns:
            Logits of shape (batch_size, chunk_size)
        """
        # Apply context-capturing layers based on mode
        if self.mode == 'sequential':
            x, _ = self.context_layers(x)
        elif self.mode == 'attention':
            # Self-attention with residual connection and normalization
            residual = x
            attn_output, _ = self.context_layers(x, x, x)
            x = self.norm1(residual + attn_output)
            
            # Feed-forward network with residual connection
            residual = x
            x = self.feed_forward(x)
            x = self.norm2(residual + x)

        # Reshape for segment-wise classification
        batch_size, chunk_size, emb_size = x.shape
        
        # Apply classifier to each segment while preserving batch and segment dimensions
        logits = self.classifier(x.view(-1, emb_size)).view(batch_size, chunk_size)
        
        return logits


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
    
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 1200):
        """Initialize the positional encoding.
        
        Args:
            d_model: Dimensionality of the model.
            dropout: Dropout rate.
            max_len: Maximum sequence length.
        """
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.logger = logging.getLogger(__name__ + ".PositionalEncoding")
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
        self.patch_time_embedding = PatchTimeEmbedding(emb_size=emb_size, patch_size=patch_size, overlap=overlap)

        self.transformer = LinearAttentionTransformer(
            dim=emb_size,
            heads=heads,
            depth=depth,
            max_seq_len=1024,
            attn_layer_dropout=0.2,  # dropout right after self-attention layer
            attn_dropout=0.2,  # dropout post-attention
        )
        self.positional_encoding = PositionalEncoding(emb_size)

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

    def forward(self, x, n_channel_offset=0, perturb=False):
        """Forward pass that processes all segments together while preserving segment awareness.
        
        Args:
            x: Input tensor of shape (batch_size, n_channels, n_segments, ts_per_segment)
            n_channel_offset: Offset for channel tokens
            perturb: Whether to randomly perturb the sequence
            
        Returns:
            Segment-level embeddings of shape (batch_size, n_segments, emb_size)
        """
        batch_size, n_channels, n_segments, ts_per_segment = x.shape
        segment_embs = []
        
        # For each channel
        for i in range(n_channels):
            # Process each segment within this channel
            channel_data = x[:, i:i+1, :, :].reshape(batch_size, 1, n_segments * ts_per_segment)
            
            # Process either with time or frequency embeddings
            if self.raw:
                # Process the entire sequence with segment boundaries preserved
                channel_emb = self.patch_time_embedding(channel_data)  # (batch_size, n_patches, emb_size)
            else:
                # Spectral data processing
                channel_spec = self.stft(channel_data)  # (batch_size, freq, time)
                channel_emb = self.patch_frequency_embedding(channel_spec)  # (batch_size, time, emb_size)
            
            # Add segment position encodings
            # This is crucial - we need to help the transformer distinguish different segments
            n_tokens_per_segment = channel_emb.shape[1] // n_segments
            
            # Add channel token embedding
            channel_token = self.channel_tokens(self.index[i + n_channel_offset])
            channel_token = channel_token.unsqueeze(0).unsqueeze(1)  # (1, 1, emb_size)
            channel_token = channel_token.expand(batch_size, channel_emb.shape[1], -1)  # (batch_size, time, emb_size)
            
            # Add segment encodings
            segment_ids = torch.arange(n_segments, device=x.device).repeat_interleave(n_tokens_per_segment)
            segment_ids = segment_ids[:channel_emb.shape[1]]  # Handle potential length mismatch
            segment_ids = segment_ids.unsqueeze(0).expand(batch_size, -1)  # (batch_size, time)
            
            # Create learnable segment embeddings
            segment_embeddings = nn.Embedding(n_segments, channel_emb.shape[-1]).to(x.device)
            segment_pos = segment_embeddings(segment_ids)  # (batch_size, time, emb_size)
            
            # Combine embeddings: content + channel token + segment position
            channel_emb = channel_emb + channel_token + segment_pos
            
            # Apply standard positional encoding on top
            channel_emb = self.positional_encoding(channel_emb)
            
            # Optional perturbation (while preserving segment structure)
            if perturb:
                # Implement perturbation that respects segment boundaries
                perturbed_embs = []
                for b in range(batch_size):
                    batch_emb = channel_emb[b]
                    segment_lengths = [n_tokens_per_segment] * n_segments
                    
                    # Ensure we don't exceed the actual length
                    if sum(segment_lengths) > batch_emb.shape[0]:
                        segment_lengths[-1] = batch_emb.shape[0] - sum(segment_lengths[:-1])
                    
                    perturbed_batch = []
                    start_idx = 0
                    for seg_len in segment_lengths:
                        if seg_len <= 0:
                            continue
                        
                        seg_data = batch_emb[start_idx:start_idx+seg_len]
                        # Perturb this segment
                        ts_new = np.random.randint(max(seg_len // 2, 1), seg_len)
                        selected_ts = np.random.choice(range(seg_len), ts_new, replace=False)
                        perturbed_batch.append(seg_data[selected_ts])
                        start_idx += seg_len
                    
                    # Concatenate perturbed segments
                    perturbed_embs.append(torch.cat(perturbed_batch, dim=0))
                
                channel_emb = torch.stack(perturbed_embs)
            
            segment_embs.append(channel_emb)
        
        # Concatenate all channels' embeddings
        # Shape: (batch_size, n_channels*time, emb_size)
        emb = torch.cat(segment_embs, dim=1)
        
        # Process through transformer
        transformed = self.transformer(emb)  # (batch_size, n_channels*time, emb_size)
        
        # Now, extract segment-specific representations
        # We need to aggregate tokens that belong to the same segment
        segment_embeddings = []
        
        # Calculate the number of tokens per segment after transformer processing
        tokens_per_channel = transformed.shape[1] // n_channels
        tokens_per_segment = tokens_per_channel // n_segments
        
        for s in range(n_segments):
            segment_tokens = []
            for c in range(n_channels):
                # Calculate start and end indices for this segment in this channel
                start_idx = (c * tokens_per_channel) + (s * tokens_per_segment)
                end_idx = start_idx + tokens_per_segment
                
                # Extract tokens for this segment
                channel_segment_tokens = transformed[:, start_idx:end_idx, :]  # (batch_size, tokens_per_segment, emb_size)
                segment_tokens.append(channel_segment_tokens)
            
            # Concatenate all channels' tokens for this segment
            segment_token_concat = torch.cat(segment_tokens, dim=1)  # (batch_size, n_channels*tokens_per_segment, emb_size)
            
            # Aggregate to get a single embedding per segment (average pooling)
            segment_embedding = segment_token_concat.mean(dim=1)  # (batch_size, emb_size)
            segment_embeddings.append(segment_embedding)
        
        # Stack to get final shape: (batch_size, n_segments, emb_size)
        segment_embeddings = torch.stack(segment_embeddings, dim=1)
        
        return segment_embeddings

        
class BIOTClassifier(nn.Module):
    """Biomedical Input-Output Transformer (BIOT) Classifier."""
    
    def __init__(self, emb_size: int = 256, heads: int = 8, depth: int = 4, chunk_size=100,
                 raw: bool = False, classification_mode: str = 'attention',
                 log_dir: Optional[str] = None, **kwargs):
        """Initialize the BIOT classifier.
        
        Args:
            emb_size: Size of the embedding vectors.
            heads: Number of attention heads.
            depth: Number of transformer layers.
            chunk_size: Number of segments in each chunk.
            raw: Whether to use raw time-domain processing.
            classification_mode: How to process the chunk for classification:
                - 'independent': Process each segment independently
                - 'sequential': Use LSTM to capture sequence information
                - 'attention': Use self-attention to capture context
            log_dir: Optional directory for log files.
            **kwargs: Additional parameters passed to BIOTEncoder.
        """
        super().__init__()
        self.logger = logging.getLogger(__name__ + ".BIOTClassifier")
        self.logger.info(f"Initializing BIOT classifier with emb_size={emb_size}, heads={heads}, "
                         f"depth={depth}, raw={raw}")
        
        self.biot = BIOTEncoder(emb_size=emb_size, heads=heads, depth=depth, raw=raw, log_dir=log_dir, **kwargs)
        self.classifier = ChunkClassificationHead(emb_size=emb_size, chunk_size=chunk_size, mode=classification_mode)

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