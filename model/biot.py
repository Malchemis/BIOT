import time
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from linear_attention_transformer import LinearAttentionTransformer


class PatchFrequencyEmbedding(nn.Module):
    def __init__(self, emb_size=256, n_freq=101):
        super().__init__()
        self.projection = nn.Linear(n_freq, emb_size)

    def forward(self, x):
        """
        x: (batch, freq, time)
        out: (batch, time, emb_size)
        """
        x = x.permute(0, 2, 1)
        x = self.projection(x)
        return x


class PatchTimeEmbedding(nn.Module):
    def __init__(self, emb_size=256, patch_size=100, overlap=0.0):
        super().__init__()
        self.patch_size = patch_size
        self.overlap = overlap
        self.projection = nn.Linear(patch_size, emb_size)

    def forward(self, x):
        """
        x: (batch, channel, time)
        out: (batch, n_patches, emb_size)

        With overlap, n_patches will be larger than time // patch_size
        """
        time_steps = x.shape[2]

        # Calculate stride based on overlap
        stride = int(self.patch_size * (1 - self.overlap))
        stride = max(1, stride)  # Ensure stride is at least 1

        # Ensure we have enough time steps for at least one patch
        if time_steps < self.patch_size:
            raise ValueError(f"Input length ({time_steps}) must be >= patch_size ({self.patch_size})")

        # Use unfold with the calculated stride to create overlapping patches
        x = x.squeeze(1)  # Remove channel dim: (batch, time)
        x = x.unfold(1, self.patch_size, stride)  # (batch, n_patches, patch_size)

        # Project to embedding dimension
        x = self.projection(x)  # (batch, n_patches, emb_size)
        return x


class ClassificationHead(nn.Sequential):
    def __init__(self, emb_size, n_classes):
        super().__init__()
        self.clshead = nn.Sequential(
            nn.ELU(),
            nn.Linear(emb_size, n_classes),
        )

    def forward(self, x):
        out = self.clshead(x)
        return out


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 1000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

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

    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        """
        Args:
            x: `embeddings`, shape (batch, max_len, d_model)
        Returns:
            `encoder input`, shape (batch, max_len, d_model)
        """
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


class BIOTEncoder(nn.Module):
    def __init__(
            self,
            emb_size=256,
            heads=8,
            depth=4,
            n_channels=16,
            n_fft=200,
            hop_length=100,
            raw=False,      # Parameter to toggle between spectral and raw data
            patch_size=100, # Patch size for raw data mode
            overlap=0.0,    # Overlap between patches for raw data mode
            **kwargs
    ):
        super().__init__()

        self.n_fft = n_fft
        self.hop_length = hop_length
        self.raw = raw  # Store the raw mode flag

        # Create both embedding modules
        self.patch_frequency_embedding = PatchFrequencyEmbedding(
            emb_size=emb_size, n_freq=self.n_fft // 2 + 1
        )

        # Add the new patch time embedding for raw data
        self.patch_time_embedding = PatchTimeEmbedding(
            emb_size=emb_size, patch_size=patch_size, overlap=overlap
        )

        self.transformer = LinearAttentionTransformer(
            dim=emb_size,
            heads=heads,
            depth=depth,
            max_seq_len=1024,
            attn_layer_dropout=0.2,  # dropout right after self-attention layer
            attn_dropout=0.2,  # dropout post-attention
        )
        self.positional_encoding = PositionalEncoding(emb_size)

        # channel token, N_channels >= your actual channels
        self.channel_tokens = nn.Embedding(n_channels, 256)
        self.index = nn.Parameter(
            torch.LongTensor(range(n_channels)), requires_grad=False
        )

    def stft(self, sample):
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
        """
        x: [batch_size, channel, ts] for example if sampling rate is 200,
        and we take 10s windows we get 2000 time steps per channel and sample of a batch
        output: [batch_size, emb_size]
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
        emb = self.transformer(emb).mean(dim=1)
        return emb


# Modified to pass raw parameter to BIOTEncoder
class BIOTClassifier(nn.Module):
    def __init__(self, emb_size=256, heads=8, depth=4, n_classes=6, raw=False, overlap=0.0, **kwargs):
        super().__init__()
        self.biot = BIOTEncoder(emb_size=emb_size, heads=heads, depth=depth, raw=raw, overlap=overlap, **kwargs)
        self.classifier = ClassificationHead(emb_size, n_classes)

    def forward(self, x):
        x = self.biot(x)
        x = self.classifier(x)
        return x


# Modified to pass raw parameter to BIOTEncoder
class UnsupervisedPretrain(nn.Module):
    def __init__(self, emb_size=256, heads=8, depth=4, n_channels=18, raw=False, **kwargs):
        super(UnsupervisedPretrain, self).__init__()
        self.biot = BIOTEncoder(emb_size, heads, depth, n_channels, raw=raw, **kwargs)
        self.prediction = nn.Sequential(
            nn.Linear(256, 256),
            nn.GELU(),
            nn.Linear(256, 256),
        )

    def forward(self, x, n_channel_offset=0):
        emb = self.biot(x, n_channel_offset, perturb=True)
        emb = self.prediction(emb)
        pred_emb = self.biot(x, n_channel_offset)
        return emb, pred_emb


# Modified to pass raw parameter to BIOTEncoder
class SupervisedPretrain(nn.Module):
    def __init__(self, emb_size=256, heads=8, depth=4, raw=False, **kwargs):
        super().__init__()
        self.biot = BIOTEncoder(emb_size=emb_size, heads=heads, depth=depth, raw=raw, **kwargs)
        self.classifier_chb_mit = ClassificationHead(emb_size, 1)
        self.classifier_iiic_seizure = ClassificationHead(emb_size, 6)
        self.classifier_tuab = ClassificationHead(emb_size, 1)
        self.classifier_tuev = ClassificationHead(emb_size, 6)

    def forward(self, x, task="chb-mit"):
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
            raise NotImplementedError
        return x


if __name__ == "__main__":
    x = torch.randn(16, 2, 2000)

    # Test with spectral processing
    model = BIOTClassifier(n_fft=200, hop_length=100, depth=4, heads=8, raw=False)
    out = model(x)
    print("Spectral output shape:", out.shape)

    # Test with raw data processing (no overlap)
    model_raw = BIOTClassifier(n_fft=200, hop_length=100, depth=4, heads=8, raw=True, patch_size=100, overlap=0.0)
    out_raw = model_raw(x)
    print("Raw output shape (no overlap):", out_raw.shape)

    # Test with raw data processing (50% overlap)
    model_raw_overlap = BIOTClassifier(n_fft=200, hop_length=100, depth=4, heads=8, raw=True, patch_size=100,
                                       overlap=0.5)
    out_raw_overlap = model_raw_overlap(x)
    print("Raw output shape (50% overlap):", out_raw_overlap.shape)

    # Test unsupervised pretraining with raw data and overlap
    model_unsup = UnsupervisedPretrain(n_fft=200, hop_length=100, depth=4, heads=8, raw=True, patch_size=100,
                                       overlap=0.25)
    out1, out2 = model_unsup(x)
    print("Unsupervised raw output shapes:", out1.shape, out2.shape)