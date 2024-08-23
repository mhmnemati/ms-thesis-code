import os
import math
import numpy as np
import torch as pt
import torch.nn as nn
from linear_attention_transformer import LinearAttentionTransformer

import torch.nn as nn


class ConvDenoisingAutoencoder(nn.Module):
    def __init__(self, input_channels=1, hidden_channels=32, kernel_size=3, stride=1, padding=1):
        super(ConvDenoisingAutoencoder, self).__init__()

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv1d(input_channels, hidden_channels, kernel_size, stride, padding),
            nn.ReLU(True),
            nn.Conv1d(hidden_channels, hidden_channels // 2, kernel_size, stride, padding),
            nn.ReLU(True),
            nn.Conv1d(hidden_channels // 2, hidden_channels // 4, kernel_size, stride, padding),
            nn.ReLU(True)
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(hidden_channels // 4, hidden_channels // 2, kernel_size, stride, padding),
            nn.ReLU(True),
            nn.ConvTranspose1d(hidden_channels // 2, hidden_channels, kernel_size, stride, padding),
            nn.ReLU(True),
            nn.ConvTranspose1d(hidden_channels, input_channels, kernel_size, stride, padding),
            nn.Sigmoid()  # Assuming the input data is normalized
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

    def add_noise(self, x, noise_factor=0.3):
        noisy_x = x + noise_factor * pt.randn_like(x)
        noisy_x = pt.clip(noisy_x, 0., 1.)
        return noisy_x


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


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 1000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = pt.zeros(max_len, d_model)
        position = pt.arange(0, max_len).unsqueeze(1).float()
        div_term = pt.exp(
            pt.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = pt.sin(position * div_term)
        pe[:, 1::2] = pt.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x: pt.FloatTensor) -> pt.FloatTensor:
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


class ContextEmbedding(nn.Module):
    def __init__(self, embedding_dim, num_modalities, context_dim):
        super(ContextEmbedding, self).__init__()
        self.modal_embedding = nn.Embedding(num_modalities, embedding_dim)

        # Linear layer to process the context (e.g., X, Y, Z for EEG)
        self.context_net = nn.Sequential(
            nn.Linear(context_dim, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, embedding_dim)
        )

    def forward(self, modality_ids, context):
        """
        modality_ids: Tensor indicating the modality for each channel
        context: Tensor containing context information (e.g., X, Y, Z positions for EEG)
        """
        # Get modal embeddings based on modality IDs
        modal_emb = self.modal_embedding(modality_ids)

        # Generate context-based embeddings
        context_emb = self.context_net(context)

        # Combine modal and context embeddings
        combined_emb = modal_emb + context_emb

        return combined_emb  # Shape: [num_channels, embedding_dim]


class BIOTEncoder(nn.Module):
    def __init__(
        self,
        emb_size=64,
        heads=2,
        depth=4,
        n_channels=16,
        n_modalities=2,  # Add the number of modalities (e.g., EEG, EOG)
        n_fft=200,
        hop_length=100,
        fine_tune=False,
        **kwargs
    ):
        super().__init__()

        # Initialize the DAE
        self.dae = ConvDenoisingAutoencoder(input_channels=1, hidden_channels=32)

        self.n_fft = n_fft
        self.hop_length = hop_length

        self.patch_frequency_embedding = PatchFrequencyEmbedding(emb_size, n_freq=self.n_fft // 2 + 1)
        self.positional_encoding = PositionalEncoding(emb_size)
        self.context_embedding = ContextEmbedding(emb_size, num_modalities=n_modalities, context_dim=3)
        self.transformer = LinearAttentionTransformer(
            dim=emb_size,
            heads=heads,
            depth=depth,
            max_seq_len=1024,
            attn_layer_dropout=0.2,
            attn_dropout=0.2,
        )

        if fine_tune:
            self.load_state_dict(
                pt.load(f"{os.path.dirname(__file__)}/EEG-six-datasets-18-channels.ckpt")
            )

    def stft(self, sample):
        spectral = pt.stft(
            input=sample.squeeze(1),
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            center=False,
            onesided=True,
            return_complex=True,
        )
        return pt.abs(spectral)

    def forward(self, x, modality_ids=None, context=None, perturb=False):
        """
        x: [batch_size, channel, ts]
        modality_ids: [channel], an array that indicates the modality of each channel
        output: [batch_size, emb_size]
        """
        emb_seq = []
        for i in range(x.shape[1]):
            channel_data = x[:, i, :]

            # Add noise and denoise with the ConvDAE
            noisy_channel_data = self.dae.add_noise(channel_data.unsqueeze(1))
            denoised_channel_data = self.dae(noisy_channel_data).squeeze(1)

            channel_spec_emb = self.stft(denoised_channel_data.unsqueeze(1))
            channel_spec_emb = self.patch_frequency_embedding(channel_spec_emb)
            batch_size, ts, _ = channel_spec_emb.shape

            # Unified Context Embedding
            channel_emb = self.context_embedding(modality_ids[i].unsqueeze(0), context[i].unsqueeze(0))

            # Expand the channel embedding across the sequence length
            channel_emb = channel_emb.unsqueeze(1).expand(batch_size, ts, -1)

            # Combine channel embedding with the patch embedding
            channel_emb = channel_spec_emb + channel_emb

            # Positional encoding
            channel_emb = self.positional_encoding(channel_emb)

            # Perturbation (if enabled)
            if perturb:
                ts = channel_emb.shape[1]
                ts_new = np.random.randint(ts // 2, ts)
                selected_ts = np.random.choice(range(ts), ts_new, replace=False)
                channel_emb = channel_emb[:, selected_ts]

            emb_seq.append(channel_emb)

        # Concatenate along the time dimension
        emb = pt.cat(emb_seq, dim=1)

        # Pass through the transformer and average over the sequence
        emb = self.transformer(emb).mean(dim=1)
        return emb
