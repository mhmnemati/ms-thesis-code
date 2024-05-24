import os
import math
import numpy as np
import scipy as sp
import torch as pt
import torch.nn as nn

from .base import BaseModel
from torch.utils.data import DataLoader
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


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 1000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space.
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
        **kwargs
    ):
        super().__init__()

        self.n_fft = n_fft
        self.hop_length = hop_length

        self.patch_embedding = PatchFrequencyEmbedding(
            emb_size=emb_size, n_freq=self.n_fft // 2 + 1
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
            pt.LongTensor(range(n_channels)), requires_grad=False
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

    def forward(self, x, n_channel_offset=0, perturb=False):
        """
        x: [batch_size, channel, ts]
        output: [batch_size, emb_size]
        """
        emb_seq = []
        for i in range(x.shape[1]):
            channel_spec_emb = self.stft(x[:, i: i + 1, :])
            channel_spec_emb = self.patch_embedding(channel_spec_emb)
            batch_size, ts, _ = channel_spec_emb.shape
            # (batch_size, ts, emb)
            channel_token_emb = (
                self.channel_tokens(self.index[i + n_channel_offset])
                .unsqueeze(0)
                .unsqueeze(0)
                .repeat(batch_size, ts, 1)
            )
            # (batch_size, ts, emb)
            channel_emb = self.positional_encoding(channel_spec_emb + channel_token_emb)

            # perturb
            if perturb:
                ts = channel_emb.shape[1]
                ts_new = np.random.randint(ts // 2, ts)
                selected_ts = np.random.choice(range(ts), ts_new, replace=False)
                channel_emb = channel_emb[:, selected_ts]
            emb_seq.append(channel_emb)

        # (batch_size, 16 * ts, emb)
        emb = pt.cat(emb_seq, dim=1)
        # (batch_size, emb)
        emb = self.transformer(emb).mean(dim=1)
        return emb


class BIOTModel(nn.Module):
    def __init__(self, n_outputs):
        super().__init__()

        self.model = BIOTEncoder(
            emb_size=256,
            heads=8,
            depth=4,
            n_classes=5,
            n_fft=200,
            hop_length=100,
            n_channels=18,
        )
        self.model.load_state_dict(
            pt.load(f"{os.path.dirname(__file__)}/EEG-six-datasets-18-channels.ckpt")
        )
        self.classifier = nn.Sequential(
            nn.Linear(in_features=256, out_features=n_outputs),
        )

    def forward(self, x):
        x = self.model(x)
        x = self.classifier(x)
        return x


class BIOT(BaseModel):
    data_loader = DataLoader

    def __init__(self, **kwargs):
        hparams = {k: v for k, v in kwargs.items() if k in ["n_outputs"]}
        super().__init__(
            num_classes=hparams["n_outputs"],
            hparams=hparams,
            model=BIOTModel(**hparams),
            loss=pt.nn.CrossEntropyLoss()
        )

    def transform(self, item):
        for i in range(item["data"].shape[0]):
            percentile_95 = np.percentile(np.abs(item["data"][i]), 95, axis=0, keepdims=True)
            item["data"][i] = item["data"][i] / percentile_95

        channels = [
            "FP1-F7",
            "F7-T7",
            "T7-P7",
            "P7-O1",
            "FP2-F8",
            "F8-T8",
            "T8-P8",
            "P8-O2",
            "FP1-F3",
            "F3-C3",
            "C3-P3",
            "P3-O1",
            "FP2-F4",
            "F4-C4",
            "C4-P4",
            "P4-O2",
            "C3-A2",
            "C4-A1",
        ]

        data = np.zeros((len(channels), 30 * 200), dtype=np.float32)
        for idx, ch_name in enumerate(item["ch_names"]):
            if ch_name == "EEG Fpz-Cz":
                signal = sp.signal.resample(item["data"][idx], 30 * 200) / 2
                data[channels.index("FP1-F3")] = signal
                data[channels.index("F3-C3")] = signal
                data[channels.index("FP2-F4")] = signal
                data[channels.index("F4-C4")] = signal

            if ch_name == "EEG Pz-Oz":
                signal = sp.signal.resample(item["data"][idx], 30 * 200)
                data[channels.index("P3-O1")] = signal
                data[channels.index("P4-O2")] = signal

        return (data, item["labels"].max())

    @staticmethod
    def add_arguments(parent_parser):
        parser = parent_parser.add_argument_group("Brain2Seq")
        parser.add_argument("--n_outputs", type=int, default=5)
        return parent_parser
