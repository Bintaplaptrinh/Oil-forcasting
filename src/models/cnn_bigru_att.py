"""
CNN-BiGRU-Attention — PyTorch port.

PyTorch reimplementation of the TensorFlow `CNN_BIGRU_Attention` graph
(`CNN-BiGRU-Attention.py`). This is the *temporal* port: the convolutions run as
1-D convolutions over the time window (so the GRU and attention model the real
sequence), which matches the original author's own in-code note that the conv
"should be 1-D" (此处应换为一维卷积神经网络). The component sizes are kept:

    Conv1d(D_in -> 6, k=3) -> ReLU -> MaxPool(2)
    Conv1d(6   -> 16, k=3) -> ReLU -> MaxPool(2)
    per-timestep Linear(16 -> fc1=128)            # the TF "conv_fc" dense layer
    BiGRU(128 -> 20, num_layers=2, bidirectional) # TF stacked 2x GRUCell(20), both directions
    additive attention (attn_size=64) over time   # TF `attention(...)`
    Linear(2*20 -> n_class)                        # TF "output" dense layer

Input  : x of shape [B, L, D_in]  (a length-L window of D_in features)
Output : [B, n_class]

Difference from the literal TF graph: the TF code flattened everything through the
dense-128 layer and then fed a length-1 "sequence" to the BiGRU, making the GRU and
attention degenerate (one timestep). Here the time axis is preserved end-to-end.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class AdditiveAttention(nn.Module):
    """Bahdanau-style attention, faithful to the TF `attention()` function.

        v       = tanh(inputs @ W_omega + b_omega)     # [B, T, A]
        scores  = v @ u_omega                           # [B, T]
        alphas  = softmax(scores, dim=time)             # [B, T]
        context = sum_t alphas_t * inputs_t             # [B, H]
    """

    def __init__(self, hidden_size: int, attention_size: int = 64) -> None:
        super().__init__()
        self.W = nn.Linear(hidden_size, attention_size)      # W_omega + b_omega
        self.u = nn.Linear(attention_size, 1, bias=False)    # u_omega

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:  # inputs: [B, T, H]
        v = torch.tanh(self.W(inputs))                        # [B, T, A]
        scores = self.u(v).squeeze(-1)                        # [B, T]
        alphas = torch.softmax(scores, dim=1)                 # [B, T]
        context = torch.bmm(alphas.unsqueeze(1), inputs).squeeze(1)  # [B, H]
        return context


class CNN_BiGRU_Attention(nn.Module):
    def __init__(
        self,
        d_in: int,
        n_class: int,
        seq_len: int | None = None,
        conv1: int = 6,
        conv2: int = 16,
        fc1: int = 128,
        gru_hidden: int = 20,
        gru_layers: int = 2,
        attn_size: int = 64,
    ) -> None:
        super().__init__()
        self.conv1 = nn.Conv1d(d_in, conv1, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(conv1, conv2, kernel_size=3, padding=1)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2, ceil_mode=True)  # SAME-like
        self.fc1 = nn.Linear(conv2, fc1)                                   # per-timestep dense
        self.bigru = nn.GRU(fc1, gru_hidden, num_layers=gru_layers,
                            batch_first=True, bidirectional=True)
        self.attn = AdditiveAttention(2 * gru_hidden, attn_size)
        self.out = nn.Linear(2 * gru_hidden, n_class)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # x: [B, L, d_in]
        x = x.transpose(1, 2)                  # [B, d_in, L]  (channels-first for Conv1d)
        x = self.pool(F.relu(self.conv1(x)))   # [B, 6,  L/2]
        x = self.pool(F.relu(self.conv2(x)))   # [B, 16, L/4]
        x = x.transpose(1, 2)                  # [B, L/4, 16]
        x = self.fc1(x)                        # [B, L/4, 128]  (no activation, as in TF)
        seq, _ = self.bigru(x)                 # [B, L/4, 2*gru_hidden]
        ctx = self.attn(seq)                   # [B, 2*gru_hidden]
        return self.out(ctx)                   # [B, n_class]


# ----------------------------------------------------------------------------------
# Minimal self-test
# ----------------------------------------------------------------------------------
if __name__ == "__main__":
    torch.manual_seed(42)
    B, L, d_in, n_class = 8, 32, 39, 4
    model = CNN_BiGRU_Attention(d_in, n_class, seq_len=L)
    x = torch.randn(B, L, d_in)
    y = model(x)
    assert y.shape == (B, n_class), y.shape
    n_params = sum(p.numel() for p in model.parameters())
    print(f"forward OK: {tuple(x.shape)} -> {tuple(y.shape)} | params: {n_params:,}")
