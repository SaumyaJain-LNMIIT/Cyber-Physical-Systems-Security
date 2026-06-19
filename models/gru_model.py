"""
GRU Model (Optional Alternative)
==================================
GRU variant for CPS intrusion detection. Handles vanishing gradients
better than vanilla Elman RNN with similar computational cost.

Uses Linear layers for Opacus compatibility.
"""

import torch
import torch.nn as nn


class GRUCell(nn.Module):
    """Manual GRU cell using Linear layers (Opacus-compatible)."""

    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size

        # Reset gate
        self.W_ir = nn.Linear(input_size, hidden_size)
        self.W_hr = nn.Linear(hidden_size, hidden_size)

        # Update gate
        self.W_iz = nn.Linear(input_size, hidden_size)
        self.W_hz = nn.Linear(hidden_size, hidden_size)

        # New gate (candidate hidden state)
        self.W_in = nn.Linear(input_size, hidden_size)
        self.W_hn = nn.Linear(hidden_size, hidden_size)

    def forward(self, x, h_prev):
        r = torch.sigmoid(self.W_ir(x) + self.W_hr(h_prev))  # reset gate
        z = torch.sigmoid(self.W_iz(x) + self.W_hz(h_prev))  # update gate
        n = torch.tanh(self.W_in(x) + r * self.W_hn(h_prev))  # candidate
        h_new = (1 - z) * n + z * h_prev
        return h_new


class GRUModel(nn.Module):
    """GRU model for binary classification of CPS sensor sequences."""

    def __init__(self, input_size, hidden_size=64, num_layers=1, dropout=0.1):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.cells = nn.ModuleList()
        for i in range(num_layers):
            cell_input = input_size if i == 0 else hidden_size
            self.cells.append(GRUCell(cell_input, hidden_size))

        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        device = x.device

        h = [torch.zeros(batch_size, self.hidden_size, device=device)
             for _ in range(self.num_layers)]

        for t in range(seq_len):
            x_t = x[:, t, :]
            for layer_idx, cell in enumerate(self.cells):
                h[layer_idx] = cell(x_t, h[layer_idx])
                x_t = h[layer_idx]
                if layer_idx < self.num_layers - 1:
                    x_t = self.dropout(x_t)

        final_hidden = h[-1]
        logit = self.fc(final_hidden)
        output = self.sigmoid(logit).squeeze(-1)
        return output

    def get_hidden_size(self):
        return self.hidden_size
