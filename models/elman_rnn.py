"""
Elman Recurrent Neural Network
================================
Classic Elman RNN for binary classification of CPS sensor sequences.

Architecture:
  Input (batch, seq_len, input_size)
    → RNN layers with tanh activation
    → Take last hidden state
    → Fully connected layer
    → Sigmoid → probability of attack

This implementation uses nn.Linear layers to manually construct the Elman RNN
instead of nn.RNN for Opacus (Differential Privacy) compatibility.
Opacus does not support nn.RNN directly due to per-sample gradient computation.
"""

import torch
import torch.nn as nn


class ElmanRNNCell(nn.Module):
    """
    Manual Elman RNN cell using Linear layers (Opacus-compatible).
    
    Elman RNN equation:
        h_t = tanh(W_ih @ x_t + b_ih + W_hh @ h_{t-1} + b_hh)
    """

    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        # Input-to-hidden transformation
        self.i2h = nn.Linear(input_size, hidden_size)
        # Hidden-to-hidden transformation (recurrence)
        self.h2h = nn.Linear(hidden_size, hidden_size)
        self.activation = nn.Tanh()

    def forward(self, x, h_prev):
        """
        Args:
            x: input at current time step (batch, input_size)
            h_prev: previous hidden state (batch, hidden_size)
        Returns:
            h_new: new hidden state (batch, hidden_size)
        """
        h_new = self.activation(self.i2h(x) + self.h2h(h_prev))
        return h_new


class ElmanRNN(nn.Module):
    """
    Complete Elman RNN model for binary classification.

    Uses manual RNN cells (Linear layers) for Opacus compatibility.
    The model processes a sequence of sensor readings and outputs
    the probability that the current state represents a cyber attack.
    """

    def __init__(self, input_size, hidden_size=64, num_layers=1, dropout=0.1):
        """
        Args:
            input_size: number of sensor features
            hidden_size: RNN hidden state dimension
            num_layers: number of stacked RNN layers
            dropout: dropout rate between layers
        """
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # Stack of Elman RNN cells
        self.cells = nn.ModuleList()
        for i in range(num_layers):
            cell_input = input_size if i == 0 else hidden_size
            self.cells.append(ElmanRNNCell(cell_input, hidden_size))

        # Dropout between layers
        self.dropout = nn.Dropout(dropout)

        # Output layer: hidden state → binary prediction
        self.fc = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """
        Forward pass through the Elman RNN.

        Args:
            x: input sequence (batch_size, seq_len, input_size)

        Returns:
            output: attack probability (batch_size,)
        """
        batch_size, seq_len, _ = x.shape
        device = x.device

        # Initialize hidden states for all layers
        h = [torch.zeros(batch_size, self.hidden_size, device=device)
             for _ in range(self.num_layers)]

        # Process sequence step by step
        for t in range(seq_len):
            x_t = x[:, t, :]  # (batch, input_size)

            for layer_idx, cell in enumerate(self.cells):
                h[layer_idx] = cell(x_t, h[layer_idx])
                x_t = h[layer_idx]  # output of this layer is input to next

                # Apply dropout between layers (not after last)
                if layer_idx < self.num_layers - 1:
                    x_t = self.dropout(x_t)

        # Use the final hidden state of the last layer
        final_hidden = h[-1]  # (batch, hidden_size)

        # Classification
        logit = self.fc(final_hidden)  # (batch, 1)
        output = self.sigmoid(logit).squeeze(-1)  # (batch,)

        return output

    def get_hidden_size(self):
        return self.hidden_size
