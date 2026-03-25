"""
Character-Level Name Generation Models (From Scratch)
=======================================================
Implements three recurrent neural network architectures for generating
Indian names character by character:

1. Vanilla RNN - Basic recurrent unit with tanh activation
2. Bidirectional LSTM (BLSTM) - Forward + backward LSTM with projection
3. RNN with Attention - Vanilla RNN enhanced with a basic attention mechanism

All models follow the same interface:
- forward(input_seq, lengths): for training with teacher forcing
- generate(vocab, max_len, temperature): for sampling new names

Architecture overview:
  Embedding -> RNN/LSTM core -> FC output layer -> Softmax over characters
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# ============================================================
# MODEL 1: VANILLA RNN
# ============================================================
class VanillaRNN(nn.Module):
    """
    Vanilla Recurrent Neural Network for character-level name generation.

    Architecture:
    - Character embedding layer: maps each character index to a dense vector
    - Single or multi-layer RNN with tanh activation (standard Elman RNN)
    - Fully connected output layer: maps hidden state to character probabilities

    The RNN processes the input sequence one character at a time, maintaining
    a hidden state h_t that captures information about characters seen so far:
        h_t = tanh(W_ih * x_t + b_ih + W_hh * h_{t-1} + b_hh)

    At each timestep, the output layer projects the hidden state to a
    distribution over the character vocabulary.
    """

    def __init__(self, vocab_size, embed_dim=32, hidden_size=128, num_layers=1, dropout=0.1):
        """
        Args:
            vocab_size: Size of the character vocabulary (including special tokens)
            embed_dim: Dimensionality of character embeddings
            hidden_size: Number of units in the RNN hidden state
            num_layers: Number of stacked RNN layers
            dropout: Dropout probability between RNN layers (only used if num_layers > 1)
        """
        super(VanillaRNN, self).__init__()

        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # Character embedding: converts one-hot indices to dense vectors
        # This learned representation captures character similarities
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)

        # Core RNN: processes embedded characters sequentially
        # Uses tanh activation (standard for vanilla RNN)
        # batch_first=True means input shape is (batch, seq_len, features)
        self.rnn = nn.RNN(
            input_size=embed_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
        )

        # Output projection: maps hidden state to character logits
        # The softmax is applied in the loss function (CrossEntropyLoss)
        self.fc_out = nn.Linear(hidden_size, vocab_size)

        # Dropout for regularization
        self.dropout = nn.Dropout(dropout)

    def forward(self, input_seq, lengths=None):
        """
        Forward pass with teacher forcing (uses ground truth as input).

        Args:
            input_seq: Tensor of character indices [batch_size, seq_len]
            lengths: Tensor of actual sequence lengths [batch_size] (for packing)

        Returns:
            logits: Raw predictions [batch_size, seq_len, vocab_size]
        """
        # Embed input characters: [batch, seq_len] -> [batch, seq_len, embed_dim]
        embedded = self.dropout(self.embedding(input_seq))

        # Pack padded sequences for efficient RNN processing
        # This tells the RNN to skip padding positions
        if lengths is not None:
            embedded = nn.utils.rnn.pack_padded_sequence(
                embedded, lengths.cpu(), batch_first=True, enforce_sorted=True
            )

        # Run through RNN: output contains hidden states for all timesteps
        output, hidden = self.rnn(embedded)

        # Unpack if we packed earlier
        if lengths is not None:
            output, _ = nn.utils.rnn.pad_packed_sequence(output, batch_first=True)

        # Project hidden states to character logits
        logits = self.fc_out(self.dropout(output))
        return logits

    def generate(self, vocab, max_len=20, temperature=0.8):
        """
        Generates a new name by sampling characters one at a time.

        Starting from the SOS token, at each step:
        1. Feed the current character through the model
        2. Sample the next character from the output distribution
        3. Stop if EOS is generated or max_len is reached

        Temperature controls randomness:
        - Low temperature (0.1): more deterministic, common names
        - High temperature (1.5): more random, creative names

        Args:
            vocab: CharVocabulary object
            max_len: Maximum name length before forced stopping
            temperature: Sampling temperature (higher = more random)

        Returns:
            Generated name string
        """
        self.eval()
        with torch.no_grad():
            # Start with SOS token: shape [1, 1] for (batch=1, seq_len=1)
            current_char = torch.tensor([[vocab.sos_idx]], dtype=torch.long)
            hidden = None
            generated = []

            for _ in range(max_len):
                # Embed and run through RNN
                embedded = self.embedding(current_char)  # [1, 1, embed_dim]
                output, hidden = self.rnn(embedded, hidden)
                logits = self.fc_out(output[:, -1, :])  # Take last timestep: [1, vocab]

                # Apply temperature scaling to logits before softmax
                # Higher temperature -> flatter distribution -> more random
                probs = F.softmax(logits / temperature, dim=-1)

                # Sample next character from the probability distribution
                next_char_idx = torch.multinomial(probs, 1).item()

                # Stop if we generated the EOS token
                if next_char_idx == vocab.eos_idx:
                    break

                # Skip PAD and SOS tokens in output
                if next_char_idx not in (vocab.pad_idx, vocab.sos_idx):
                    generated.append(next_char_idx)

                # Use this character as input for the next timestep: [1, 1]
                current_char = torch.tensor([[next_char_idx]], dtype=torch.long)

        self.train()
        return vocab.decode(generated)

    def count_parameters(self):
        """Returns the total number of trainable parameters in the model."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ============================================================
# MODEL 2: BIDIRECTIONAL LSTM (BLSTM)
# ============================================================
class BLSTM(nn.Module):
    """
    Bidirectional LSTM for character-level name generation.

    Architecture:
    - Character embedding layer
    - Bidirectional LSTM: processes sequence in both forward and backward directions
    - Projection layer: combines forward and backward hidden states (2*hidden -> hidden)
    - Output layer: maps projected state to character probabilities

    The bidirectional nature means each output position has context from both
    past AND future characters. This helps learn name patterns where later
    characters influence earlier decisions (e.g., gendered name endings).

    For generation (left-to-right), we use only the forward direction, but
    the training with bidirectional context improves the learned representations.

    LSTM cell equations (per direction):
        f_t = sigmoid(W_f * [h_{t-1}, x_t] + b_f)   # Forget gate
        i_t = sigmoid(W_i * [h_{t-1}, x_t] + b_i)   # Input gate
        c̃_t = tanh(W_c * [h_{t-1}, x_t] + b_c)     # Candidate cell
        c_t = f_t * c_{t-1} + i_t * c̃_t             # Cell state update
        o_t = sigmoid(W_o * [h_{t-1}, x_t] + b_o)   # Output gate
        h_t = o_t * tanh(c_t)                         # Hidden state
    """

    def __init__(self, vocab_size, embed_dim=32, hidden_size=128, num_layers=1, dropout=0.1):
        """
        Args:
            vocab_size: Size of character vocabulary
            embed_dim: Character embedding dimension
            hidden_size: LSTM hidden state size (per direction)
            num_layers: Number of stacked LSTM layers
            dropout: Dropout probability
        """
        super(BLSTM, self).__init__()

        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # Character embedding (same as vanilla RNN)
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)

        # Bidirectional LSTM: doubles the output dimension since it concatenates
        # forward and backward hidden states
        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,  # Key difference: processes both directions
            dropout=dropout if num_layers > 1 else 0,
        )

        # Projection layer: reduces 2*hidden_size (bidirectional) back to hidden_size
        # This is necessary because bidirectional LSTM outputs are concatenated
        self.projection = nn.Linear(hidden_size * 2, hidden_size)

        # Output layer: maps to vocabulary distribution
        self.fc_out = nn.Linear(hidden_size, vocab_size)

        self.dropout = nn.Dropout(dropout)

        # Separate unidirectional LSTM for generation (can't use bidirectional
        # during autoregressive generation since future characters don't exist yet)
        # This is jointly trained: its output also contributes to the loss
        self.gen_lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
        )
        # Generation output uses same hidden_size (no bidirectional doubling)
        self.gen_fc_out = nn.Linear(hidden_size, vocab_size)

    def forward(self, input_seq, lengths=None):
        """
        Forward pass using bidirectional LSTM + jointly trained unidirectional LSTM.

        We train both the bidirectional LSTM (for better contextual representations)
        and a unidirectional LSTM (for generation). The final logits are the average
        of both paths, so the unidirectional gen_lstm learns alongside the BLSTM.

        Args:
            input_seq: [batch_size, seq_len] character indices
            lengths: [batch_size] actual sequence lengths

        Returns:
            logits: [batch_size, seq_len, vocab_size]
        """
        embedded = self.dropout(self.embedding(input_seq))

        if lengths is not None:
            packed = nn.utils.rnn.pack_padded_sequence(
                embedded, lengths.cpu(), batch_first=True, enforce_sorted=True
            )
        else:
            packed = embedded

        # BLSTM path: bidirectional output [batch, seq_len, 2*hidden]
        blstm_out, _ = self.lstm(packed)
        # Gen LSTM path: unidirectional output [batch, seq_len, hidden]
        gen_out, _ = self.gen_lstm(packed)

        if lengths is not None:
            blstm_out, _ = nn.utils.rnn.pad_packed_sequence(blstm_out, batch_first=True)
            gen_out, _ = nn.utils.rnn.pad_packed_sequence(gen_out, batch_first=True)

        # Bidirectional path -> projection -> logits
        projected = torch.tanh(self.projection(self.dropout(blstm_out)))
        blstm_logits = self.fc_out(self.dropout(projected))

        # Unidirectional generation path -> logits
        gen_logits = self.gen_fc_out(self.dropout(gen_out))

        # Combined loss: average of both paths ensures gen_lstm is trained
        logits = 0.5 * blstm_logits + 0.5 * gen_logits
        return logits

    def generate(self, vocab, max_len=20, temperature=0.8):
        """
        Generates a name using the unidirectional generation LSTM.

        During generation we can't use bidirectional processing (no future context),
        so we use a separate forward-only LSTM (gen_lstm) that was trained
        alongside the main BLSTM via shared embeddings.

        Args:
            vocab: CharVocabulary object
            max_len: Maximum name length
            temperature: Sampling temperature

        Returns:
            Generated name string
        """
        self.eval()
        with torch.no_grad():
            current_char = torch.tensor([[vocab.sos_idx]], dtype=torch.long)
            hidden = None
            generated = []

            for _ in range(max_len):
                embedded = self.embedding(current_char)  # [1, 1, embed_dim]
                output, hidden = self.gen_lstm(embedded, hidden)
                logits = self.gen_fc_out(output[:, -1, :])  # [1, vocab]

                probs = F.softmax(logits / temperature, dim=-1)
                next_char_idx = torch.multinomial(probs, 1).item()

                if next_char_idx == vocab.eos_idx:
                    break
                if next_char_idx not in (vocab.pad_idx, vocab.sos_idx):
                    generated.append(next_char_idx)

                current_char = torch.tensor([[next_char_idx]], dtype=torch.long)

        self.train()
        return vocab.decode(generated)

    def count_parameters(self):
        """Returns total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ============================================================
# MODEL 3: RNN WITH BASIC ATTENTION MECHANISM
# ============================================================
class AttentionRNN(nn.Module):
    """
    RNN with a basic (self) attention mechanism for character-level generation.

    Architecture:
    - Character embedding layer
    - LSTM encoder: processes the sequence and stores all hidden states
    - Attention layer: computes weighted sum of all previous hidden states
    - Concat + projection: combines attention context with current hidden state
    - Output layer: maps to character probabilities

    The attention mechanism allows the model to "look back" at all previously
    generated characters when deciding the next one, rather than relying solely
    on the compressed hidden state. This is particularly useful for names where
    early characters constrain later ones (e.g., "Shri..." likely ends differently
    than "Pra...").

    Attention computation (Bahdanau-style additive attention):
        score(h_t, h_s) = v^T * tanh(W_1 * h_t + W_2 * h_s)
        alpha = softmax(scores)
        context = sum(alpha * h_s)
    """

    def __init__(self, vocab_size, embed_dim=32, hidden_size=128, num_layers=1, dropout=0.1):
        """
        Args:
            vocab_size: Size of character vocabulary
            embed_dim: Character embedding dimension
            hidden_size: RNN hidden state dimension
            num_layers: Number of stacked LSTM layers
            dropout: Dropout probability
        """
        super(AttentionRNN, self).__init__()

        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # Character embedding
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)

        # LSTM encoder (unidirectional for autoregressive generation)
        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
        )

        # --- Attention layers (Bahdanau/additive style) ---
        # W_1: projects current hidden state (query)
        self.attn_query = nn.Linear(hidden_size, hidden_size, bias=False)
        # W_2: projects all previous hidden states (keys)
        self.attn_key = nn.Linear(hidden_size, hidden_size, bias=False)
        # v: computes scalar attention score from the combined projection
        self.attn_v = nn.Linear(hidden_size, 1, bias=False)

        # Combines attention context + current hidden state before output
        # Input is hidden_size (attention context) + hidden_size (current state) = 2*hidden
        self.attn_combine = nn.Linear(hidden_size * 2, hidden_size)

        # Output layer
        self.fc_out = nn.Linear(hidden_size, vocab_size)

        self.dropout = nn.Dropout(dropout)

    def _compute_attention(self, current_hidden, all_hiddens, mask=None):
        """
        Computes attention weights and context vector.

        Uses additive (Bahdanau) attention:
        1. Project current hidden state and all previous hidden states
        2. Combine with tanh and project to scalar scores
        3. Apply softmax to get attention weights
        4. Compute weighted sum of hidden states as context

        Args:
            current_hidden: Current timestep hidden state [batch, hidden_size]
            all_hiddens: All hidden states so far [batch, t, hidden_size]
            mask: Optional mask for padding positions [batch, t]

        Returns:
            context: Attention-weighted context vector [batch, hidden_size]
            attn_weights: Attention weight distribution [batch, t]
        """
        # Project query (current state): [batch, 1, hidden]
        query = self.attn_query(current_hidden).unsqueeze(1)

        # Project keys (all previous states): [batch, t, hidden]
        keys = self.attn_key(all_hiddens)

        # Compute attention scores: [batch, t, 1] -> [batch, t]
        # tanh(W1*h_t + W2*h_s) captures interaction between current and past
        scores = self.attn_v(torch.tanh(query + keys)).squeeze(-1)

        # Mask padding positions with large negative value before softmax
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))

        # Softmax normalizes scores to a probability distribution
        attn_weights = F.softmax(scores, dim=-1)

        # Compute context as weighted sum of hidden states
        # [batch, 1, t] @ [batch, t, hidden] -> [batch, 1, hidden] -> [batch, hidden]
        context = torch.bmm(attn_weights.unsqueeze(1), all_hiddens).squeeze(1)

        return context, attn_weights

    def forward(self, input_seq, lengths=None):
        """
        Forward pass with attention over all previous positions.

        For each position t, the model:
        1. Runs the LSTM to get h_t
        2. Attends over h_1...h_t (all states up to current)
        3. Combines attention context with h_t
        4. Projects to character logits

        Args:
            input_seq: [batch_size, seq_len] character indices
            lengths: [batch_size] actual lengths

        Returns:
            logits: [batch_size, seq_len, vocab_size]
        """
        batch_size, seq_len = input_seq.size()

        embedded = self.dropout(self.embedding(input_seq))

        # Run full sequence through LSTM first to get all hidden states
        if lengths is not None:
            packed = nn.utils.rnn.pack_padded_sequence(
                embedded, lengths.cpu(), batch_first=True, enforce_sorted=True
            )
            lstm_out, _ = self.lstm(packed)
            lstm_out, _ = nn.utils.rnn.pad_packed_sequence(lstm_out, batch_first=True)
        else:
            lstm_out, _ = self.lstm(embedded)

        # Create mask for valid (non-padding) positions
        if lengths is not None:
            mask = torch.arange(seq_len).unsqueeze(0).expand(batch_size, -1)
            mask = (mask < lengths.unsqueeze(1)).float()
        else:
            mask = torch.ones(batch_size, seq_len)

        # Apply attention at each timestep
        outputs = []
        for t in range(seq_len):
            # Current hidden state at position t
            current_h = lstm_out[:, t, :]  # [batch, hidden]

            # Attend over all positions up to and including t
            # (causal attention - can only look at past and present)
            context_states = lstm_out[:, :t+1, :]  # [batch, t+1, hidden]
            context_mask = mask[:, :t+1]  # [batch, t+1]

            # Compute attention context
            context, _ = self._compute_attention(current_h, context_states, context_mask)

            # Combine attention context with current hidden state
            combined = torch.cat([current_h, context], dim=-1)  # [batch, 2*hidden]
            combined = torch.tanh(self.attn_combine(combined))  # [batch, hidden]

            outputs.append(combined)

        # Stack all timestep outputs: [batch, seq_len, hidden]
        output = torch.stack(outputs, dim=1)

        # Project to character logits
        logits = self.fc_out(self.dropout(output))
        return logits

    def generate(self, vocab, max_len=20, temperature=0.8):
        """
        Generates a name using the attention mechanism.

        At each step, the model attends over all previously generated
        character representations to decide the next character.

        Args:
            vocab: CharVocabulary object
            max_len: Maximum name length
            temperature: Sampling temperature

        Returns:
            Generated name string
        """
        self.eval()
        with torch.no_grad():
            current_char = torch.tensor([[vocab.sos_idx]], dtype=torch.long)
            hidden = None
            generated = []
            all_hiddens = []  # Store all hidden states for attention

            for step in range(max_len):
                embedded = self.embedding(current_char)  # [1, 1, embed_dim]
                lstm_out, hidden = self.lstm(embedded, hidden)

                # Current hidden state: [1, hidden]
                current_h = lstm_out[:, -1, :]
                all_hiddens.append(current_h)

                # Stack all hidden states collected so far: [1, step+1, hidden]
                stacked = torch.stack(all_hiddens, dim=1)

                # Compute attention over all previous hidden states
                context, _ = self._compute_attention(current_h, stacked)

                # Combine and project
                combined = torch.cat([current_h, context], dim=-1)
                combined = torch.tanh(self.attn_combine(combined))
                logits = self.fc_out(combined)

                # Sample next character
                probs = F.softmax(logits / temperature, dim=-1)
                next_char_idx = torch.multinomial(probs, 1).item()

                if next_char_idx == vocab.eos_idx:
                    break
                if next_char_idx not in (vocab.pad_idx, vocab.sos_idx):
                    generated.append(next_char_idx)

                current_char = torch.tensor([[next_char_idx]], dtype=torch.long)

        self.train()
        return vocab.decode(generated)

    def count_parameters(self):
        """Returns total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
