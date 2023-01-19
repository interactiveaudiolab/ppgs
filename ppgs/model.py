"""model.py - model definition"""

import math
import functools
import torch
import ppgs


###############################################################################
# Model
###############################################################################


class BaselineModel(torch.nn.Sequential):
    """Create a baseline model to compare with. Basedon on torch Sequential superclass."""

    def __init__(
        self,
        input_channels=None, #dimensionality of input time series
        output_channels=None, #phoneme time series dimensionality
        hidden_channels=128,
        kernel_size=5):

        if input_channels is None:
            input_channels = ppgs.INPUT_CHANNELS
        if output_channels is None:
            output_channels = ppgs.OUTPUT_CHANNELS

        conv_fn = functools.partial(
            torch.nn.Conv1d,
            kernel_size=kernel_size,
            padding='same')
        super().__init__(
            conv_fn(input_channels, hidden_channels),
            torch.nn.ReLU(),
            conv_fn(hidden_channels, hidden_channels),
            torch.nn.ReLU(),
            conv_fn(hidden_channels, output_channels))


class MultiHeadAttention(torch.nn.Module):

    def __init__(
        self,
        channels,
        out_channels,
        n_heads,
        p_dropout=0.,
        window_size=4):
        super().__init__()
        assert channels % n_heads == 0
        # Setup layers
        self.n_heads = n_heads
        self.window_size = window_size
        self.conv_q = torch.nn.Conv1d(channels, channels, 1)
        self.conv_k = torch.nn.Conv1d(channels, channels, 1)
        self.conv_v = torch.nn.Conv1d(channels, channels, 1)
        self.conv_o = torch.nn.Conv1d(channels, out_channels, 1)
        self.dropout = torch.nn.Dropout(p_dropout)

        # Setup relative positional embedding
        self.k_channels = channels // n_heads
        rel_stddev = self.k_channels ** -0.5
        self.emb_rel_k = torch.nn.Parameter(
            rel_stddev * torch.randn(
                1,
                window_size * 2 + 1,
                self.k_channels))
        self.emb_rel_v = torch.nn.Parameter(
            rel_stddev * torch.randn(
                1,
                window_size * 2 + 1,
                self.k_channels))

        # Initialize weights
        torch.nn.init.xavier_uniform_(self.conv_q.weight)
        torch.nn.init.xavier_uniform_(self.conv_k.weight)
        torch.nn.init.xavier_uniform_(self.conv_v.weight)

    def forward(self, x, c, mask=None):
        q = self.conv_q(x)
        k = self.conv_k(c)
        v = self.conv_v(c)
        x, self.attn = self.attention(q, k, v, mask=mask)
        return self.conv_o(x)

    def attention(self, query, key, value, mask=None):
        # Reshape (batch, channels, time) ->
        #         (batch, heads, time, channels // heads)
        batch, channels, t_s, t_t = (*key.size(), query.size(2))
        query = query.view(
            batch,
            self.n_heads,
            self.k_channels,
            t_t).transpose(2, 3)
        key = key.view(
            batch,
            self.n_heads,
            self.k_channels,
            t_s).transpose(2, 3)
        value = value.view(
            batch,
            self.n_heads,
            self.k_channels,
            t_s).transpose(2, 3)

        # Compute attention matrix
        scores = torch.matmul(
            query / math.sqrt(self.k_channels),
            key.transpose(-2, -1))

        # Relative positional representation
        relative_embeddings = self.relative_embeddings(self.emb_rel_k, t_s)
        relative_logits = torch.matmul(
            query / math.sqrt(self.k_channels),
            relative_embeddings.unsqueeze(0).transpose(-2, -1))
        scores += self.relative_to_absolute(relative_logits)

        # Apply sequence mask
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e4)

        # Compute output activation
        # (batch, heads, t_t, t_s)
        attention = self.dropout(torch.nn.functional.softmax(scores, dim=-1))
        output = torch.matmul(attention, value)

        # Convert to absolute positional representation to adjust output
        output += torch.matmul(
            self.absolute_to_relative(attention),
            self.relative_embeddings(self.emb_rel_v, t_s).unsqueeze(0))

        # Reshape (batch, heads, time, channels // heads) ->
        #         (batch, channels, time]
        output = output.transpose(2, 3).contiguous().view(batch, channels, t_t)

        return output, attention

    def relative_embeddings(self, embedding, length):
        # Pad first before slice to avoid using cond ops
        pad_length = max(length - (self.window_size + 1), 0)
        start = max((self.window_size + 1) - length, 0)
        end = start + 2 * length - 1
        if pad_length > 0:
            padded_embedding = torch.nn.functional.pad(
                embedding,
                (0, 0, pad_length, pad_length))
        else:
            padded_embedding = embedding
        return padded_embedding[:, start:end]

    def relative_to_absolute(self, x):
        """
        x: [b, h, l, 2*l-1]
        ret: [b, h, l, l]
        """
        batch, heads, length, _ = x.size()

        # Concat columns of pad to shift from relative to absolute indexing
        x = torch.nn.functional.pad(x, (0, 1))

        # Concat extra elements so to add up to shape (len + 1, 2 * len - 1)
        x_flat = x.view([batch, heads, 2 * length * length])
        x_flat = torch.nn.functional.pad(x_flat, (0, length - 1))

        # Reshape and slice out the padded elements.
        shape = (batch, heads, length + 1, 2 * length - 1)
        return x_flat.view(shape)[:, :, :length, length - 1:]

    def absolute_to_relative(self, x):
        """
        x: [b, h, l, l]
        ret: [b, h, l, 2*l-1]
        """
        batch, heads, length, _ = x.size()

        # Pad along column
        x = torch.nn.functional.pad(x, (0, length - 1))
        x_flat = x.view([batch, heads, length ** 2 + length * (length - 1)])

        # Add 0's in the beginning that will skew the elements after reshape
        x_flat = torch.nn.functional.pad(x_flat, (length, 0))
        return x_flat.view([batch, heads, length, 2 * length])[:, :, :, 1:]