# -*- coding: utf-8 -*-
# @Author: Wenwen Yu
# @Email: yuwenwen62@gmail.com
# @Created Time: 10/3/2020 1:27 PM

import math
import copy

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class Encoder(nn.Module):
    "Core encoder is a stack of N layers"

    def __init__(self, layer, N):
        """

        :param layer: single encoder layer
        :param N: number of laybers
        """
        super(Encoder, self).__init__()

        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, *input):
        "Pass the input (and mask) through each layer in turn."

        x = input[0]
        mask = input[1]
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


class LayerNorm(nn.Module):
    "Construct a layernorm module"

    def __init__(self, feature_shape, eps=1e-6):
        """

        :param feature_shape:
        :param eps:
        """
        super(LayerNorm, self).__init__()

        # self.a_2 = nn.Parameter(torch.ones(feature_shape))
        # self.b_2 = nn.Parameter(torch.zeros(feature_shape))
        # self.eps = eps
        self._norm = torch.nn.LayerNorm(feature_shape, eps=eps)

    def forward(self, *input):
        x = input[0]
        # mean = x.mean(-1, keepdim=True)
        # std = x.std(-1, keepdim=True)

        # return self.a_2 * (x - mean) / (std + self.eps) + self.b_2
        return self._norm(x)


class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """

    def __init__(self, size, dropout):
        """

        :param size:
        :param dropout:
        """
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, *input):
        x = input[0]
        sublayer = input[1]
        "Apply residual connection to any sublayer with the same size."
        return x + self.dropout(sublayer(self.norm(x)))


class EncoderLayer(nn.Module):
    "Encoder is made up of self-attn and feed forward (defined below)"

    def __init__(self, size, self_attn, feed_forward, dropout):
        """

        :param size:
        :param self_attn:
        :param feed_forward:
        :param dropput:
        """
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size

    def forward(self, *input):
        x = input[0]
        mask = input[1]
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayer[1](x, self.feed_forward)


class Decoder(nn.Module):
    "Generic N layer decoder with masking."

    def __init__(self, layer, N):
        super(Decoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, *input):
        x = input[0]
        memory = input[1]
        src_mask = input[2]
        tgt_mask = input[3]

        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return self.norm(x)


class DecoderLayer(nn.Module):
    "Decoder is made of self-attn, src-attn, and feed forward (defined below)"

    def __init__(self, size, self_attn, src_attn, feed_forward, dropout):
        """

        :param size:
        :param self_attn: Masked Multi-Head Attention
        :param src_attn: Multi-Head Attention
        :param feed_forward:
        :param dropout:
        """
        super(DecoderLayer, self).__init__()
        self.size = size
        self.self_attn = self_attn
        self.src_attn = src_attn  # cross attention
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 3)

    def forward(self, *input):
        x = input[0]
        memory = input[1]  # from encoder as key and value, x as query for src_attention(multi-head attention)
        src_mask = input[2]
        tgt_mask = input[3]

        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))
        x = self.sublayer[1](x, lambda x: self.src_attn(x, memory, memory, src_mask))
        return self.sublayer[2](x, self.feed_forward)


def subsequent_mask(size):
    "mask out subsequent position"

    attn_shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return torch.from_numpy(subsequent_mask) == 0


def dot_product_attention(query, key, value, mask=None, dropout=None):
    """
    Compute 'Scaled Dot Product Attention

    :param query: (N, h, seq_len, d_q), h is multi-head
    :param key: (N, h, seq_len, d_k)
    :param value: (N, h, seq_len, d_v)
    :param mask: None or (N, 1, seq_len, seq_len), 0 will be replaced with -1e9
    :param dropout:
    :return:
    """

    d_k = value.size(-1)
    score = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)  # (N, h, seq_len, seq_len)
    if mask is not None:
        score = score.masked_fill(mask == 0, -1e9)  # score (N, h, seq_len, seq_len)

    p_attn = F.softmax(score, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    # (N, h, seq_len, d_v), (N, h, seq_len, seq_len)
    return torch.matmul(p_attn, value), p_attn


class MultiHeadAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        """

        :param h: number of self attention head
        :param d_model: dimension of model
        :param dropout:
        """
        super(MultiHeadAttention, self).__init__()

        assert d_model % h == 0
        # requires d_v = d_k, d_q = d_k = d_v = d_m / h
        self.d_k = int(d_model / h)
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)  # (q, k, v, last output layer)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, *input):
        query = input[0]  # (N, seq_len, d_m)
        key = input[1]  # (N, seq_len, d_m)
        value = input[2]  # (N, seq_len, d_m)
        # None or (N, 1, seq_len, seq_len), attention score will be masked with -1e9 where mask==False
        mask = input[3]

        # if mask is not None:
        #     mask = mask.unsqueeze(1)

        nbatches = query.size(0)

        # do all the linear projections in batch from d_model => h x d_k
        # (N, seq_len, d_m) -> (N, seq_len, h, d_k) -> (N, h, seq_len, d_k)
        query, key, value = \
            [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
             for l, x in zip(self.linears, (query, key, value))]

        # apply attention on all the projected vectors in batch.
        # (N, h, seq_len, d_v), (N, h, seq_len, seq_len)
        x, self.attn = dot_product_attention(query, key, value, mask=mask,
                                             dropout=self.dropout)

        # "Concat" using a view and apply a final linear.
        # (N, seq_len, d_m)
        x = x.transpose(1, 2).contiguous() \
            .view(nbatches, -1, self.h * self.d_k)

        # (N, seq_len, d_m)
        return self.linears[-1](x)


class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        """

        :param d_model:
        :param d_ff:
        :param dropout:
        """
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, *input):
        x = input[0]
        return self.w_2(self.dropout(F.relu(self.w_1(x))))


class Embeddings(nn.Module):
    def __init__(self, d_model, vocab):
        """

        :param d_model:
        :param vocab:
        """
        super(Embeddings, self).__init__()

        self.lut = nn.Embedding(vocab, d_model)
        self.d_model = d_model

    def forward(self, *input):
        x = input[0]
        return self.lut(x) * math.sqrt(self.d_model)


class PositionalEncoding(nn.Module):
    "Implement the PE function."

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        """

        :param d_model:
        :param dropout:
        :param max_len:
        """
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, *input):
        x = input[0]
        x = x + self.pe[:, :x.size(1)]  # pe 1 5000 512
        return self.dropout(x)
