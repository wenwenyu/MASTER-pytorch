# -*- coding: utf-8 -*-
# @Author: Wenwen Yu
# @Created Time: 10/4/2020 17:48


import torch
import torch.nn.functional as F


def subsequent_mask(tgt, padding_symbol):
    """
    tag: (bs, seq_len)
    """
    trg_pad_mask = (tgt != padding_symbol).unsqueeze(1).unsqueeze(3) # (bs, 1, seq_len, 1)
    tgt_len = tgt.size(1) # seq_len,
    trg_sub_mask = torch.tril(torch.ones((tgt_len, tgt_len), dtype=torch.uint8)).to(tgt.device) # (seq_len, seq_len)
    tgt_mask = trg_pad_mask & trg_sub_mask.bool()
    return tgt_mask # (bs, 1, seq_len, seq_len)


def greedy_decode(model, input, max_len, start_symbol, padding_symbol=None, device='cpu', padding=False):
    """
    output predicted transcript
    :param model:
    :param input:
    :param max_len:
    :param start_symbol:
    :param padding_symbol:
    :param device:
    :param padding: if padding is True, max_len will be used. if paddding is False and max_len == -1, max_len will
    be set to 100, otherwise max_len will be used.
    :return:
    """
    B = input.size(0)
    memory = model.encode(input, None)

    if padding:
        if padding_symbol is None:
            raise RuntimeError('Padding Symbol cannot be None.')

        assert max_len > 0

        ys = torch.ones((B, max_len + 2), dtype=torch.long).fill_(padding_symbol).to(device)
        ys[:, 0] = start_symbol
    else:
        if max_len == -1:
            max_len = 100
        ys = torch.ones((B, 1), dtype=torch.long).fill_(start_symbol).to(device)

    # decode with max_len + 1 time step, (include eos）
    for i in range(max_len + 1):
        out = model.decode(memory, None, ys, subsequent_mask(ys, padding_symbol).to(device))
        out = model.generator(out)
        prob = F.softmax(out, dim=-1)
        _, next_word = torch.max(prob, dim=-1)

        if padding:
            ys[:, i + 1] = next_word[:, i]
        else:
            ys = torch.cat([ys, next_word[:, -1].unsqueeze(-1)], dim=1)

    return ys


def greedy_decode_with_probability(model, input, max_len, start_symbol, padding_symbol=None, device='cpu',
                                   padding=False):
    """
        output predicted transcript and corresponding probability
       :param model:
       :param input:
       :param max_len:
       :param start_symbol:
       :param padding_symbol:
       :param device:
       :param padding: if padding is True, max_len will be used. if paddding is False and max_len == -1, max_len will
       be set to 100, otherwise max_len will be used.
       :return:
       """
    B = input.size(0)
    memory = model.encode(input, None)

    if padding:
        if padding_symbol is None:
            raise RuntimeError('Padding Symbol cannot be None.')

        assert max_len > 0

        ys = torch.ones((B, max_len + 2), dtype=torch.long).fill_(padding_symbol).to(device)
        probs = torch.ones((B, max_len + 2), dtype=torch.float).to(device)
        ys[:, 0] = start_symbol
        # probs[:, 0] = 1.0
    else:
        if max_len == -1:
            max_len = 100
        ys = torch.ones((B, 1), dtype=torch.long).fill_(start_symbol).to(device)
        probs = torch.ones((B, 1), dtype=torch.float).to(device)

    # decode with max_len + 1 time step, (include eos）
    for i in range(max_len + 1):
        out = model.decode(memory, None, ys, subsequent_mask(ys, padding_symbol).to(device))
        out = model.generator(out)
        prob = F.softmax(out, dim=-1)
        max_probs, next_word = torch.max(prob, dim=-1)  # (bs, t)

        if padding:
            ys[:, i + 1] = next_word[:, i]
            probs[:, i + 1] = max_probs[:, i]
        else:
            ys = torch.cat([ys, next_word[:, -1].unsqueeze(-1)], dim=1)
            probs = torch.cat([probs, max_probs[:, -1].unsqueeze(-1)], dim=1)

    return ys, probs

# def memory_cache_based_decode(model, input, max_len, start_symbol, padding_symbol=None, device='cpu', padding=False):
#     pass
#     # TODO memory-cache based decode
