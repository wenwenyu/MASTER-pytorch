# -*- coding: utf-8 -*-
# @Author: Wenwen Yu
# @Created Time: 10/4/2020 17:48


import torch
import torch.nn.functional as F
from model.master import predict


# def subsequent_mask(tgt, padding_symbol):
#     """
#     tag: (bs, seq_len)
#     """
#     trg_pad_mask = (tgt != padding_symbol).unsqueeze(1).unsqueeze(3)  # (bs, 1, seq_len, 1)
#     tgt_len = tgt.size(1)  # seq_len,
#     trg_sub_mask = torch.tril(torch.ones((tgt_len, tgt_len), dtype=torch.uint8)).to(tgt.device)  # (seq_len, seq_len)
#     tgt_mask = trg_pad_mask & trg_sub_mask.bool()
#     return tgt_mask  # (bs, 1, seq_len, seq_len)


def greedy_decode_with_probability(_model,
                                   _input_tensor,
                                   _max_sequence_length,
                                   _start_symbol_index,
                                   _end_symbol_index,
                                   _padding_symbol_index=None,
                                   _result_device='cpu',
                                   _is_padding=False):
    """
        output predicted transcript and corresponding probability
       :param _model:   master model
       :param _input_tensor:    to predict tensor
       :param _max_sequence_length: max sequence length to predict
       :param _start_symbol_index:  sos label index
       :param _end_symbol_index:    eos label index
       :param _padding_symbol_index:    padding label index
       :param _result_device:   target device(not used)
       :param _is_padding: if padding is True, max_len will be used. if paddding is False and max_len == -1, max_len will
       be set to 100, otherwise max_len will be used.
       :return: label,label probability
       """
    memory = _model.encode_stage(_input_tensor)
    predicted_label, predicted_label_probability = predict(
        memory,
        _input_tensor,
        _model.decode_stage,
        _max_sequence_length,
        _start_symbol_index,
        _end_symbol_index,
        _padding_symbol_index
    )

    return predicted_label, predicted_label_probability
