'''
Adapted from /home/dozhang/Language-Specific-Neurons/hf_llama/ori_model.py
'''

import math
import argparse
import json
import os
import numpy as np
import logging
from tqdm import tqdm
import matplotlib.pyplot as plt

import torch
import subprocess
import signal
import torch.nn.functional as F
from fairscale.nn.model_parallel.mappings import reduce_from_model_parallel_region
from types import MethodType
from typing import ClassVar, List, Optional, Sequence, Union, cast, overload

from ori_model import apply_rotary_emb, repeat_kv


def ori_adapt(model):
    #import pdb; pdb.set_trace()
    max_length = model.model.params.max_seq_len
    num_layers = model.model.params.n_layers
    hidden_dim = int(2 * model.model.params.dim / 3)
    if model.model.params.ffn_dim_multiplier is not None:
        hidden_dim = int(model.model.params.ffn_dim_multiplier * hidden_dim)
    intermediate_size = model.model.params.multiple_of * ((hidden_dim + model.model.params.multiple_of - 1) // model.model.params.multiple_of)
    
    sum1 = torch.zeros(num_layers, intermediate_size).to('cuda')
    sum2 = torch.zeros(num_layers, intermediate_size).to('cuda')
    sum3 = torch.zeros(num_layers, intermediate_size).to('cuda')
    sum4 = torch.zeros(num_layers, intermediate_size).to('cuda')
    over_zero = torch.zeros(num_layers, intermediate_size, dtype=torch.int32).to('cuda')
    flat_zero = torch.zeros(num_layers, 2048, intermediate_size).to('cuda')
    activation_mask = torch.zeros(num_layers, intermediate_size, dtype=torch.int32).to('cuda')

    def Emb_factory():
        def Emb_forward(self, input_: torch.Tensor) -> torch.Tensor:  # type: ignore
            #import pdb; pdb.set_trace()
            # Build the mask.
            input_mask = (input_ < self.vocab_start_index) | (input_ >= self.vocab_end_index)
            # Mask the input.
            masked_input = input_.clone() - self.vocab_start_index
            masked_input[input_mask] = 0
            # Get the embeddings.
            output_parallel = F.embedding(
                masked_input,
                self.weight,
                self.padding_idx,
                self.max_norm,
                self.norm_type,
                self.scale_grad_by_freq,
                self.sparse,
            )
            # zdd: acculuate <bos> + <eos>
            # if input_.size(1) > 1:
            #     import pdb; pdb.set_trace() 
            #     output_parallel[0,0] = (output_parallel[0,0] + self.weight[128001]) / 2

            # Mask the output embedding.
            output_parallel[input_mask, :] = 0.0
            # Reduce across all the model parallel GPUs.
            output = reduce_from_model_parallel_region(output_parallel)
            return output
        return Emb_forward

    def Attn_factory(layerIdx):
        def Attn_forward(
            self,
            x: torch.Tensor,
            start_pos: int,
            freqs_cis: torch.Tensor,
            mask: Optional[torch.Tensor],
        ):
            bsz, seqlen, _ = x.shape
            xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)

            xq = xq.view(bsz, seqlen, self.n_local_heads, self.head_dim)
            xk = xk.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)
            xv = xv.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)

            xq, xk = apply_rotary_emb(xq, xk, freqs_cis=freqs_cis)

            self.cache_k = self.cache_k.to(xq)
            self.cache_v = self.cache_v.to(xq)

            self.cache_k[:bsz, start_pos : start_pos + seqlen] = xk
            self.cache_v[:bsz, start_pos : start_pos + seqlen] = xv

            keys = self.cache_k[:bsz, : start_pos + seqlen]
            values = self.cache_v[:bsz, : start_pos + seqlen]

            # repeat k/v heads if n_kv_heads < n_heads
            keys = repeat_kv(keys, self.n_rep)  # (bs, cache_len + seqlen, n_local_heads, head_dim)
            values = repeat_kv(values, self.n_rep)  # (bs, cache_len + seqlen, n_local_heads, head_dim)

            xq = xq.transpose(1, 2)	# (bs, n_local_heads, seqlen, head_dim)
            keys = keys.transpose(1, 2)	# (bs, n_local_heads, cache_len + seqlen, head_dim)
            values = values.transpose(1, 2)	# (bs, n_local_heads, cache_len + seqlen, head_dim)
            scores = torch.matmul(xq, keys.transpose(2, 3)) / math.sqrt(self.head_dim)
            if mask is not None:
                scores = scores + mask  # (bs, n_local_heads, seqlen, cache_len + seqlen)
            scores = F.softmax(scores.float(), dim=-1).type_as(xq)
            #import pdb; pdb.set_trace()
            output = torch.matmul(scores, values)  # (bs, n_local_heads, seqlen, head_dim)
            output = output.transpose(1, 2).contiguous().view(bsz, seqlen, -1)
            return self.wo(output)
        return Attn_forward


    def Ffn_factory(layerIdx):
        def Ffn_forward(self, x):
            #import pdb; pdb.set_trace()
            #return self.w2(F.silu(self.w1(x)) * self.w3(x))
            gate_up = self.w1(x)
            activation = F.silu(gate_up)
            retieve = activation * self.w3(x)
            x = self.w2(retieve)
            return x

        return Ffn_forward
    
    def Infer_factory(posIdx = 891200, layerIdx = 1024):
        def Infer_forward(self, tokens: torch.Tensor, start_pos: int):
            bsz, seqlen = tokens.shape
            hs = self.tok_embeddings(tokens)
            self.freqs_cis = self.freqs_cis.to(hs.device)
            freqs_cis = self.freqs_cis[start_pos : start_pos + seqlen]

            mask = None
            if seqlen > 1:
                mask = torch.full((seqlen, seqlen), float("-inf"), device=tokens.device)

                mask = torch.triu(mask, diagonal=1)

                mask = torch.hstack(
                [torch.zeros((seqlen, start_pos), device=tokens.device), mask]
                ).type_as(hs)

            #import pdb; pdb.set_trace()
            nLayer = 0
            for layer in self.layers:
                hs = layer(hs, start_pos, freqs_cis, mask)
                if start_pos >= posIdx and nLayer >= layerIdx:
                    #import pdb; pdb.set_trace()
                    break
                nLayer = nLayer + 1

            hs = self.norm(hs)
            output = self.output(hs).float()
            return output
        
        return Infer_forward


    embobj = model.model.tok_embeddings
    embobj.forward = MethodType(Emb_factory(), embobj)
    #import pdb; pdb.set_trace()   
    for i in range(len(model.model.layers)):
        attnobj = model.model.layers[i].attention
        attnobj.forward = MethodType(Attn_factory(i), attnobj)
        ffnobj = model.model.layers[i].feed_forward
        ffnobj.forward = MethodType(Ffn_factory(i), ffnobj)
    
    modelobj = model.model
    modelobj.forward = MethodType(Infer_factory(145, 25), modelobj)

    return model, sum1, sum2, sum3, over_zero, flat_zero
