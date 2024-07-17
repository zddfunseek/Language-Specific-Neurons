import argparse
import json
import os
import numpy as np
from types import MethodType
import logging

import torch
import subprocess
import signal
import torch.nn.functional as F
import vllm
from vllm import LLM, SamplingParams
from vllm.attention import AttentionMetadata
from typing import Iterable, List, Optional, Tuple
from vllm.distributed import (divide, get_tensor_model_parallel_rank,
                              get_tensor_model_parallel_world_size,
                              tensor_model_parallel_all_reduce)


#logging.basicConfig(filename='debug.log', level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logging.basicConfig(level=logging.INFO, format='')
logger = logging.getLogger(__name__)
file_handler = logging.FileHandler('debug.log1')
file_handler.setLevel(logging.INFO)
logger.addHandler(file_handler)

is_oldver_vllm = (vllm.__version__ < '0.4.0')
is_llama = True

model = LLM(model='/home/dozhang/Llama-3/Meta-Llama-3-8B-Instruct', tensor_parallel_size=torch.cuda.device_count(), enforce_eager=True)
sampling_params = SamplingParams(temperature=0, repetition_penalty=1.1, max_tokens = 2048, stop = ["</s>", "<|eot_id|>", "<|end_of_text|>"], logprobs=5, prompt_logprobs=5)

max_length = model.llm_engine.model_config.max_model_len
num_layers = model.llm_engine.model_config.hf_config.num_hidden_layers
intermediate_size = model.llm_engine.model_config.hf_config.intermediate_size if is_llama else model.llm_engine.model_config.hf_config.hidden_size * 4

sum1 = torch.zeros(num_layers, intermediate_size).to('cuda')
sum2 = torch.zeros(num_layers, intermediate_size).to('cuda')
sum3 = torch.zeros(num_layers, intermediate_size).to('cuda')
sum4 = torch.zeros(num_layers, intermediate_size).to('cuda')
over_zero = torch.zeros(num_layers, intermediate_size, dtype=torch.int32).to('cuda')
activation_mask = torch.zeros(num_layers, intermediate_size, dtype=torch.int32).to('cuda')

def get_masked_input_and_mask(
        input_: torch.Tensor, org_vocab_start_index: int,
        org_vocab_end_index: int, num_org_vocab_padding: int,
        added_vocab_start_index: int,
        added_vocab_end_index: int) -> Tuple[torch.Tensor, torch.Tensor]:
    # torch.jit.script will fuse all of the pointwise ops below
    # into a single kernel, making it very fast
    org_vocab_mask = (input_ >= org_vocab_start_index) & (input_ <
                                                          org_vocab_end_index)
    added_vocab_mask = (input_ >= added_vocab_start_index) & (
        input_ < added_vocab_end_index)
    added_offset = added_vocab_start_index - (
        org_vocab_end_index - org_vocab_start_index) - num_org_vocab_padding
    valid_offset = (org_vocab_start_index *
                    org_vocab_mask) + (added_offset * added_vocab_mask)
    vocab_mask = org_vocab_mask | added_vocab_mask
    input_ = vocab_mask * (input_ - valid_offset)
    return input_, ~vocab_mask

def Emb_factory():
    def Embed_forward(self, input_):
        import pdb; pdb.set_trace()
        if self.tp_size > 1:
            # Build the mask.
            masked_input, input_mask = get_masked_input_and_mask(
                input_, self.shard_indices.org_vocab_start_index,
                self.shard_indices.org_vocab_end_index,
                self.shard_indices.num_org_vocab_padding,
                self.shard_indices.added_vocab_start_index,
                self.shard_indices.added_vocab_end_index)
        else:
            masked_input = input_
        # Get the embeddings.
        output_parallel = F.embedding(masked_input.long(), self.weight)
        # Mask the output embedding.
        if self.tp_size > 1:
            output_parallel.masked_fill_(input_mask.unsqueeze(-1), 0)
        # Reduce across all the model parallel GPUs.
        output = tensor_model_parallel_all_reduce(output_parallel)
        return output
    return Embed_forward

def Attn_factory():
    def Attn_forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: AttentionMetadata,
    ) -> torch.Tensor:
        import pdb; pdb.set_trace()
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        q, k = self.rotary_emb(positions, q, k)
        attn_output = self.attn(q, k, v, kv_cache, attn_metadata)
        output, _ = self.o_proj(attn_output)
        return output
    return Attn_forward

def Mlp_factory(idx, mask):
    def llama_forward(self, x):
        gate_up, _ = self.gate_up_proj(x)  # b * l, 2i
        i = gate_up.size(-1)
        activation = F.silu(gate_up[..., : i // 2])
        #activation.index_fill_(-1, mask, 0)
        # if idx ==0:
        #import pdb; pdb.set_trace()
        sum1[idx, :] += activation.sum(dim=(0))
        sum2[idx, :] += activation.pow(2).sum(dim=(0))
        over_zero[idx, :] += (activation > 0).sum(dim=(0))
        x = activation * gate_up[..., i // 2 :]
        x, _ = self.down_proj(x)
        return x

    def bloom_forward(self, x: torch.Tensor):
        x, _ = self.dense_h_to_4h(x)
        x = self.gelu_impl(x)
        activation = x.float()
        #x.index_fill_(2, mask, 0)
        over_zero[idx, :] += (activation > 0).sum(dim=(0,1))
        x, _ = self.dense_4h_to_h(x)
        return x

    if is_llama:
        return llama_forward
    else:
        return bloom_forward

#import pdb; pdb.set_trace()
embobj = model.llm_engine.model_executor.driver_worker.model_runner.model.model.embed_tokens
embobj.forward = MethodType(Emb_factory(), embobj)

attnobj = model.llm_engine.model_executor.driver_worker.model_runner.model.model.layers[0].self_attn
attnobj.forward = MethodType(Attn_factory(), attnobj)

for i, layer_mask in enumerate(activation_mask):
    #import pdb; pdb.set_trace()
    if is_llama:
        if is_oldver_vllm:
            obj = model.llm_engine.driver_worker.model_runner.model.model.layers[i].mlp
        else:
            obj = model.llm_engine.model_executor.driver_worker.model_runner.model.model.layers[i].mlp
    else:
        obj = model.llm_engine.driver_worker.model_runner.model.transformer.h[i].mlp
    obj.forward = MethodType(Mlp_factory(i, layer_mask.to('cuda')), obj)


thinking = "These equations don't follow the standard rules of arithmetic, so it looks like there's a hidden pattern or rule at play. Let's try to figure out the pattern based on the given examples:\n\n1. For 1 + 3 = 10:\n   - One possible interpretation is that the equation represents \( (1 + 3) \times 2 = 8 \), but this doesn't match 10.\n   - Another interpretation could be \( 1 + 3 + 6 = 10 \), where 6 might be an added constant, but this seems arbitrary.\n\n2. For 2 + 5 = 27:\n   - One approach is \( (2 + 5) \times 3 = 21 \), but this doesn't match 27.\n   - Another interpretation could be \( 2 + 5 + 20 = 27 \), where 20 is an added constant.\n\nGiven these interpretations, let's try a different approach to see if a consistent pattern emerges:\n   - Notice that the results seem to be significantly larger than the simple sum of the numbers.\n\nLet's consider the possibility that the equations might be based on a pattern involving multiplication:\n\n1. \( 1 + 3 = 10 \):\n   - \( 1 \times 3 + 3 \times 1 = 3 + 3 = 6 \) - this doesn't match 10.\n   - \( 1 + 3 = 4 \), but we need to reach 10.\n\n2. \( 2 + 5 = 27 \):\n   - \( 2 \times 5 + 5 \times 2 = 10 + 10 = 20 \) - this doesn't match 27.\n   - \( 2 + 5 = 7 \), but we need to reach 27.\n\nNow, let's try a pattern involving powers or exponents:\n\n1. \( 1 + 3 = 10 \):\n   - \( 1 + 3 = 4 \)\n   - Consider \( 4^2 - 6 = 10 \), where we square the sum and then subtract 6.\n\n2. \( 2 + 5 = 27 \):\n   - \( 2 + 5 = 7 \)\n   - Consider \( 7^2 - 22 = 27 \), where we square the sum and then subtract 22.\n\nThese adjustments seem arbitrary, but they suggest a possible pattern involving squaring:\n\nLet’s try to find a more systematic approach. Consider that each equation might be following a non-linear transformation:\n\n1. For \( 1 + 3 = 10 \):\n   - \( (1 + 3)^2 - 6 = 4^2 - 6 = 16 - 6 = 10 \).\n\n2. For \( 2 + 5 = 27 \):\n   - \( (2 + 5)^2 - 18 = 7^2 - 18 = 49 - 22 = 27 \).\n\nThis approach suggests a pattern of squaring the sum and then subtracting a specific number. Let’s apply this pattern to \( 3 + 4 \):\n\n- \( 3 + 4 = 7 \)\n- Square the sum: \( 7^2 = 49 \)\n- Determine the constant to subtract: For consistency, let’s subtract 22 as in the second example:\n  - \( 49 - 22 = 27 \)\n\nBut this doesn't match a consistent pattern. Let’s try another consistent subtraction:\n\nGiven the complexity, let’s use the pattern \( (a + b)^2 - (a + b) \):\n\n- For \( 1 + 3 = 10 \):\n  - \( (1 + 3)^2 - (1 + 3) = 4^2 - 4 = 16 - 6 = 10 \)."

template = "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nYou are a helpful AI assistant for travel tips and recommendations<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{instruction}<|eot_id|><|start_header_id|>assistant<|end_header_id|>"
query = f"if 1+3=10 and 2+5=27, then 3+4=? ### Some hints: {thinking} ###"
query = f"if 1+3=10 and 2+5=27, then 3+4="
#query = f"Who is the assassin of Trump?"
#query = template.format(instruction=query)
output =  model.generate([query], sampling_params)
#import pdb; pdb.set_trace()

def Access_Logits(output):
    # access logits
    candidate_logits = []
    for label in ["A", "B", "C", "D"]:
        try:
            candidate_logits.append(output[0].outputs[0].logprobs[0][model.llm_engine.tokenizer.tokenizer.convert_tokens_to_ids(label)].logprob)
        except:
            # If an option is not in the first 1000, set its logit to -100
            print("Warning: {} not found. Artificially adding log prob of -100.".format(label))
            candidate_logits.append(-100)

    #import pdb; pdb.set_trace()
    candidate_logits = torch.tensor(candidate_logits).to(torch.float32)
    probs = (torch.nn.functional.softmax(candidate_logits,dim=0,).detach().cpu().numpy())
    answer = {i: k for i, k in enumerate(["A", "B", "C", "D"])}[np.argmax(probs)]

def Print_Prompt_logits(output):
    #import pdb; pdb.set_trace()
    logger.info(f'Prompt: {output[0].prompt}')
    for idx, (prompt_tok_id, logit) in enumerate(zip(output[0].prompt_token_ids, output[0].prompt_logprobs)):
        logger.info(f"Position #{idx}:")
        logger.info(f'\tPrompt_tok: {model.llm_engine.tokenizer.tokenizer.convert_ids_to_tokens(prompt_tok_id)}')
        logger.info(f"\tTop_Logprobs:")
        if logit is None:
            continue
        for key, val in logit.items():
            logger.info(f'\t\t{repr(val.decoded_token):<20}\t{val.rank:6}\t{val.logprob:<20}\t{np.exp(val.logprob):<20}\t{key}')
    logger.info('\n')

def Print_Output_logits(output):
    logger.info(f'Output: {output[0].outputs[0].text}')
    for idx, (out_tok_id, logit) in enumerate(zip(output[0].outputs[0].token_ids, output[0].outputs[0].logprobs)):
        logger.info(f"Position #{idx}:")
        logger.info(f'\tDecode_tok: {model.llm_engine.tokenizer.tokenizer.convert_ids_to_tokens(out_tok_id)}')
        logger.info(f"\tTop_Logprobs:")
        for key, val in logit.items():
            logger.info(f'\t\t{repr(val.decoded_token):<20}\t{val.rank:6}\t{val.logprob:<20}\t{np.exp(val.logprob):<20}\t{key}')
    logger.info('\n')

def Print_Layer_Activation(output):
    logger.info(f"+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
    logger.info(f"Activations distribution")
    logger.info(f'Prompt: {output[0].prompt}')
    logger.info(f'Output: {output[0].outputs[0].text}')
    for idx in range(num_layers):
        logger.info(f"\tLayer#{idx:<5}\t{over_zero[idx]}")
        #logger.info(f"\tLayer#{idx:<5}\t{sum1[idx]}")
        logger.info(f"\tLayer#{idx:<5}\t{sum2[idx]}")

    logger.info("-------------------------------------------------------------")
    logger.info("Bin Information^^^^^")
    # 定义区间
    tokCount = len(output[0].prompt_token_ids) + len(output[0].outputs[0].token_ids)
    for idx in range(num_layers):
        # 使用NumPy的histogram函数计算每个区间的频数
        actival = sum2[idx].cpu()
        #import pdb; pdb.set_trace()
        bins = [b for b in np.linspace(torch.min(actival), torch.max(actival), 11)]
        hist, bin_edges = np.histogram(actival, bins=bins)
        # 计算每个区间的比例
        total_count = len(actival)
        proportions = hist / total_count
        logger.info(f"\tLayer#{idx:<5}")
        logger.info(f"\t\tbins:\t{bins}")
        logger.info(f"\t\tbin_edges\t{bin_edges}")
        logger.info(f"\t\thist:\t{hist}")
        logger.info(f"\t\tproportions:\t{proportions}")
        


#Print_Prompt_logits(output)
#Print_Output_logits(output)
Print_Layer_Activation(output)