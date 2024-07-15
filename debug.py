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

def factory(idx, mask):
    def llama_forward(self, x):
        gate_up, _ = self.gate_up_proj(x)  # b * l, 2i
        i = gate_up.size(-1)
        activation = F.silu(gate_up[..., : i // 2])
        #activation.index_fill_(-1, mask, 0)
        sum1[idx, :] += activation.sum(dim=(0,1))
        over_zero[idx, :] += (activation > 0).sum(dim=(0,1))
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

for i, layer_mask in enumerate(activation_mask):
    if is_llama:
        if is_oldver_vllm:
            obj = model.llm_engine.driver_worker.model_runner.model.model.layers[i].mlp
        else:
            obj = model.llm_engine.model_executor.driver_worker.model_runner.model.model.layers[i].mlp
    else:
        obj = model.llm_engine.driver_worker.model_runner.model.transformer.h[i].mlp
    obj.forward = MethodType(factory(i, layer_mask.to('cuda')), obj)

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

#Print_Prompt_logits(output)
#Print_Output_logits(output)
Print_Layer_Activation(output)