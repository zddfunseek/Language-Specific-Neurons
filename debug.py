import argparse
import json
import os
import numpy as np
from types import MethodType

import torch
import subprocess
import signal
import torch.nn.functional as F
import vllm
from vllm import LLM, SamplingParams

is_oldver_vllm = (vllm.__version__ < '0.5.0')
is_llama = True

model = LLM(model='/home/dozhang/Llama-3/Meta-Llama-3-8B-Instruct', tensor_parallel_size=torch.cuda.device_count(), enforce_eager=True)
sampling_params = SamplingParams(temperature=0, repetition_penalty=1.1, max_tokens = 2048, stop = ["</s>", "<|eot_id|>"], logprobs=20)

def factory(mask):
    def llama_forward(self, x):
        gate_up, _ = self.gate_up_proj(x)  # b * l, 2i
        i = gate_up.size(-1)
        activation = F.silu(gate_up[..., : i // 2])
        #activation.index_fill_(-1, mask, 0)
        x = activation * gate_up[..., i // 2 :]
        x, _ = self.down_proj(x)
        return x

    def bloom_forward(self, x: torch.Tensor):
        x, _ = self.dense_h_to_4h(x)
        x = self.gelu_impl(x)
        x.index_fill_(2, mask, 0)
        x, _ = self.dense_4h_to_h(x)
        return x

    if is_llama:
        return llama_forward
    else:
        return bloom_forward


max_length = model.llm_engine.model_config.max_model_len
num_layers = model.llm_engine.model_config.hf_config.num_hidden_layers
intermediate_size = model.llm_engine.model_config.hf_config.intermediate_size if is_llama else model.llm_engine.model_config.hf_config.hidden_size * 4

activation_mask = torch.zeros(num_layers, intermediate_size, dtype=torch.int32).to('cuda')

for i, layer_mask in enumerate(activation_mask):
    if is_llama:
        if is_oldver_vllm:
            obj = model.llm_engine.driver_worker.model_runner.model.model.layers[i].mlp
        else:
            obj = model.llm_engine.model_executor.driver_worker.model_runner.model.model.layers[i].mlp
    else:
        obj = model.llm_engine.driver_worker.model_runner.model.transformer.h[i].mlp
    obj.forward = MethodType(factory(layer_mask.to('cuda')), obj)

import pdb; pdb.set_trace()
output =  model.generate(['1=A, 2=B, 3=?'], sampling_params)

# access logits
candidate_logits = []
for label in ["A", "B", "C", "D"]:
    try:
        candidate_logits.append(output[0].outputs[0].logprobs[0][model.llm_engine.tokenizer.tokenizer.convert_tokens_to_ids(label)].logprob)
    except:
        # If an option is not in the first 1000, set its logit to -100
        print("Warning: {} not found. Artificially adding log prob of -100.".format(label))
        candidate_logits.append(-100)

import pdb; pdb.set_trace()
candidate_logits = torch.tensor(candidate_logits).to(torch.float32)
probs = (
    torch.nn.functional.softmax(
        candidate_logits,
        dim=0,
    )
    .detach()
    .cpu()
    .numpy()
)
answer = {i: k for i, k in enumerate(["A", "B", "C", "D"])}[np.argmax(probs)]