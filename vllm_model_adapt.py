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
from types import MethodType
from typing import ClassVar, List, Optional, Sequence, Union, cast, overload

import vllm
from vllm import LLM, SamplingParams
from vllm.attention import AttentionMetadata
from typing import Iterable, List, Optional, Tuple
from vllm.distributed import (divide, get_tensor_model_parallel_rank,
                              get_tensor_model_parallel_world_size,
                              tensor_model_parallel_all_reduce)
from vllm.model_executor.layers.vocab_parallel_embedding import get_masked_input_and_mask
from vllm.outputs import EmbeddingRequestOutput, RequestOutput

from vllm.attention import Attention, AttentionMetadata
from vllm.config import CacheConfig, LoRAConfig
from vllm.distributed import (get_pp_group, get_tensor_model_parallel_rank,
                              get_tensor_model_parallel_world_size)
from vllm.model_executor.layers.activation import SiluAndMul
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import (MergedColumnParallelLinear,
                                               QKVParallelLinear,
                                               RowParallelLinear)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig)
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.layers.sampler import Sampler
from vllm.model_executor.layers.vocab_parallel_embedding import (
    DEFAULT_VOCAB_PADDING_SIZE, ParallelLMHead, VocabParallelEmbedding)
from vllm.model_executor.model_loader.weight_utils import (
    default_weight_loader, kv_cache_scales_loader)
from vllm.model_executor.sampling_metadata import SamplingMetadata
from vllm.sequence import IntermediateTensors, SamplerOutput
from vllm.utils import is_hip, print_warning_once

def Emb_factory(noise_scale=1e-1):
    def Embed_forward(self, input_):
        #import pdb; pdb.set_trace()
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
        if input_.size(0) > 1:
            #import pdb; pdb.set_trace()
            start=22
            end=output.size(0) - 4

            ### add random nosise
            # noise = torch.from_numpy(np.random.normal(loc=0.0, scale=noise_scale, size=output[start:end,:].shape)).to(torch.bfloat16).to('cuda')
            # #print('raw output:')
            # # print(output[...,:20])
            # # print('noise:')
            # # print(noise[...,:20])
            # output[start:end,:] = output[start:end,:] + noise

            ### add reversily
            # print('new output:')
            # print(output[...,:20])
            output[start:end,:] = torch.add(output[start:end,:], output[start:end,:].flip(dims=[0]))
            output[start:end,:] = output[start:end,:].flip(dims=[0])

            ### shuffle tokens
            # indices = torch.randperm(end - start) + start
            # print ('Random permutation of embeddings:\n\t')
            # print (indices)
            # shuffled_suboutput = output[indices, :]
            # output[start:end,:] = shuffled_suboutput

        return output
    return Embed_forward

def llama_adapt(model):
    max_length = model.llm_engine.model_config.max_model_len
    num_layers = model.llm_engine.model_config.hf_config.num_hidden_layers
    intermediate_size = model.llm_engine.model_config.hf_config.intermediate_size

    sum1 = torch.zeros(num_layers, intermediate_size).to('cuda')
    sum2 = torch.zeros(num_layers, intermediate_size).to('cuda')
    sum3 = torch.zeros(num_layers, intermediate_size).to('cuda')
    sum4 = torch.zeros(num_layers, intermediate_size).to('cuda')
    over_zero = torch.zeros(num_layers, intermediate_size, dtype=torch.int32).to('cuda')
    flat_zero = torch.zeros(num_layers, 2048, intermediate_size).to('cuda')
    activation_mask = torch.zeros(num_layers, intermediate_size, dtype=torch.int32).to('cuda')

    def Attn_factory():
        def Attn_forward(
            self,
            positions: torch.Tensor,
            hidden_states: torch.Tensor,
            kv_cache: torch.Tensor,
            attn_metadata: AttentionMetadata,
        ) -> torch.Tensor:
            global glob_posi
            glob_posi = positions[0]
            # if is_Debug and positions[0] == 23:
            #     import pdb; pdb.set_trace()
            #     global Flag
            #     Flag = True
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
            # if is_Debug and Flag:
            #     import pdb; pdb.set_trace()
            #import pdb; pdb.set_trace()
            sum1[idx, :] += activation.sum(dim=(0))
            sum2[idx, :] += activation.pow(2).sum(dim=(0))
            over_zero[idx, :] += (activation > 0).sum(dim=(0))
            flat_zero[idx, glob_posi:glob_posi + activation.size(0), :] = activation > 0.2
            x = activation * gate_up[..., i // 2 :]
            x, _ = self.down_proj(x)
            return x

        return llama_forward

    def Model_factory():
        def model_forward(
            self,
            input_ids: Optional[torch.Tensor],
            positions: torch.Tensor,
            kv_caches: List[torch.Tensor],
            attn_metadata: AttentionMetadata,
            intermediate_tensors: Optional[IntermediateTensors],
            inputs_embeds: Optional[torch.Tensor] = None,
        ) -> Union[torch.Tensor, IntermediateTensors]:
            #import pdb; pdb.set_trace()
            if get_pp_group().is_first_rank:
                if inputs_embeds is not None:
                    hidden_states = inputs_embeds
                else:
                    hidden_states = self.get_input_embeddings(input_ids)
                residual = None
            else:
                assert intermediate_tensors is not None
                hidden_states = intermediate_tensors["hidden_states"]
                residual = intermediate_tensors["residual"]

            for i in range(self.start_layer, self.end_layer):
                layer = self.layers[i]
                hidden_states, residual = layer(
                    positions,
                    hidden_states,
                    kv_caches[i - self.start_layer],
                    attn_metadata,
                    residual,
                )

            if not get_pp_group().is_last_rank:
                return IntermediateTensors({
                    "hidden_states": hidden_states,
                    "residual": residual
                })

            hidden_states, _ = self.norm(hidden_states, residual)
            return hidden_states
        return model_forward

    def CLM_factory():
        def clm_forward(
            self,
            input_ids: torch.Tensor,
            positions: torch.Tensor,
            kv_caches: List[torch.Tensor],
            attn_metadata: AttentionMetadata,
            intermediate_tensors: Optional[IntermediateTensors] = None,
        ) -> Union[torch.Tensor, IntermediateTensors]:
            #import pdb; pdb.set_trace()
            # indices = torch.randperm(input_ids.size(0))
            # print ('Random permutation of input_ids:\n\t')
            # print (indices)
            # input_ids = input_ids[indices]
            model_output = self.model(input_ids, positions, kv_caches,
                                    attn_metadata, intermediate_tensors)
            return model_output
        return clm_forward

    def Engine_factory():
        def _run_engine(self, *, use_tqdm: bool) -> List[Union[RequestOutput, EmbeddingRequestOutput]]:
            # Run the engine.
            outputs: List[Union[RequestOutput, EmbeddingRequestOutput]] = []
            total_in_toks = 0
            total_out_toks = 0
            while self.llm_engine.has_unfinished_requests():
                step_outputs = self.llm_engine.step()
                for output in step_outputs:
                    if output.finished:
                        outputs.append(output)
            # Sort the outputs by request ID.
            # This is necessary because some requests may be finished earlier than
            # its previous requests.
            return sorted(outputs, key=lambda x: int(x.request_id))
        return _run_engine

    #import pdb; pdb.set_trace()
    embobj = model.llm_engine.model_executor.driver_worker.model_runner.model.model.embed_tokens
    embobj.forward = MethodType(Emb_factory(), embobj)
    attnobj = model.llm_engine.model_executor.driver_worker.model_runner.model.model.layers[0].self_attn
    attnobj.forward = MethodType(Attn_factory(), attnobj)
    modelobj = model.llm_engine.model_executor.driver_worker.model_runner.model.model
    modelobj.forward = MethodType(Model_factory(), modelobj)
    clmobj = model.llm_engine.model_executor.driver_worker.model_runner.model
    clmobj.forward = MethodType(CLM_factory(), clmobj)
    model._run_engine = MethodType(Engine_factory(), model)


    for i, layer_mask in enumerate(activation_mask):
        obj = model.llm_engine.model_executor.driver_worker.model_runner.model.model.layers[i].mlp
        obj.forward = MethodType(Mlp_factory(i, layer_mask.to('cuda')), obj)
    
    return model, sum1, sum2, sum3, over_zero, flat_zero


def bloom_adapt(model):
    max_length = model.llm_engine.model_config.max_model_len
    num_layers = model.llm_engine.model_config.hf_config.num_hidden_layers
    intermediate_size = model.llm_engine.model_config.hf_config.hidden_size * 4

    sum1 = torch.zeros(num_layers, intermediate_size).to('cuda')
    sum2 = torch.zeros(num_layers, intermediate_size).to('cuda')
    sum3 = torch.zeros(num_layers, intermediate_size).to('cuda')
    sum4 = torch.zeros(num_layers, intermediate_size).to('cuda')
    over_zero = torch.zeros(num_layers, intermediate_size, dtype=torch.int32).to('cuda')
    flat_zero = torch.zeros(num_layers, 2048, intermediate_size).to('cuda')
    activation_mask = torch.zeros(num_layers, intermediate_size, dtype=torch.int32).to('cuda')

    def Emb_factory():
        def Embed_forward(self, input_):
            #import pdb; pdb.set_trace()
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
            if input_.size(0) > 1:
                #import pdb; pdb.set_trace()
                # noise = torch.from_numpy(np.random.normal(loc=0.0, scale=1e-2, size=output.shape)).to(torch.bfloat16).to('cuda')
                print('raw output:')
                # print(output[...,:20])
                # print('noise:')
                # print(noise[...,:20])
                # output = output + noise
                # print('new output:')
                # print(output[...,:20])
                # torch.add(output, output.flip(dims=[0]))
                # indices = torch.randperm(output.size(0))
                # print ('Random permutation of embeddings:\n\t')
                # print (indices)
                # output = output[indices, :]
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
            global glob_posi
            glob_posi = positions[0]
            # if is_Debug and positions[0] == 23:
            #     import pdb; pdb.set_trace()
            #     global Flag
            #     Flag = True
            qkv, _ = self.qkv_proj(hidden_states)
            q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
            q, k = self.rotary_emb(positions, q, k)
            attn_output = self.attn(q, k, v, kv_cache, attn_metadata)
            output, _ = self.o_proj(attn_output)
            return output
        return Attn_forward

    def Mlp_factory(idx, mask):
        def bloom_forward(self, x: torch.Tensor):
            x, _ = self.dense_h_to_4h(x)
            x = self.gelu_impl(x)
            activation = x.float()
            #x.index_fill_(2, mask, 0)
            over_zero[idx, :] += (activation > 0).sum(dim=(0,1))
            x, _ = self.dense_4h_to_h(x)
            return x

        return bloom_forward

    def Model_factory():
        def model_forward(
            self,
            input_ids: Optional[torch.Tensor],
            positions: torch.Tensor,
            kv_caches: List[torch.Tensor],
            attn_metadata: AttentionMetadata,
            intermediate_tensors: Optional[IntermediateTensors],
            inputs_embeds: Optional[torch.Tensor] = None,
        ) -> Union[torch.Tensor, IntermediateTensors]:
            #import pdb; pdb.set_trace()
            if get_pp_group().is_first_rank:
                if inputs_embeds is not None:
                    hidden_states = inputs_embeds
                else:
                    hidden_states = self.get_input_embeddings(input_ids)
                residual = None
            else:
                assert intermediate_tensors is not None
                hidden_states = intermediate_tensors["hidden_states"]
                residual = intermediate_tensors["residual"]

            for i in range(self.start_layer, self.end_layer):
                layer = self.layers[i]
                hidden_states, residual = layer(
                    positions,
                    hidden_states,
                    kv_caches[i - self.start_layer],
                    attn_metadata,
                    residual,
                )

            if not get_pp_group().is_last_rank:
                return IntermediateTensors({
                    "hidden_states": hidden_states,
                    "residual": residual
                })

            hidden_states, _ = self.norm(hidden_states, residual)
            return hidden_states
        return model_forward

    def CLM_factory():
        def clm_forward(
            self,
            input_ids: torch.Tensor,
            positions: torch.Tensor,
            kv_caches: List[torch.Tensor],
            attn_metadata: AttentionMetadata,
            intermediate_tensors: Optional[IntermediateTensors] = None,
        ) -> Union[torch.Tensor, IntermediateTensors]:
            #import pdb; pdb.set_trace()
            # indices = torch.randperm(input_ids.size(0))
            # print ('Random permutation of input_ids:\n\t')
            # print (indices)
            # input_ids = input_ids[indices]
            model_output = self.model(input_ids, positions, kv_caches,
                                    attn_metadata, intermediate_tensors)
            return model_output
        return clm_forward

    def Engine_factory():
        def _run_engine(self, *, use_tqdm: bool) -> List[Union[RequestOutput, EmbeddingRequestOutput]]:
            # Run the engine.
            outputs: List[Union[RequestOutput, EmbeddingRequestOutput]] = []
            total_in_toks = 0
            total_out_toks = 0
            while self.llm_engine.has_unfinished_requests():
                step_outputs = self.llm_engine.step()
                for output in step_outputs:
                    if output.finished:
                        outputs.append(output)
            # Sort the outputs by request ID.
            # This is necessary because some requests may be finished earlier than
            # its previous requests.
            return sorted(outputs, key=lambda x: int(x.request_id))
        return _run_engine

    #import pdb; pdb.set_trace()
    embobj = model.llm_engine.model_executor.driver_worker.model_runner.model.model.embed_tokens
    embobj.forward = MethodType(Emb_factory(), embobj)
    attnobj = model.llm_engine.model_executor.driver_worker.model_runner.model.model.layers[0].self_attn
    attnobj.forward = MethodType(Attn_factory(), attnobj)
    modelobj = model.llm_engine.model_executor.driver_worker.model_runner.model.model
    modelobj.forward = MethodType(Model_factory(), modelobj)
    clmobj = model.llm_engine.model_executor.driver_worker.model_runner.model
    clmobj.forward = MethodType(CLM_factory(), clmobj)
    model._run_engine = MethodType(Engine_factory(), model)

    for i, layer_mask in enumerate(activation_mask):
        obj = model.llm_engine.driver_worker.model_runner.model.transformer.h[i].mlp
        obj.forward = MethodType(Mlp_factory(i, layer_mask.to('cuda')), obj)