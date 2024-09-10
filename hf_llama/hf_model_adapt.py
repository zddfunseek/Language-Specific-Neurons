'''
Adapted from: /home/dozhang/miniconda3/envs/py31/lib/python3.10/site-packages/transformers/models/llama/modeling_llama.py
'''

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

from transformers.cache_utils import Cache, DynamicCache, StaticCache
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
    QuestionAnsweringModelOutput,
    SequenceClassifierOutputWithPast,
    TokenClassifierOutput,
)


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

def hf_adapt(model):
    #max_length = model.config.rope_scaling['original_max_position_embeddings']
    #max_length = model.config.max_position_embeddings
    num_layers = model.config.num_hidden_layers
    intermediate_size = model.config.intermediate_size

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
            input_ids: torch.LongTensor = None,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            cache_position: Optional[torch.LongTensor] = None,
        ) -> Union[Tuple, BaseModelOutputWithPast]:
            output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
            output_hidden_states = (
                output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
            )
            use_cache = use_cache if use_cache is not None else self.config.use_cache
            return_dict = return_dict if return_dict is not None else self.config.use_return_dict

            if (input_ids is None) ^ (inputs_embeds is not None):
                raise ValueError(
                    "You cannot specify both input_ids and inputs_embeds at the same time, and must specify either one"
                )

            if self.gradient_checkpointing and self.training and use_cache:
                print(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`."
                )
                use_cache = False

            if inputs_embeds is None:
                inputs_embeds = self.embed_tokens(input_ids)

            return_legacy_cache = False
            if (
                use_cache and not isinstance(past_key_values, Cache) and not self.training
            ):  # kept for BC (non `Cache` `past_key_values` inputs)
                return_legacy_cache = True
                past_key_values = DynamicCache.from_legacy_cache(past_key_values)
                print(
                    "We detected that you are passing `past_key_values` as a tuple and this is deprecated and will be removed in v4.43. "
                    "Please use an appropriate `Cache` class (https://huggingface.co/docs/transformers/v4.41.3/en/internal/generation_utils#transformers.Cache)"
                )

            if cache_position is None:
                past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
                cache_position = torch.arange(
                    past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
                )
            if position_ids is None:
                position_ids = cache_position.unsqueeze(0)

            causal_mask = self._update_causal_mask(
                attention_mask, inputs_embeds, cache_position, past_key_values, output_attentions
            )
            hidden_states = inputs_embeds

            # create position embeddings to be shared across the decoder layers
            position_embeddings = self.rotary_emb(hidden_states, position_ids)

            # decoder layers
            all_hidden_states = () if output_hidden_states else None
            all_self_attns = () if output_attentions else None
            next_decoder_cache = None

            #import pdb; pdb.set_trace()
            idxLayer = 0
            nHighSimContinuousLayers = 0
            nWarmupTok = 90
            nOutLayer = 2
            nBarLayer = 24
            valBarSim = 0.96 #0.975
            for _i in range(len(self.layers) - nOutLayer):
            #for decoder_layer in self.layers:
                decoder_layer = self.layers[_i]
                if output_hidden_states:
                    all_hidden_states += (hidden_states,)

                if self.gradient_checkpointing and self.training:
                    layer_outputs = self._gradient_checkpointing_func(
                        decoder_layer.__call__,
                        hidden_states,
                        causal_mask,
                        position_ids,
                        past_key_values,
                        output_attentions,
                        use_cache,
                        cache_position,
                        position_embeddings,
                    )
                else:
                    layer_outputs = decoder_layer(
                        hidden_states,
                        attention_mask=causal_mask,
                        position_ids=position_ids,
                        past_key_value=past_key_values,
                        output_attentions=output_attentions,
                        use_cache=use_cache,
                        cache_position=cache_position,
                        position_embeddings=position_embeddings,
                    )

                if use_cache:
                    next_decoder_cache = layer_outputs[2 if output_attentions else 1]

                if output_attentions:
                    all_self_attns += (layer_outputs[1],)

                #import pdb; pdb.set_trace()
                hidden_states_next = layer_outputs[0]
                cos_sim = F.cosine_similarity(hidden_states, hidden_states_next, dim=-1)
                hidden_states = hidden_states_next
                if cos_sim[:,-1] > valBarSim:
                    nHighSimContinuousLayers = nHighSimContinuousLayers + 1
                else:
                    nHighSimContinuousLayers = 0
                
                if idxLayer >= nBarLayer and nHighSimContinuousLayers >= 3:
                #if idxLayer >= nBarLayer and nHighSimContinuousLayers >= 3 and position_ids[-1][-1] > nWarmupTok:
                    print (f'@@@ Start to trucate at {position_ids[-1][-1]} position of {idxLayer} layer @@@\n')
                    break
                idxLayer = idxLayer + 1
            
            # process last specified output layers
            for _i in range(len(self.layers) - nOutLayer, len(self.layers)):
                layer_outputs = self.layers[_i](
                        hidden_states,
                        attention_mask=causal_mask,
                        position_ids=position_ids,
                        past_key_value=past_key_values,
                        output_attentions=output_attentions,
                        use_cache=use_cache,
                        cache_position=cache_position,
                        position_embeddings=position_embeddings,
                    )
                hidden_states = layer_outputs[0]


            hidden_states = self.norm(hidden_states)

            # add hidden states from the last decoder layer
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            next_cache = next_decoder_cache if use_cache else None
            if return_legacy_cache:
                next_cache = next_cache.to_legacy_cache()

            if not return_dict:
                return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
            return BaseModelOutputWithPast(
                last_hidden_state=hidden_states,
                past_key_values=next_cache,
                hidden_states=all_hidden_states,
                attentions=all_self_attns,
            )

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
    # embobj = model.model.embed_tokens
    # embobj.forward = MethodType(Emb_factory(), embobj)
    # attnobj = model.model.layers[0].self_attn
    # attnobj.forward = MethodType(Attn_factory(), attnobj)
    modelobj = model.model
    modelobj.forward = MethodType(Model_factory(), modelobj)
    # clmobj = model
    # clmobj.forward = MethodType(CLM_factory(), clmobj)
    # model._run_engine = MethodType(Engine_factory(), model)


    # for i, layer_mask in enumerate(activation_mask):
    #     obj = model.model.layers[i].mlp
    #     obj.forward = MethodType(Mlp_factory(i, layer_mask.to('cuda')), obj)
    
    return model, sum1, sum2, sum3, over_zero, flat_zero
