'''
Adapted from: /home/dozhang/miniconda3/envs/py31/lib/python3.10/site-packages/transformers/models/llama/modeling_llama.py
self_attention_type: LlamaSdpaAttention
'''

import argparse
import json
import os
import numpy as np
import logging
from tqdm import tqdm
import matplotlib.pyplot as plt
import random

import torch
import subprocess
import signal
import torch.nn.functional as F
from types import MethodType
from typing import List, Optional, Sequence, Union
from typing import List, Optional, Tuple

from transformers.cache_utils import Cache, DynamicCache, StaticCache
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
    QuestionAnsweringModelOutput,
    SequenceClassifierOutputWithPast,
    TokenClassifierOutput,
)
from transformers.models.llama.modeling_llama import (
    apply_rotary_pos_emb,
    repeat_kv,
)

globalBarLayer = -1
global_hidden_states = []
global_layerwise_hiddenstates = []

def bamboo(model, tokenizer, max_length, nBarLayer=60, valBarSim=0.99, nOutLayer = 3, nCheckLayer = 3, nWarmupTok = 90, globalBarLayer=-1, verbose=False):
    #import pdb; pdb.set_trace()
    num_layers = model.config.num_hidden_layers
    intermediate_size = model.config.intermediate_size
    hidden_size = model.config.hidden_size
  

    def Attn_factory():
        def Sdpa_forward(
            self,
            hidden_states: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_value: Optional[Cache] = None,
            output_attentions: bool = False,
            use_cache: bool = False,
            cache_position: Optional[torch.LongTensor] = None,
            position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # will become mandatory in v4.45
            **kwargs,
        ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
            #import pdb; pdb.set_trace()
            if output_attentions:
                # TODO: Improve this warning with e.g. `model.config.attn_implementation = "manual"` once this is implemented.
                logger.warning_once(
                    "LlamaModel is using LlamaSdpaAttention, but `torch.nn.functional.scaled_dot_product_attention` does not support `output_attentions=True`. Falling back to the manual attention implementation, "
                    'but specifying the manual implementation will be required from Transformers version v5.0.0 onwards. This warning can be removed using the argument `attn_implementation="eager"` when loading the model.'
                )
                return super().forward(
                    hidden_states=hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_value,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    cache_position=cache_position,
                    position_embeddings=position_embeddings,
                )

            bsz, q_len, _ = hidden_states.size()

            query_states = self.q_proj(hidden_states)
            key_states = self.k_proj(hidden_states)
            value_states = self.v_proj(hidden_states)

            query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
            key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
            value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

            if position_embeddings is None:
                logger.warning_once(
                    "The attention layers in this model are transitioning from computing the RoPE embeddings internally "
                    "through `position_ids` (2D tensor with the indexes of the tokens), to using externally computed "
                    "`position_embeddings` (Tuple of tensors, containing cos and sin). In v4.45 `position_ids` will be "
                    "removed and `position_embeddings` will be mandatory."
                )
                cos, sin = self.rotary_emb(value_states, position_ids)
            else:
                cos, sin = position_embeddings
            query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

            if past_key_value is not None:
                # sin and cos are specific to RoPE models; cache_position needed for the static cache
                cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
                key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

            key_states = repeat_kv(key_states, self.num_key_value_groups)
            value_states = repeat_kv(value_states, self.num_key_value_groups)

            causal_mask = attention_mask
            if attention_mask is not None:
                causal_mask = causal_mask[:, :, :, : key_states.shape[-2]]

            # SDPA with memory-efficient backend is currently (torch==2.1.2) bugged with non-contiguous inputs with custom attn_mask,
            # Reference: https://github.com/pytorch/pytorch/issues/112577.
            if query_states.device.type == "cuda" and causal_mask is not None:
                query_states = query_states.contiguous()
                key_states = key_states.contiguous()
                value_states = value_states.contiguous()

            # We dispatch to SDPA's Flash Attention or Efficient kernels via this `is_causal` if statement instead of an inline conditional assignment
            # in SDPA to support both torch.compile's dynamic shapes and full graph options. An inline conditional prevents dynamic shapes from compiling.
            is_causal = True if causal_mask is None and q_len > 1 else False

            attn_output = torch.nn.functional.scaled_dot_product_attention(
                query_states,
                key_states,
                value_states,
                attn_mask=causal_mask,
                dropout_p=self.attention_dropout if self.training else 0.0,
                is_causal=is_causal,
            )

            attn_output = attn_output.transpose(1, 2).contiguous()
            attn_output = attn_output.view(bsz, q_len, -1)

            attn_output = self.o_proj(attn_output)

            return attn_output, None, past_key_value

        return Sdpa_forward

    def Mlp_factory(idx, mask):
        def mlp_forward(self, x):
            if self.config.pretraining_tp > 1:
                slice = self.intermediate_size // self.config.pretraining_tp
                gate_proj_slices = self.gate_proj.weight.split(slice, dim=0)
                up_proj_slices = self.up_proj.weight.split(slice, dim=0)
                down_proj_slices = self.down_proj.weight.split(slice, dim=1)

                gate_proj = torch.cat(
                    [F.linear(x, gate_proj_slices[i]) for i in range(self.config.pretraining_tp)], dim=-1
                )
                up_proj = torch.cat([F.linear(x, up_proj_slices[i]) for i in range(self.config.pretraining_tp)], dim=-1)

                intermediate_states = (self.act_fn(gate_proj) * up_proj).split(slice, dim=2)
                down_proj = [
                    F.linear(intermediate_states[i], down_proj_slices[i]) for i in range(self.config.pretraining_tp)
                ]
                down_proj = sum(down_proj)
            else:
                down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))

            return down_proj

        return mlp_forward


    def custom_similarity(hidden_states_gather, hidden_states_next_gather, alpha=0.5):
        # Cosine similarity along the specified dimension
        cosine_sim = F.cosine_similarity(hidden_states_gather, hidden_states_next_gather, dim=-1)

        # Calculate magnitudes
        norm_gather = torch.norm(hidden_states_gather, dim=-1)
        norm_next_gather = torch.norm(hidden_states_next_gather, dim=-1)

        # Magnitude similarity (1 - normalized difference)
        mag_sim = 1 - torch.abs(norm_gather - norm_next_gather) / torch.max(norm_gather, norm_next_gather)

        # Combine cosine similarity and magnitude similarity
        combined_similarity = alpha * cosine_sim + (1 - alpha) * mag_sim

        return combined_similarity

    def adjusted_cosine_similarity(vec_a, vec_b):
        # Cosine similarity
        cosine_sim = F.cosine_similarity(vec_a, vec_b, dim=-1)
        
        # Norms of vectors
        norm_a = torch.norm(vec_a, dim=-1)
        norm_b = torch.norm(vec_b, dim=-1)
        
        # Calculate magnitude weighting factor
        magnitude_weight = torch.min(norm_a, norm_b) / torch.max(norm_a, norm_b)
        
        # Adjusted similarity
        adjusted_similarity = cosine_sim * magnitude_weight
        return adjusted_similarity

    globalNumDecodedLayer = torch.zeros(max_length).to('cuda')
    globalNumSkippedLayer = torch.zeros(max_length).to('cuda')
    def Model_factory():
        globalBarLayer = -1   
        global_hidden_states = []
        global_layerwise_hiddenstates = []
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

            global global_hidden_states
            global global_layerwise_hiddenstates
            ### Alternate 0-layer hiddenstates
            #import pdb; pdb.set_trace()
            if len(input_ids[-1]) > 1:
                hidden_states = inputs_embeds
            else:
                hidden_states = global_hidden_states[-1][:,-1:,:]
            hidden_states = inputs_embeds

            # create position embeddings to be shared across the decoder layers
            position_embeddings = self.rotary_emb(hidden_states, position_ids)

            # decoder layers
            all_hidden_states = () if output_hidden_states else None
            all_self_attns = () if output_attentions else None
            next_decoder_cache = None

            #import pdb; pdb.set_trace()
            global globalBarLayer
            numDecodedLayer = 0
            numSkippedLayer = 0
            idxLayer = 0
            nHighSimContinuousLayers = 0
            layerwise_hiddenstates=[]
            # nWarmupTok = 90
            # nOutLayer = 3
            # nBarLayer = 60 #(70B) #24 #24(7B)
            #valBarSim = 1.1 #0.99 #0.96 #0.975(7B)
            isActive = False
            for _i in range(len(self.layers) - nOutLayer):
            #for _i in range(len(self.layers)):
            #for _i in [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31]:
            #for _i in [0,1,2,3,4,5,5,6,6,7,7,8,8,9,9,10,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31]:
            #for _i in [0,1,2,3,4,5,5,5,6,6,6,7,7,7,8,8,8,9,9,9,10,10,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31]:
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
                # Assuming hidden_states and hidden_states_next are outputs from different GPUs
                hidden_states_gather = torch.nn.parallel.gather(hidden_states, target_device='cuda:0')
                hidden_states_next_gather = torch.nn.parallel.gather(hidden_states_next, target_device='cuda:0')
                # Compute cosine similarity
                cos_sim = F.cosine_similarity(hidden_states_gather, hidden_states_next_gather, dim=-1)
                #cos_sim = custom_similarity(hidden_states_gather, hidden_states_next_gather)
                #cos_sim = adjusted_cosine_similarity(hidden_states_gather, hidden_states_next_gather)
                cos_sim = torch.mean(cos_sim)
                hidden_states = hidden_states_next
                global_layerwise_hiddenstates.append(self.norm(hidden_states))
                idxLayer = idxLayer + 1
                numDecodedLayer = numDecodedLayer + 1
                if cos_sim > valBarSim:
                    nHighSimContinuousLayers = nHighSimContinuousLayers + 1
                else:
                    nHighSimContinuousLayers = 0
                
                #import pdb; pdb.set_trace()
                if nHighSimContinuousLayers >= nCheckLayer and len(input_ids[-1]) > 1 and globalBarLayer < 0:
                    #import pdb; pdb.set_trace()
                    globalBarLayer = max(idxLayer, nBarLayer)
                    if verbose:
                        print (f'\n*** Set BarLayer={globalBarLayer} based on prompt-layer similairty over {len(input_ids[-1])} tokens.\n')
                # layerwise_hiddenstates[idxLayer, position_ids[:]] = hidden_states_next[:]
                # layerwise_avgsim[:, idxLayer, position_ids[:]] = F.cosine_similarity(layerwise_hiddenstates[idxLayer, position_ids[:,:-1]].mean(dim=1), hidden_states_next, dim=-1)
                #if idxLayer >= nBarLayer and nHighSimContinuousLayers >= 3:
                #if nHighSimContinuousLayers >= 3:
                ### Only allow to layer-trucation on generation, instead of prompting stage
                if len(input_ids[-1]) < 2 and globalBarLayer > 0:
                    if idxLayer >= globalBarLayer and nHighSimContinuousLayers >= nCheckLayer and position_ids[-1][-1] > nWarmupTok:
                        #import pdb; pdb.set_trace()
                        if verbose:
                            print (f'@@@ Layer-truncation at #layer {idxLayer}/{num_layers}, #position {position_ids[-1][-1]}, #SimScore {cos_sim}, for token {tokenizer.convert_ids_to_tokens(input_ids[-1])}\n')
                        isActive = True
                        break               

            # fullfill the empty attention cache for the skipped layers with the highest-low-layer keys and values
            for _cacheIdx in range(idxLayer, len(self.layers) - nOutLayer):
                # Check if the key and value states are on the same device with previous key-value states, Move them to the same device if necessary
                pre_key_states = past_key_values[_cacheIdx][0][:, :, -1:, :]
                pre_value_states = past_key_values[_cacheIdx][1][:, :, -1:, :]
                key_states = past_key_values[_cacheIdx - 1][0][:, :, -1:, :]
                value_states = past_key_values[_cacheIdx - 1][1][:, :, -1:, :]
                if key_states.device != pre_key_states.device:
                    key_states = key_states.to(pre_key_states.device)
                if value_states.device != pre_value_states.device:
                    value_states = value_states.to(pre_value_states.device)
                past_key_values.update(key_states, value_states, _cacheIdx)
                numSkippedLayer = numSkippedLayer + 1
                global_layerwise_hiddenstates.append(global_layerwise_hiddenstates[-1])
                
            if not isActive and verbose:
                print (f'--- No truncation at #position {position_ids[-1][-1]}, #SimScore {cos_sim}, for token {tokenizer.convert_ids_to_tokens(input_ids[-1])}\n')

            # process last specified output layers
            for _lastIdx in range(len(self.layers) - nOutLayer, len(self.layers)):
                layer_outputs = self.layers[_lastIdx](
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
                global_layerwise_hiddenstates.append(self.norm(hidden_states))
                #print (f'+++ Perform on {position_ids[-1][-1]} position of {_lastIdx} layer +++\n')
                numDecodedLayer = numDecodedLayer + 1

            #import pdb; pdb.set_trace()
            globalNumDecodedLayer[position_ids[-1][-1]] = numDecodedLayer
            globalNumSkippedLayer[position_ids[-1][-1]] = numSkippedLayer

            hidden_states = self.norm(hidden_states)

            global_hidden_states.append(hidden_states)            

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

    def LLMHead_factory():
        def llmhead_forward(
            self,
            input_ids: torch.LongTensor = None,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            labels: Optional[torch.LongTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            cache_position: Optional[torch.LongTensor] = None,
            num_logits_to_keep: int = 0,
            **loss_kwargs,
        ) -> Union[Tuple, CausalLMOutputWithPast]:
            r"""
            Args:
                labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                    Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                    config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                    (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

                num_logits_to_keep (`int`, *optional*):
                    Calculate logits for the last `num_logits_to_keep` tokens. If `0`, calculate logits for all
                    `input_ids` (special case). Only last token logits are needed for generation, and calculating them only for that
                    token can save memory, which becomes pretty significant for long sequences or large vocabulary size.

            Returns:

            Example:

            ```python
            >>> from transformers import AutoTokenizer, LlamaForCausalLM

            >>> model = LlamaForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")
            >>> tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")

            >>> prompt = "Hey, are you conscious? Can you talk to me?"
            >>> inputs = tokenizer(prompt, return_tensors="pt")

            >>> # Generate
            >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
            >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
            "Hey, are you conscious? Can you talk to me?\nI'm not conscious, but I can talk to you."
            ```"""
            output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
            output_hidden_states = (
                output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
            )
            return_dict = return_dict if return_dict is not None else self.config.use_return_dict

            # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                cache_position=cache_position,
            )

            hidden_states = outputs[0]
            if self.config.pretraining_tp > 1:
                lm_head_slices = self.lm_head.weight.split(self.vocab_size // self.config.pretraining_tp, dim=0)
                logits = [F.linear(hidden_states, lm_head_slices[i]) for i in range(self.config.pretraining_tp)]
                logits = torch.cat(logits, dim=-1)
            else:
                # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
                logits = self.lm_head(hidden_states[:, -num_logits_to_keep:, :])

            ## Get lay-wise predicated tokens
            #import pdb; pdb.set_trace()
            for _idx, lhd in enumerate(global_layerwise_hiddenstates):
                l_logits = self.lm_head(lhd[:, :, :]).float()
                l_layer_tokens = torch.argmax(l_logits, dim=-1)
                #print (f'/// Predicted at #position {position_ids[-1][-1]} on #layer {_idx} with the token {tokenizer.convert_ids_to_tokens(l_layer_tokens[-1])}\n')
            avg_logits = torch.mean(torch.stack(global_layerwise_hiddenstates[-4:]),dim=0)
            avg_logits = self.lm_head(avg_logits).float()
            avg_layer_tokens = torch.argmax(avg_logits, dim=-1)
            #print (f'^^^ Average at #position {position_ids[-1][-1]} on #layer {_idx} with the token {tokenizer.convert_ids_to_tokens(avg_layer_tokens[-1])}\n')
            global_layerwise_hiddenstates.clear()
            logits = avg_logits



            loss = None
            if labels is not None:
                loss = self.loss_function(logits=logits, labels=labels, vocab_size=self.config.vocab_size, **loss_kwargs)

            if not return_dict:
                output = (logits,) + outputs[1:]
                return (loss,) + output if loss is not None else output

            return CausalLMOutputWithPast(
                loss=loss,
                logits=logits,
                past_key_values=outputs.past_key_values,
                hidden_states=outputs.hidden_states,
                attentions=outputs.attentions,
            )
        
        return llmhead_forward

    #import pdb; pdb.set_trace()
    # embobj = model.model.embed_tokens
    # embobj.forward = MethodType(Emb_factory(), embobj)
    #attnobj = model.model.layers[0].self_attn
    #attnobj.forward = MethodType(Attn_factory(), attnobj)
    modelobj = model.model
    modelobj.forward = MethodType(Model_factory(), modelobj)
    llmheadobj = model
    llmheadobj.forward = MethodType(LLMHead_factory(), llmheadobj)
    # model._run_engine = MethodType(Engine_factory(), model)



    # for i, layer_mask in enumerate(activation_mask):
    #     obj = model.model.layers[i].mlp
    #     obj.forward = MethodType(Mlp_factory(i, layer_mask.to('cuda')), obj)
    
    return model, globalNumDecodedLayer, globalNumSkippedLayer