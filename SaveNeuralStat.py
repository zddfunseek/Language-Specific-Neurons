import argparse
from types import MethodType

import torch
from vllm import LLM, SamplingParams


parser = argparse.ArgumentParser()
parser.add_argument("-m", "--model", type=str, default="/home/dozhang/Llama-3/Meta-Llama-3-8B-Instruct")
parser.add_argument("-c", "--traindata", type=str, default="/home/dozhang/hh-rlhf/harmless-base/train.jsonl.gz")
args = parser.parse_args()

def Load_traindata():
    # process input data files
    input_file = args.traindata
    import gzip
    import json
    datainput = {}
    datainput['chosen'] = []
    datainput['rejected'] = []
    with gzip.open(input_file, 'rt', encoding='utf-8') as f_in:
        for line in f_in:
            json_obj = json.loads(line)
            p1data = json_obj['chosen']
            p2data = json_obj['rejected']
            datainput['chosen'].append(p1data)
            datainput['rejected'].append(p2data)
    return datainput

is_llama = bool(args.model.lower().find('llama') >= 0)
#model = LLM(model=args.model, tensor_parallel_size=torch.cuda.device_count(), enforce_eager=True)
model = LLM(model=args.model, tokenizer=args.model, tokenizer_mode="auto",  tensor_parallel_size=torch.cuda.device_count(), trust_remote_code=True)
tokenizer = model.llm_engine.tokenizer

max_length = model.llm_engine.model_config.max_model_len
num_layers = model.llm_engine.model_config.hf_config.num_hidden_layers
intermediate_size = model.llm_engine.model_config.hf_config.intermediate_size if is_llama else model.llm_engine.model_config.hf_config.hidden_size * 4

sum1 = torch.zeros(num_layers, intermediate_size).to('cuda')
sum2 = torch.zeros(num_layers, intermediate_size).to('cuda')
sum3 = torch.zeros(num_layers, intermediate_size).to('cuda')
sum4 = torch.zeros(num_layers, intermediate_size).to('cuda')
over_zero = torch.zeros(num_layers, intermediate_size, dtype=torch.int32).to('cuda')
num_activation = torch.zeros(num_layers, dtype=torch.int32).to('cuda')

def factory(idx):
    def llama_forward(self, x):
        gate_up, _ = self.gate_up_proj(x)  # b, l, 2i
        i = gate_up.size(-1)
        gate_up[:, :, : i // 2] = torch.nn.SiLU()(gate_up[:, :, : i // 2])
        activation = gate_up[:, :, : i // 2].float() # b, l, i
        sum1[idx, :] += activation.sum(dim=(0,1))
        sum2[idx, :] += activation.pow(2).sum(dim=(0,1))
        sum3[idx, :] += activation.pow(3).sum(dim=(0,1))
        sum4[idx, :] += activation.pow(4).sum(dim=(0,1))
        
        over_zero[idx, :] += (activation > 0).sum(dim=(0,1))
        num_activation[idx] += torch.tensor(activation.size(0) * activation.size(1))
        x = gate_up[:, :, : i // 2] * gate_up[:, :, i // 2 :]
        x, _ = self.down_proj(x)
        return x

    def bloom_forward(self, x: torch.Tensor):
        x, _ = self.dense_h_to_4h(x)
        x = self.gelu_impl(x)
        activation = x.float()
        sum1[idx, :] += activation.sum(dim=(0,1))
        sum2[idx, :] += activation.pow(2).sum(dim=(0,1))
        sum3[idx, :] += activation.pow(3).sum(dim=(0,1))
        sum4[idx, :] += activation.pow(4).sum(dim=(0,1))
        over_zero[idx, :] += (activation > 0).sum(dim=(0,1))
        num_activation[idx] += torch.tensor(activation.size(0) * activation.size(1))
        x, _ = self.dense_4h_to_h(x)
        return x

    if is_llama:
        return llama_forward
    else:
        return bloom_forward

# import pdb; pdb.set_trace()
for i in range(num_layers):
    if is_llama:
        obj = model.llm_engine.driver_worker.model_runner.model.model.layers[i].mlp
    else:
        obj = model.llm_engine.driver_worker.model_runner.model.transformer.h[i].mlp
    obj.forward = MethodType(factory(i), obj)

inputdata = Load_traindata()

for datatype in inputdata.keys():
    print (f'Processing {datatype} data ...')
    ## reset the statistics
    sum1 = torch.zeros(num_layers, intermediate_size).to('cuda')
    sum2 = torch.zeros(num_layers, intermediate_size).to('cuda')
    sum3 = torch.zeros(num_layers, intermediate_size).to('cuda')
    sum4 = torch.zeros(num_layers, intermediate_size).to('cuda')
    over_zero = torch.zeros(num_layers, intermediate_size, dtype=torch.int32).to('cuda')
    num_activation = torch.zeros(num_layers, dtype=torch.int32).to('cuda')

    input_ids = [tokenizer(prompt, truncation=False, add_special_tokens=False).input_ids[:max_length] for prompt in inputdata[datatype]]
    output = model.generate(prompt_token_ids=input_ids, sampling_params=SamplingParams(max_tokens=1, temperature=0, stop=["</s>"]))
    out_dict = dict(n=num_activation[0], sum1=sum1.to('cpu'), sum2=sum2.to('cpu'), sum3=sum3.to('cpu'), sum4=sum4.to('cpu'), over_zero=over_zero.to('cpu'))
    
    if is_llama:
        torch.save(out_dict, f'output/hh-rlhf/neuronstate.train_{datatype}.llama-3-inst')
    else:
        torch.save(out_dict, f'output/hh-rlhf/neuronstate.train_{datatype}.train.bloom-7b')
