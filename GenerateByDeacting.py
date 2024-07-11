import argparse
import json
import os
from types import MethodType

import torch
import subprocess
import signal
import torch.nn.functional as F
import vllm
from vllm import LLM, SamplingParams

answer_lang = {
    "zh": "请用中文回答。",
    "en": " Answer in English.",
    "fr": " Veuillez répondre en français.",
    "es": " Por favor responda en español.",
    "id": " Tolong dijawab dalam bahasa Indonesia.",
    "ja": "日本語で答えてください。",
    "vi": " Hãy trả lời bằng tiếng Việt.",
}

is_oldver_vllm = (vllm.__version__ < '0.5.0')

parser = argparse.ArgumentParser()
parser.add_argument("-m", "--model", type=str, default="/home/dozhang/Llama-3/Meta-Llama-3-8B-Instruct")
parser.add_argument("-t", "--taskname", type=str, default="gsm8k")
parser.add_argument("-a", "--activation_mask", type=str, default="")
args = parser.parse_args()
args.activation_mask = f'output/{args.taskname}/train.activations.llama-3-Instruct'

llama_3_inst_template = "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nYou are a helpful AI assistant for travel tips and recommendations<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{instruction}<|eot_id|><|start_header_id|>assistant<|end_header_id|>"

def Load_testdata():
    # process input data files
    input_file = '/home/dozhang/EvalLLM/output/gsm8k/Meta-Llama-3-8B-Instruct/predictions_GSM8K.jsonl'
    import gzip
    import json
    datainput = {}
    datainput['chosen'] = []
    datainput['rejected'] = []
    with open(input_file, 'rt', encoding='utf-8') as f_in:
        for line in f_in:
            json_obj = json.loads(line)
            item = {
                    'prompt': llama_3_inst_template.format(instruction=json_obj['prompt']),
                    'answer': json_obj['answer'],
                    'prediction': json_obj['prediction'],
                    'oriresult': json_obj['model_output'],
                    }
            if json_obj['answer'] == json_obj['prediction']:
                datainput['chosen'].append(item)
            else:
                datainput['rejected'].append(item)

    return datainput

def Load_testdata_gz():
    # process input data files
    input_file = '/home/dozhang/hh-rlhf/harmless-base/test.jsonl.gz'
    import gzip
    import json
    datainput = {}
    datainput['prompt'] = []
    datainput['chosen'] = []
    datainput['rejected'] = []
    with gzip.open(input_file, 'rt', encoding='utf-8') as f_in:
        for line in f_in:
            json_obj = json.loads(line)
            p1data = json_obj['chosen']
            p2data = json_obj['rejected']
            
            humanAchor = '\n\nHuman: '
            achor = '\n\nAssistant: '
            last_occurrence_index = p1data.rfind(achor)
            first_occurrence_index = p1data.index(achor)
            assert (p1data[:last_occurrence_index].strip() == p2data[:last_occurrence_index].strip(), f"{json_obj}")

            #datainput['prompt'].append(p1data[:last_occurrence_index].strip())
            datainput['prompt'].append(p1data[:first_occurrence_index][len(humanAchor):].strip())
            datainput['chosen'].append(p1data[last_occurrence_index + len(achor):].strip())
            datainput['rejected'].append(p2data[last_occurrence_index + len(achor):].strip())

    return datainput

def DestroyModel():
    # Run the nvidia-smi command with query options
    command = "nvidia-smi --query-compute-apps=pid --format=csv,noheader"
    result = subprocess.run(command.split(), stdout=subprocess.PIPE, text=True)

    # Get the output and split by newlines to get individual PIDs
    pids = result.stdout.strip().split('\n')
    os.kill(int(pids[0]), signal.SIGTERM)

testdata = Load_testdata()
activation_masks = torch.load(args.activation_mask)
#activation_masks = [None]

is_llama = bool(args.model.lower().find("llama") >= 0)

# if args.activation_mask:
#     activation_masks = torch.load(args.activation_mask)
#     activation_mask_name = args.activation_mask.split("/")[-1].split(".")
#     activation_mask_name = ".".join(activation_mask_name[1:])
# else:
#     activation_masks = [None]


output_folder = f"output/{args.taskname}/results"
os.makedirs(output_folder, exist_ok=True)

tasks = ["chosen", "rejected"]
for activation_mask, mask_lang in zip(activation_masks[:1], tasks[:1]):

    model = LLM(model=args.model, tensor_parallel_size=torch.cuda.device_count(), enforce_eager=True)
    sampling_params = SamplingParams(temperature=0, repetition_penalty=1.1, max_tokens = 2048, stop = ["</s>", "<|eot_id|>"])
    # import pdb; pdb.set_trace()
    # DestroyModel()
    # torch.cuda.empty_cache()
    # model = LLM(model=args.model, tensor_parallel_size=torch.cuda.device_count(), enforce_eager=True)
    # a = model.generate(['what is the earth'], sampling_params)
    # print (a)
    # import pdb; pdb.set_trace()

    if activation_mask:
        def factory(mask):
            def llama_forward(self, x):
                gate_up, _ = self.gate_up_proj(x)  # b * l, 2i
                i = gate_up.size(-1)
                activation = F.silu(gate_up[..., : i // 2])
                activation.index_fill_(-1, mask, 0)
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

        for i, layer_mask in enumerate(activation_mask):
            if is_llama:
                if is_oldver_vllm:
                    obj = model.llm_engine.driver_worker.model_runner.model.model.layers[i].mlp
                else:
                    obj = model.llm_engine.model_executor.driver_worker.model_runner.model.model.layers[i].mlp
            else:
                obj = model.llm_engine.driver_worker.model_runner.model.transformer.h[i].mlp
            obj.forward = MethodType(factory(layer_mask.to('cuda')), obj)
    
    
    for lang in tasks:    
        texts = [it['prompt'] for it in testdata[lang]]
        outputs = model.generate(texts, sampling_params)
        outputs = [o.outputs[0].text.strip() for o in outputs]

        if activation_mask:
            output_file = f"{output_folder}/{lang}.deactivate_by.{mask_lang}.jsonl.{str(is_oldver_vllm)}.template"
        else:
            output_file = f"{output_folder}/{lang}.llama3-base_normal.jsonl"

        results = []
        for t, o in zip(testdata[lang], outputs):
            out = {
                    "lang": lang,
                    "prompt": t['prompt'], 
                    'answer': t['answer'],
                    'prediction': t['prediction'],
                    'oriresult': t['oriresult'],
                    "newoutput": o
                  }
            results.append(out)

        with open(output_file, "w", encoding="utf-8") as f:
            f.write(json.dumps(results, indent=4, ensure_ascii=False) + "\n")
    
    model = None
